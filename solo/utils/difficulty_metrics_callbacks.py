"""
Callback for tracking difficulty metrics during Temporal JEPA or MAE training.
"""

import os
import time
import inspect  # Added for debugging method signatures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from solo.data.custom.temporal_core50 import TemporalCore50
from solo.data.custom.core50 import Core50
from solo.utils.metrics import compute_shannon_entropy, compute_edge_density


class DifficultyMetricsCallback(Callback):
    """Callback that computes and logs difficulty metrics at regular intervals.
    
    This callback tracks how difficulty metrics evolve during training and
    analyzes their stability, allowing for real-time monitoring of the
    learning process. Supports both Temporal JEPA and Temporal MAE models,
    as well as standard MAE/JEPA models on single-image datasets.
    """
    
    def __init__(
        self,
        dataset_path: str,
        eval_frequency: int = 5,
        num_samples: int = 1000,
        compute_rank_stability: bool = True,
        save_metrics: bool = True,
        output_dir: Optional[str] = None,
        dataset_kwargs: Optional[Dict] = None,
        model_type: str = "jepa",  # Options: "jepa" or "mae"
    ):
        """Initialize the callback.
        
        Args:
            dataset_path (str): Path to the dataset.
            eval_frequency (int): Frequency (in epochs) to compute metrics.
            num_samples (int): Number of samples to use for evaluation.
            compute_rank_stability (bool): Whether to compute rank stability between checkpoints.
            save_metrics (bool): Whether to save metrics to disk.
            output_dir (Optional[str]): Directory to save metrics files.
            dataset_kwargs (Optional[Dict]): Kwargs for the dataset (e.g., backgrounds, time_window).
            model_type (str): Type of model being analyzed ("jepa" or "mae").
        """
        super().__init__()
        
        self.dataset_path = dataset_path
        self.eval_frequency = eval_frequency
        self.num_samples = num_samples
        self.compute_rank_stability = compute_rank_stability
        self.save_metrics = save_metrics
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}
        self.model_type = model_type.lower()
        
        # Determine if this is a temporal setting based on dataset_kwargs
        self.is_temporal = 'time_window' in self.dataset_kwargs
        
        # Validate model_type
        if self.model_type not in ["jepa", "mae"]:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'jepa' or 'mae'.")
        
        print(f"Initializing DifficultyMetricsCallback for {self.model_type.upper()} model")
        print(f"Temporal mode: {self.is_temporal}")
        
        if save_metrics:
            self.output_dir = Path(output_dir) if output_dir else Path("metrics_output")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store previous metrics for stability analysis
        self.metrics_history = {}
        self.sample_indices = None
        self.dataset = None
        self.device = None
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None):
        """Set up the callback.
        
        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The model.
            stage (Optional[str]): The stage (fit, validate, test).
        """
        if stage == "fit":
            # Define preprocessing for model input (Resize, ToTensor, Normalize)
            self.preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])
            
            # Keep dataset with identity transform for RAW images (classical metrics)
            raw_transform = lambda x: x # Standard transform returns single image
            if self.is_temporal:
                raw_transform = lambda x, y: (x, y) # Temporal transform returns pair
            
            # Extract backgrounds from kwargs
            backgrounds = self.dataset_kwargs.get("backgrounds", None)
            
            # Instantiate the correct dataset based on whether it's temporal
            if self.is_temporal:
                self.dataset = TemporalCore50(
                    h5_path=self.dataset_path,
                    transform=raw_transform, # Load raw PIL images
                    time_window=self.dataset_kwargs.get('time_window', 15), 
                    backgrounds=backgrounds, 
                )
            else:
                self.dataset = Core50(
                    h5_path=self.dataset_path,
                    transform=raw_transform, # Load raw PIL images
                    backgrounds=backgrounds, 
                )
            
            # Determine device
            self.device = pl_module.device
            
            # Randomly select indices ONCE to track
            if self.sample_indices is None: # Only select if not already selected
                dataset_size = len(self.dataset)
                print(f"DEBUG: DifficultyMetricsCallback - dataset_size: {dataset_size}, self.num_samples (from init): {self.num_samples}")
                if self.num_samples >= dataset_size:
                    self.sample_indices = np.arange(dataset_size)
                    self.num_samples = dataset_size
                    print(f"Tracking all {self.num_samples} samples from specified backgrounds.")
                else:
                    seed = 42 # Fix seed locally for reproducibility
                    rng = np.random.RandomState(seed)
                    self.sample_indices = rng.choice(
                        dataset_size, size=self.num_samples, replace=False
                    )
                    print(f"Selected fixed random subset of {len(self.sample_indices)} samples for difficulty metrics tracking.")
            else:
                 print(f"Using previously selected fixed subset of {len(self.sample_indices)} samples.")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Compute metrics at the end of validation epochs.
        
        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The model.
        """
        # Only run on rank 0 to avoid duplication in multi-GPU setups
        if hasattr(pl_module, 'global_rank') and pl_module.global_rank != 0:
            return
        
        current_epoch = trainer.current_epoch
        print(f"DEBUG: DifficultyMetrics - Start of on_validation_epoch_end. Current epoch from trainer: {current_epoch}")

        run_metrics_computation = False
        metrics_label = ""  # Initialize

        if trainer.sanity_checking:
            print(f"\nComputing initial difficulty metrics (during sanity check)...")
            metrics_label = "sanity_check"
            run_metrics_computation = True
        elif current_epoch % self.eval_frequency == 0:
            # This condition implies not sanity checking (due to elif)
            # and it's a designated evaluation epoch.
            # No separate print here; the try block below will announce the epoch.
            run_metrics_computation = True
            # metrics_label will be set inside the try block based on current_epoch
        else:
            # Skip if not at regular evaluation frequency and not during sanity check
            return

        if run_metrics_computation:
            # === START try-except block ===
            try:
                # Set metrics_label for regular epochs here, sanity_check label is already set
                if not trainer.sanity_checking:
                    metrics_label = f"epoch_{current_epoch}"
                    print(f"\nComputing difficulty metrics at epoch {current_epoch}...")
                
                print(f"DEBUG: DifficultyMetrics - Determined metrics_label: {metrics_label}") # DEBUG ADDED
                start_time = time.time()
                
                # Compute metrics based on model type
                metrics_df = self._compute_metrics(pl_module)
                
                # Log metrics summary (use actual epoch number)
                self._log_metrics_summary(trainer, metrics_df, current_epoch)
                
                # Analyze stability if we have previous metrics
                if self.compute_rank_stability and self.metrics_history:
                    stability_metrics = self._compute_stability(metrics_df, current_epoch)
                    self._log_stability_metrics(trainer, stability_metrics, current_epoch)
                
                # Store metrics for future stability analysis - always use actual epoch
                self.metrics_history[current_epoch] = metrics_df
                
                # Save metrics to disk with the appropriate label
                if self.save_metrics:
                    # Ensure metrics_label for saving is correctly epoch-based for regular runs
                    save_label = f"epoch_{current_epoch}" if not trainer.sanity_checking else "sanity_check"
                    metrics_file = self.output_dir / f"metrics_{save_label}.parquet"
                    rank = pl_module.global_rank if hasattr(pl_module, 'global_rank') else 'N/A' # Helper variable for rank
                    print(f"DEBUG: DifficultyMetrics - Preparing to save. Label: {save_label}, Epoch: {current_epoch}, File: {metrics_file}") # DEBUG ADDED
                    print(f"[Rank {rank}] Attempting to save metrics for {save_label} to {metrics_file}...") # DEBUG
                    try:
                        metrics_df.to_parquet(metrics_file)
                        print(f"[Rank {rank}] Successfully saved metrics to {metrics_file}") # DEBUG
                    except Exception as e_save:
                        print(f"[Rank {rank}] ERROR saving metrics to {metrics_file}: {e_save}") # DEBUG Log potential errors
                        import traceback
                        traceback.print_exc() # Print stack trace
                
                # Log completion time
                elapsed_time = time.time() - start_time
                print(f"Difficulty metrics computation completed in {elapsed_time:.2f} seconds")
            
            except Exception as e_main:
                # Distinguish error source if possible (sanity vs regular epoch)
                context_msg = "during sanity check" if trainer.sanity_checking else f"for epoch {current_epoch}"
                print(f"ERROR: An exception occurred during difficulty metrics processing {context_msg}:")
                import traceback
                traceback.print_exc()
            # === END try-except block ===
    
    def _extract_patches(self, img, patch_size):
        """Extract patches from an image.
        
        Args:
            img (torch.Tensor): Image tensor [B, C, H, W]
            patch_size (int): Size of patches
            
        Returns:
            torch.Tensor: Patches tensor [B, N, P*P*C]
        """
        B, C, H, W = img.shape
        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * patch_size ** 2)
        return patches
    
    def _compute_metrics(self, pl_module: pl.LightningModule) -> pd.DataFrame:
        """Compute difficulty metrics for selected samples. Handles both temporal and standard inputs.
        
        Args:
            pl_module (pl.LightningModule): The model.
            
        Returns:
            pd.DataFrame: DataFrame containing computed metrics.
        """
        # Initialize common metric lists
        error_values = []  # Latent_errors (JEPA) or recon_errors (MAE)
        variances = []
        entropies = []       # Entropy of the first/only image
        edge_densities = []  # Edge density of the first/only image
        entropy_diffs = []   # Only for temporal
        edge_diffs = []      # Only for temporal
        sample_ids = []
        
        error_column = "latent_error" if self.model_type == "jepa" else "reconstruction_error"
        
        # Set model to eval mode initially
        original_mode_is_training = pl_module.training
        pl_module.eval()
        
        with torch.no_grad(): # Keep no_grad for efficiency
            for idx in tqdm(self.sample_indices, desc=f"Computing {self.model_type.upper()} metrics (Temporal: {self.is_temporal})"):
                # Get RAW sample(s)
                sample_data, target = self.dataset[idx]
                
                if self.is_temporal:
                    raw_img1, raw_img2 = sample_data
                    # --- Preprocess for MODEL input --- 
                    img1_processed = self.preprocess(raw_img1).unsqueeze(0).to(self.device)
                    img2_processed = self.preprocess(raw_img2).unsqueeze(0).to(self.device)
                    model_input = [img1_processed, img2_processed] # Temporal models expect a list
                else:
                    raw_img1 = sample_data # Standard dataset returns single image
                    # --- Preprocess for MODEL input --- 
                    img1_processed = self.preprocess(raw_img1).unsqueeze(0).to(self.device)
                    model_input = img1_processed # Standard models expect a single tensor
                
                # --- Compute model-specific metrics USING PROCESSED images --- 
                if self.model_type == "jepa":
                    # JEPA logic (adapt based on whether it's temporal or not)
                    if self.is_temporal:
                        out1 = pl_module.forward(img1_processed)
                        z1 = out1["z"]
                        out2 = pl_module.momentum_forward(img2_processed)
                        z2 = out2["k"]
                        z_pred = pl_module.predictor(z1)
                        error = torch.mean((z_pred - z2) ** 2).item()
                        variance = torch.var(z_pred).item()
                    else: # Standard JEPA (predicting masked patches within the same image)
                        # Assuming standard JEPA forward returns necessary info
                        # This part needs to be adapted based on your standard JEPA implementation
                        # For now, setting placeholder values
                        print(f"Warning: Standard JEPA metric computation not fully implemented. Using NaN.")
                        error = float('nan')
                        variance = float('nan')
                        # Example adaptation:
                        # out = pl_module(model_input, compute_metrics=True) # Assuming it returns pred/target/mask
                        # if "pred" in out and "target" in out and "mask" in out:
                        #     loss = ((out["pred"] - out["target"])**2).mean(dim=-1)
                        #     error = (loss * out["mask"]).sum().item() / out["mask"].sum().item()
                        #     variance = torch.var(out["pred"]).item() # Or variance of target/latent

                else:  # MAE
                    try:
                        was_training = pl_module.training
                        if was_training: pl_module.eval()
                        
                        # Pass the correct input format (list for temporal, tensor for standard)
                        # Ensure the model's forward handles both formats or use specific forward methods
                        # The TemporalMAE forward already handles both via `isinstance(X, list)`
                        out = pl_module(model_input, compute_metrics=True)
                        
                        if was_training: pl_module.train()
                        
                        if "mask" in out and "pred" in out:
                            mask = out["mask"]
                            pred = out["pred"]
                            patch_size = pl_module._vit_patch_size # Assumes MAE model has this attr
                            
                            # Target depends on temporal or not
                            target_img_processed = img2_processed if self.is_temporal else img1_processed
                            target_patches = self._extract_patches(target_img_processed, patch_size)
                            
                            # Normalize target if needed (assuming norm_pix_loss is an attr)
                            if getattr(pl_module, 'norm_pix_loss', False):
                                mean = target_patches.mean(dim=-1, keepdim=True)
                                var = target_patches.var(dim=-1, keepdim=True)
                                target_patches = (target_patches - mean) / (var + 1e-6)**.5
                            
                            loss = ((pred - target_patches) ** 2).mean(dim=-1)  # [B, N]
                            error = (loss * mask).sum().item() / mask.sum().item()
                            variance = torch.var(pred).item()
                        else:
                            print(f"Warning: mask or pred not found in MAE output for sample {idx}. Keys: {out.keys()}")
                            error = float('nan')
                            variance = float('nan')
                    except Exception as e:
                        print(f"Error computing MAE metrics for sample {idx}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        error = float('nan')
                        variance = float('nan')
                 
                # --- Compute classical metrics using RAW images --- 
                img1_np = np.array(raw_img1) # Convert raw PIL to numpy
                entropy1 = compute_shannon_entropy(img1_np)
                edge1 = compute_edge_density(img1_np)
                
                # Compute diff metrics only if temporal
                if self.is_temporal:
                    img2_np = np.array(raw_img2)
                    entropy2 = compute_shannon_entropy(img2_np)
                    edge2 = compute_edge_density(img2_np)
                    entropy_diff = np.abs(entropy2 - entropy1)
                    edge_diff = np.abs(edge2 - edge1)
                else:
                    entropy_diff = float('nan')
                    edge_diff = float('nan')
                
                # Store metrics
                error_values.append(error)
                variances.append(variance)
                entropies.append(entropy1)
                edge_densities.append(edge1)
                if self.is_temporal:
                    entropy_diffs.append(entropy_diff)
                    edge_diffs.append(edge_diff)
                sample_ids.append(int(idx))
        
        # Restore original model mode if needed
        if original_mode_is_training:
            pl_module.train()
        
        # Create DataFrame
        metrics_data = {
            "sample_id": sample_ids,
            error_column: error_values,
            "variance": variances,
            "entropy": entropies,
            "edge_density": edge_densities,
        }
        # Add diff columns only if temporal
        if self.is_temporal:
            metrics_data["entropy_diff"] = entropy_diffs
            metrics_data["edge_diff"] = edge_diffs
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Add quartiles - handle potential NaN diff columns
        metric_cols = list(metrics_data.keys())
        metric_cols.remove("sample_id")
        
        for col in metric_cols:
            if metrics_df[col].isna().all(): # Skip if all values are NaN
                 metrics_df[f"{col}_quartile"] = float('nan')
                 print(f"Warning: All values for {col} are NaN. Skipping quartile computation.")
                 continue
            
            try:
                 # Use rank method='first' to handle ties consistently for qcut
                 metrics_df[f"{col}_quartile"] = pd.qcut(metrics_df[col].rank(method='first'), 4, labels=False, duplicates='drop')
            except ValueError as e:
                 # Handle cases where qcut fails (e.g., too few unique values)
                 print(f"Warning: Could not compute quartiles for {col}: {e}. Assigning NaN.")
                 metrics_df[f"{col}_quartile"] = float('nan')
        
        return metrics_df
    
    def _compute_stability(self, current_metrics: pd.DataFrame, current_epoch: int) -> Dict[str, Any]:
        """Compute stability metrics between current and previous metrics.
        
        Args:
            current_metrics (pd.DataFrame): Current metrics.
            current_epoch (int): Current epoch.
            
        Returns:
            Dict[str, Any]: Stability metrics.
        """
        # Find most recent previous metrics
        prev_epochs = sorted(self.metrics_history.keys())
        if not prev_epochs:
            return {}
        
        prev_epoch = prev_epochs[-1]
        prev_metrics = self.metrics_history[prev_epoch]
        
        # Merge ensuring suffixes are added
        merged = pd.merge(
            current_metrics,
            prev_metrics,
            on="sample_id",
            suffixes=("_current", "_prev"),
            how="inner" # Use inner join to only compare samples present in both
        )
        
        if merged.empty:
            print("Warning: No matching sample IDs found between current and previous metrics for stability analysis.")
            return {}
        
        error_column = "latent_error" if self.model_type == "jepa" else "reconstruction_error"
        # Define potential metric columns including temporal ones
        potential_metric_cols = [error_column, "variance", "entropy", "edge_density"]
        if self.is_temporal: # Only include diff metrics if relevant
            potential_metric_cols.extend(["entropy_diff", "edge_diff"])
        
        rank_correlations = {}
        quartile_churn = {}
        
        # Iterate through potential columns and check if they exist in the merged df
        for col in potential_metric_cols:
            current_col = f"{col}_current"
            prev_col = f"{col}_prev"
            quartile_current = f"{col}_quartile_current"
            quartile_prev = f"{col}_quartile_prev"
            
            # Check if base columns exist for rank correlation
            if current_col in merged.columns and prev_col in merged.columns:
                valid_mask = merged[[current_col, prev_col]].notna().all(axis=1)
                if valid_mask.sum() > 1:
                    rank_corr, _ = spearmanr(
                        merged.loc[valid_mask, current_col],
                        merged.loc[valid_mask, prev_col]
                    )
                    rank_correlations[col] = rank_corr
                else:
                    rank_correlations[col] = float('nan')
            
            # Check if quartile columns exist for churn analysis
            if quartile_current in merged.columns and quartile_prev in merged.columns:
                 valid_mask = merged[[quartile_current, quartile_prev]].notna().all(axis=1)
                 if valid_mask.sum() > 0:
                     changed = (merged.loc[valid_mask, quartile_current] !=
                                merged.loc[valid_mask, quartile_prev]).mean() * 100
                     quartile_churn[col] = changed
                 else:
                     quartile_churn[col] = float('nan')
        
        return {
            "prev_epoch": prev_epoch,
            "current_epoch": current_epoch,
            "rank_correlations": rank_correlations,
            "quartile_churn": quartile_churn,
        }
    
    def _log_metrics_summary(self, trainer: Optional[pl.Trainer], metrics_df: pd.DataFrame, epoch: int):
        """Log metrics summary to the logger.
        
        Args:
            trainer (pl.Trainer): The trainer instance.
            metrics_df (pd.DataFrame): Metrics DataFrame.
            epoch (int): Current epoch.
        """
        error_column = "latent_error" if self.model_type == "jepa" else "reconstruction_error"
        # Define potential metric columns
        potential_metric_cols = [error_column, "variance", "entropy", "edge_density"]
        if self.is_temporal: # Only include diff metrics if relevant
            potential_metric_cols.extend(["entropy_diff", "edge_diff"])
        
        summary = {}
        for col in potential_metric_cols:
            if col in metrics_df.columns: # Check if column actually exists
                valid_values = metrics_df[col].dropna()
                if len(valid_values) > 0:
                    summary[f"difficulty_metrics/{col}/mean"] = valid_values.mean()
                    summary[f"difficulty_metrics/{col}/std"] = valid_values.std()
                    summary[f"difficulty_metrics/{col}/min"] = valid_values.min()
                    summary[f"difficulty_metrics/{col}/max"] = valid_values.max()
                    summary[f"difficulty_metrics/{col}/median"] = valid_values.median()
        
        if trainer is not None and trainer.logger:
            trainer.logger.log_metrics(summary, step=epoch)
        else: # Handle manual calls where trainer might be None
            # Print summary if logger isn't available
            print(f"--- Metrics Summary (Epoch {epoch}) ---")
            for k, v in summary.items():
                print(f"{k}: {v:.4f}")
            print("--- End Summary ---")
    
    def _log_stability_metrics(self, trainer: Optional[pl.Trainer], stability_metrics: Dict[str, Any], epoch: int):
        """Log stability metrics to the logger.
        
        Args:
            trainer (pl.Trainer): The trainer instance.
            stability_metrics (Dict[str, Any]): Stability metrics.
            epoch (int): Current epoch.
        """
        if not stability_metrics:
            return
        
        rank_correlations = stability_metrics.get("rank_correlations", {})
        quartile_churn = stability_metrics.get("quartile_churn", {})
        prev_epoch = stability_metrics.get("prev_epoch", "N/A")
        
        logged_metrics = {}
        # Log rank correlations (handle potential NaNs)
        for metric, corr in rank_correlations.items():
             logged_metrics[f"stability/{metric}/rank_correlation"] = corr if not np.isnan(corr) else None
        # Log quartile churn (handle potential NaNs)
        for metric, churn in quartile_churn.items():
             logged_metrics[f"stability/{metric}/quartile_churn"] = churn if not np.isnan(churn) else None
        
        # Compute averages excluding NaNs/Nones
        corr_values = [v for v in rank_correlations.values() if v is not None and not np.isnan(v)]
        churn_values = [v for v in quartile_churn.values() if v is not None and not np.isnan(v)]
        
        avg_rank_corr = np.mean(corr_values) if corr_values else float('nan')
        avg_quartile_churn = np.mean(churn_values) if churn_values else float('nan')
        
        logged_metrics["stability/avg_rank_correlation"] = avg_rank_corr if not np.isnan(avg_rank_corr) else None
        logged_metrics["stability/avg_quartile_churn"] = avg_quartile_churn if not np.isnan(avg_quartile_churn) else None
        
        print(f"\nStability metrics between epochs {prev_epoch} and {epoch}:")
        if logged_metrics.get("stability/avg_rank_correlation") is not None:
             print(f"Average rank correlation: {logged_metrics['stability/avg_rank_correlation']:.4f}")
        else:
             print("Average rank correlation: N/A")
        if logged_metrics.get("stability/avg_quartile_churn") is not None:
             print(f"Average quartile churn: {logged_metrics['stability/avg_quartile_churn']:.2f}%")
        else:
             print("Average quartile churn: N/A")
        
        if trainer is not None and trainer.logger:
            # Filter out None values before logging
            metrics_to_log = {k: v for k, v in logged_metrics.items() if v is not None}
            trainer.logger.log_metrics(metrics_to_log, step=epoch)
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate final metrics analysis at the end of training.
        
        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The model.
        """
        # Only run on rank 0
        if hasattr(pl_module, 'global_rank') and pl_module.global_rank != 0:
            return
            
        # Check if we should run analysis at all
        should_run_analysis = (self.save_metrics and self.metrics_history) or \
                              (not self.save_metrics and self.metrics_history and self.sample_indices is not None)
        
        if not should_run_analysis:
             print("Skipping final metrics analysis: Conditions not met (e.g., no history, no samples, or save_metrics=False without history).")
             return
        
        print("\nGenerating final metrics analysis...")
        final_epoch = trainer.current_epoch
        
        # Check if metrics for the final epoch were already computed and should be saved
        if final_epoch not in self.metrics_history:
            # Compute final metrics if they weren't computed during the last validation step
            print(f"Computing metrics for final epoch ({final_epoch})...")
            final_metrics = self._compute_metrics(pl_module)
            self.metrics_history[final_epoch] = final_metrics
            
            # Save final metrics only if they were just computed and save_metrics is True
            if self.save_metrics:
                 final_metrics_file = self.output_dir / f"metrics_epoch_{final_epoch}.parquet"
                 try:
                     final_metrics.to_parquet(final_metrics_file)
                     print(f"Saved final metrics to {final_metrics_file}")
                 except Exception as e:
                     print(f"ERROR saving final metrics to {final_metrics_file}: {e}")
                     import traceback
                     traceback.print_exc()
        else:
             print(f"Metrics for final epoch ({final_epoch}) already computed. Skipping recomputation.")
        
        # Analyze metrics evolution using all available history, regardless of saving
        self._analyze_metrics_evolution()
        
        print("Final metrics analysis completed.")
    
    def _analyze_metrics_evolution(self):
        """Analyze how metrics evolved throughout training."""
        if not self.metrics_history:
            print("Skipping metrics evolution analysis: No history available.")
            return
        
        epochs = sorted(self.metrics_history.keys())
        if len(epochs) <= 1:
            print("Skipping metrics evolution analysis: Only one epoch's worth of data.")
            return
        
        evolution_dir = self.output_dir / "evolution"
        evolution_dir.mkdir(exist_ok=True)
        
        error_column = "latent_error" if self.model_type == "jepa" else "reconstruction_error"
        potential_metric_cols = [error_column, "variance", "entropy", "edge_density"]
        # Check if temporal diff metrics likely exist in *any* history entry
        has_temporal_metrics = any(
            'entropy_diff' in df.columns for df in self.metrics_history.values()
        )
        if has_temporal_metrics:
            potential_metric_cols.extend(["entropy_diff", "edge_diff"])
        
        print(f"Analyzing evolution for metrics: {potential_metric_cols}")
        
        for metric in potential_metric_cols:
            # Check if metric exists in *all* relevant dataframes
            epochs_with_metric = [ep for ep in epochs if metric in self.metrics_history[ep].columns]
            if len(epochs_with_metric) <= 1:
                print(f"Skipping evolution analysis for {metric} - present in <= 1 epoch(s)")
                continue
            
            n_epochs = len(epochs_with_metric)
            corr_matrix = np.full((n_epochs, n_epochs), np.nan) # Initialize with NaN
            
            for i, epoch_i in enumerate(epochs_with_metric):
                metrics_i = self.metrics_history[epoch_i]
                for j, epoch_j in enumerate(epochs_with_metric):
                    if i > j: continue # Matrix is symmetric, only compute lower triangle + diagonal
                    
                    metrics_j = self.metrics_history[epoch_j]
                    merged = pd.merge(
                        metrics_i[["sample_id", metric]],
                        metrics_j[["sample_id", metric]],
                        on="sample_id", suffixes=("_i", "_j"), how="inner"
                    )
                    if merged.empty: continue
                    
                    valid_mask = merged[[f"{metric}_i", f"{metric}_j"]].notna().all(axis=1)
                    if valid_mask.sum() > 1:
                        corr, _ = spearmanr(
                            merged.loc[valid_mask, f"{metric}_i"],
                            merged.loc[valid_mask, f"{metric}_j"]
                        )
                        corr_matrix[i, j] = corr
                        if i != j: corr_matrix[j, i] = corr # Fill symmetric part
                    # Diagonal is always 1 (correlation with self)
                    if i == j: corr_matrix[i,i] = 1.0
            
            np.save(evolution_dir / f"{metric}_rank_correlation_matrix.npy", corr_matrix)
            
            # Save corresponding epochs for the matrix
            with open(evolution_dir / f"{metric}_correlation_epochs.txt", "w") as f:
                 for epoch in epochs_with_metric: f.write(f"{epoch}\\n")
            
            # --- Quartile Churn Analysis ---
            quartile_col = f"{metric}_quartile"
            epochs_with_quartile = [ep for ep in epochs_with_metric if quartile_col in self.metrics_history[ep].columns]
            
            if len(epochs_with_quartile) <=1:
                print(f"Skipping quartile churn analysis for {metric} - quartiles present in <=1 epoch(s)")
                continue
            
            churn_data = []
            first_epoch_with_quartile = epochs_with_quartile[0]
            first_metrics = self.metrics_history[first_epoch_with_quartile]
            
            for epoch in epochs_with_quartile[1:]: # Compare subsequent epochs to the first
                current_metrics = self.metrics_history[epoch]
                merged = pd.merge(
                    first_metrics[["sample_id", quartile_col]],
                    current_metrics[["sample_id", quartile_col]],
                    on="sample_id", suffixes=("_first", "_current"), how="inner"
                )
                if merged.empty: continue
                
                valid_mask = merged[[f"{quartile_col}_first", f"{quartile_col}_current"]].notna().all(axis=1)
                if valid_mask.sum() > 0:
                    churn = (merged.loc[valid_mask, f"{quartile_col}_first"] !=
                             merged.loc[valid_mask, f"{quartile_col}_current"]).mean() * 100
                    churn_data.append({"epoch": epoch, "churn_from_first": churn})
            
            if churn_data:
                pd.DataFrame(churn_data).to_csv(evolution_dir / f"{metric}_quartile_churn.csv", index=False)
        
        # --- Final Summary ---
        summary_file = evolution_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Metrics evolution analysis for {self.model_type.upper()} model (Temporal: {has_temporal_metrics})\\n")
            f.write(f"Epochs with metrics: {epochs}\\n")
            f.write(f"Number of samples tracked: {self.num_samples}\\n")
            f.write("\\n")
            
            # Stability between first and last recorded epochs
            if len(epochs) > 1:
                first_epoch = epochs[0]
                last_epoch = epochs[-1]
                first_metrics = self.metrics_history[first_epoch]
                last_metrics = self.metrics_history[last_epoch]
                
                f.write(f"Stability analysis between first epoch ({first_epoch}) and last epoch ({last_epoch}):\\n")
                
                merged_first_last = pd.merge(first_metrics, last_metrics, on="sample_id", suffixes=("_first", "_last"), how="inner")
                
                if not merged_first_last.empty:
                    for metric in potential_metric_cols:
                        metric_first = f"{metric}_first"
                        metric_last = f"{metric}_last"
                        quartile_first = f"{metric}_quartile_first"
                        quartile_last = f"{metric}_quartile_last"
                        
                        # Check if metric columns exist
                        if metric_first in merged_first_last.columns and metric_last in merged_first_last.columns:
                            f.write(f"  {metric}:\\n")
                            valid_mask_corr = merged_first_last[[metric_first, metric_last]].notna().all(axis=1)
                            if valid_mask_corr.sum() > 1:
                                corr, _ = spearmanr(merged_first_last.loc[valid_mask_corr, metric_first], merged_first_last.loc[valid_mask_corr, metric_last])
                                f.write(f"    Rank correlation: {corr:.4f}\\n")
                            else:
                                f.write("    Rank correlation: N/A (insufficient data)\\n")
                            
                            # Check if quartile columns exist
                            if quartile_first in merged_first_last.columns and quartile_last in merged_first_last.columns:
                                 valid_mask_churn = merged_first_last[[quartile_first, quartile_last]].notna().all(axis=1)
                                 if valid_mask_churn.sum() > 0:
                                     churn = (merged_first_last.loc[valid_mask_churn, quartile_first] != merged_first_last.loc[valid_mask_churn, quartile_last]).mean() * 100
                                     f.write(f"    Quartile churn: {churn:.2f}%\\n")
                                 else:
                                     f.write("    Quartile churn: N/A (insufficient data)\\n")
                            else:
                                 f.write("    Quartile churn: N/A (quartiles missing)\\n")
                        # else: # Optionally report missing metric
                        #     f.write(f"  {metric}: Not found in first/last epoch data.\\n")
                else:
                    f.write("  Could not compare first and last epochs (no common samples or data missing).\\n")
            
            f.write("\\nDetailed evolution data saved in separate files (correlation matrices, churn CSVs).\\n")
        print(f"Saved evolution summary to {summary_file}")
    
    # Add a new method to compute metrics for a specific epoch (minor updates for flexibility)
    def compute_metrics_for_epoch(self, pl_module: pl.LightningModule, epoch: int):
        """Manually compute metrics for a specific epoch.
        
        This method bypasses the automatic validation epoch mechanism and
        allows direct computation of metrics for any epoch number.
        
        Args:
            pl_module (pl.LightningModule): The model.
            epoch (int): The epoch number to use.
        """
        print(f"\nManually computing difficulty metrics for epoch {epoch}...")
        
        start_time = time.time()
        
        # Compute metrics
        metrics_df = self._compute_metrics(pl_module)
        
        # Log metrics summary (pass None for trainer if not available)
        self._log_metrics_summary(None, metrics_df, epoch) # No trainer needed here
        
        # Analyze stability (pass None for trainer if not available)
        if self.compute_rank_stability and self.metrics_history:
            stability_metrics = self._compute_stability(metrics_df, epoch)
            self._log_stability_metrics(None, stability_metrics, epoch) # No trainer needed here
        
        # Store metrics
        self.metrics_history[epoch] = metrics_df
        
        # Save metrics
        if self.save_metrics:
            metrics_file = self.output_dir / f"metrics_epoch_{epoch}.parquet"
            print(f"Saving metrics to {metrics_file}")
            try:
                metrics_df.to_parquet(metrics_file)
                print(f"Successfully saved metrics to {metrics_file}")
            except Exception as e:
                 print(f"ERROR saving metrics to {metrics_file}: {e}")
                 import traceback
                 traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        print(f"Difficulty metrics computation completed in {elapsed_time:.2f} seconds")
        
        return metrics_df