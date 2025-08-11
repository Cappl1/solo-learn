"""
Batch script to run difficulty analysis across multiple checkpoints and track threshold evolution.
Usage: python batch_difficulty_analysis.py
"""

import os
import sys
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import yaml
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Add imports for direct integration with main_difficulty_eval
import torch
from omegaconf import OmegaConf

# Import the DifficultyEvaluator directly
try:
    from main_difficulty_eval import DifficultyEvaluator
    from solo.data.classification_dataloader import prepare_data
    # Force subprocess method due to config compatibility issues with direct import
    DIRECT_IMPORT_AVAILABLE = False  # Was True, now False to force subprocess method
    print("Note: Forcing subprocess method for reliability")
except ImportError:
    print("Warning: Could not import DifficultyEvaluator directly. Will use subprocess.")
    DIRECT_IMPORT_AVAILABLE = False


class BatchDifficultyAnalyzer:
    """Analyzes difficulty across multiple training epochs."""
    
    def __init__(self, base_dir: str, output_dir: str = "batch_difficulty_analysis"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for all results
        self.epoch_results = {}
        
    def run_single_difficulty_analysis_direct(self, 
                                             backbone_checkpoint: str,
                                             linear_checkpoint: str,
                                             config_path: str,
                                             epoch_num: int,
                                             difficulty_method: str = 'entropy',
                                             batch_size: int = 64) -> Optional[Dict]:
        """Run difficulty analysis directly using DifficultyEvaluator class."""
        
        if not DIRECT_IMPORT_AVAILABLE:
            return self.run_single_difficulty_analysis_subprocess(
                backbone_checkpoint, linear_checkpoint, config_path, 
                epoch_num, difficulty_method, batch_size
            )
        
        # Create output directory for this epoch
        epoch_output_dir = self.output_dir / f"epoch_{epoch_num:02d}" / difficulty_method
        epoch_output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Running {difficulty_method} analysis for epoch {epoch_num}...")
        
        try:
            # Initialize evaluator
            evaluator = DifficultyEvaluator(
                model_path=backbone_checkpoint,
                linear_path=linear_checkpoint,
                cfg_path=config_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                difficulty_method=difficulty_method
            )
            
            # Prepare data loader
            val_data_format = evaluator.cfg.data.format
            use_categories = False
            if evaluator.cfg.data.get('dataset_kwargs') is not None and 'use_categories' in evaluator.cfg.data.dataset_kwargs:
                use_categories = evaluator.cfg.data.dataset_kwargs.use_categories

            _, val_loader = prepare_data(
                evaluator.cfg.data.dataset,
                train_data_path=None,
                val_data_path=evaluator.cfg.data.val_path,
                data_format=val_data_format,
                batch_size=batch_size,
                num_workers=4,
                auto_augment=False,
                train_backgrounds=getattr(evaluator.cfg.data, 'train_backgrounds', None),
                val_backgrounds=getattr(evaluator.cfg.data, 'val_backgrounds', None),
                use_categories=use_categories
            )
            
            # Run evaluation
            evaluator.evaluate_dataset(val_loader, max_batches=None)
            
            # Analyze results
            results = evaluator.analyze_results()
            
            # Save raw data for batch analysis
            difficulties = np.array(evaluator.results['difficulties'])
            correct = np.array(evaluator.results['correct'])
            
            # Save raw data to epoch output directory
            raw_data_path = epoch_output_dir / "raw_data.npz"
            np.savez(raw_data_path, difficulties=difficulties, correct=correct)
            
            # Save results to epoch output directory  
            results_path = epoch_output_dir / "analysis_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Add raw data to results for threshold computation
            results['_raw_data'] = {
                'difficulties': difficulties,
                'correct': correct
            }
            
            return results
            
        except Exception as e:
            print(f"Error running direct analysis for epoch {epoch_num}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_single_difficulty_analysis_subprocess(self, 
                                                 backbone_checkpoint: str,
                                                 linear_checkpoint: str,
                                                 config_path: str,
                                                 epoch_num: int,
                                                 difficulty_method: str = 'entropy',
                                                 batch_size: int = 64) -> Optional[Dict]:
        """Run difficulty analysis for a single checkpoint using subprocess."""
        
        # Create output directory for this epoch
        epoch_output_dir = self.output_dir / f"epoch_{epoch_num:02d}" / difficulty_method
        epoch_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Run the difficulty analysis script (FIXED: correct script name)
        cmd = [
            sys.executable, "main_difficulty_eval.py",
            "--model_path", backbone_checkpoint,
            "--linear_path", linear_checkpoint,
            "--cfg_path", config_path,
            "--difficulty_method", difficulty_method,
            "--batch_size", str(batch_size),
            "--device", "cuda"
        ]
        
        print(f"Running {difficulty_method} analysis for epoch {epoch_num}...")
        
        # Change to the output directory for this run
        original_dir = os.getcwd()
        os.chdir(epoch_output_dir)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Load the results
            results_file = Path("difficulty_analysis/analysis_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Try to load raw numpy data if available
                raw_data_file = Path("difficulty_analysis/raw_data.npz")
                if raw_data_file.exists():
                    raw_data = np.load(raw_data_file)
                    results['_raw_data'] = {
                        'difficulties': raw_data['difficulties'],
                        'correct': raw_data['correct']
                    }
                else:
                    print(f"Warning: No raw data file found for epoch {epoch_num}")
                
                return results
            else:
                print(f"Warning: No results file found for epoch {epoch_num}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error running analysis for epoch {epoch_num}: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return None
        finally:
            os.chdir(original_dir)

    def run_single_difficulty_analysis(self, *args, **kwargs):
        """Main interface that chooses between direct and subprocess methods."""
        if DIRECT_IMPORT_AVAILABLE:
            return self.run_single_difficulty_analysis_direct(*args, **kwargs)
        else:
            return self.run_single_difficulty_analysis_subprocess(*args, **kwargs)
    
    def validate_checkpoint_pair(self, backbone_checkpoint: str, linear_checkpoint: str, 
                                config_path: str, epoch_num: int) -> bool:
        """Validate that checkpoint files exist and are accessible."""
        
        # Check backbone checkpoint
        if not Path(backbone_checkpoint).exists():
            print(f"Error: Backbone checkpoint not found for epoch {epoch_num}: {backbone_checkpoint}")
            return False
        
        # Check linear checkpoint
        if not Path(linear_checkpoint).exists():
            print(f"Error: Linear checkpoint not found for epoch {epoch_num}: {linear_checkpoint}")
            return False
        
        # Check config file
        if not Path(config_path).exists():
            print(f"Error: Config file not found: {config_path}")
            return False
        
        # Try to load checkpoint headers to verify they're valid
        try:
            backbone_ckpt = torch.load(backbone_checkpoint, map_location='cpu')
            if 'state_dict' not in backbone_ckpt:
                print(f"Warning: Backbone checkpoint for epoch {epoch_num} missing state_dict")
                return False
        except Exception as e:
            print(f"Error: Cannot load backbone checkpoint for epoch {epoch_num}: {e}")
            return False
        
        try:
            linear_ckpt = torch.load(linear_checkpoint, map_location='cpu')
            if 'state_dict' not in linear_ckpt:
                print(f"Warning: Linear checkpoint for epoch {epoch_num} missing state_dict")
                return False
        except Exception as e:
            print(f"Error: Cannot load linear checkpoint for epoch {epoch_num}: {e}")
            return False
        
        return True

    def compute_thresholds(self, difficulties: np.ndarray, correct: np.ndarray) -> Dict:
        """Compute various thresholds for a given difficulty distribution."""
        
        thresholds = {}
        
        # 1. Mode-based threshold
        hist, bins = np.histogram(difficulties, bins=50)
        mode_idx = np.argmax(hist)
        mode_value = (bins[mode_idx] + bins[mode_idx + 1]) / 2
        thresholds['mode'] = mode_value
        
        # Check if distribution is bimodal
        # Smooth histogram and find peaks
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
        peaks = []
        for i in range(1, len(hist_smooth) - 1):
            if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                peaks.append(i)
        
        if len(peaks) >= 2:
            # Find valley between two highest peaks
            peak_heights = [hist_smooth[p] for p in peaks]
            sorted_peaks = sorted(zip(peak_heights, peaks), reverse=True)
            if len(sorted_peaks) >= 2:
                peak1, peak2 = sorted_peaks[0][1], sorted_peaks[1][1]
                valley_idx = np.argmin(hist_smooth[min(peak1, peak2):max(peak1, peak2)]) + min(peak1, peak2)
                thresholds['valley'] = (bins[valley_idx] + bins[valley_idx + 1]) / 2
        
        # 2. Percentile-based thresholds
        thresholds['median'] = np.median(difficulties)
        thresholds['percentile_33'] = np.percentile(difficulties, 33.33)
        thresholds['percentile_67'] = np.percentile(difficulties, 66.67)
        
        # 3. Best accuracy difference threshold
        # Try different thresholds and find the one with maximum accuracy difference
        candidate_thresholds = np.linspace(difficulties.min(), difficulties.max(), 100)
        best_diff = -1
        best_threshold = thresholds['median']
        
        for t in candidate_thresholds:
            easy_mask = difficulties <= t
            hard_mask = difficulties > t
            
            if easy_mask.sum() > 10 and hard_mask.sum() > 10:  # Need enough samples
                easy_acc = correct[easy_mask].mean()
                hard_acc = correct[hard_mask].mean()
                acc_diff = easy_acc - hard_acc
                
                if acc_diff > best_diff:
                    best_diff = acc_diff
                    best_threshold = t
                    best_easy_acc = easy_acc
                    best_hard_acc = hard_acc
        
        thresholds['best_diff'] = best_threshold
        thresholds['best_diff_value'] = best_diff
        thresholds['best_diff_easy_acc'] = best_easy_acc
        thresholds['best_diff_hard_acc'] = best_hard_acc
        
        # Calculate accuracy for each threshold
        for name, threshold in thresholds.items():
            if not name.endswith('_acc') and not name.endswith('_value'):
                easy_mask = difficulties <= threshold
                hard_mask = difficulties > threshold
                
                if easy_mask.sum() > 0:
                    thresholds[f'{name}_easy_acc'] = correct[easy_mask].mean()
                    thresholds[f'{name}_easy_count'] = easy_mask.sum()
                else:
                    thresholds[f'{name}_easy_acc'] = 0
                    thresholds[f'{name}_easy_count'] = 0
                    
                if hard_mask.sum() > 0:
                    thresholds[f'{name}_hard_acc'] = correct[hard_mask].mean()
                    thresholds[f'{name}_hard_count'] = hard_mask.sum()
                else:
                    thresholds[f'{name}_hard_acc'] = 0
                    thresholds[f'{name}_hard_count'] = 0
                
                thresholds[f'{name}_acc_diff'] = thresholds[f'{name}_easy_acc'] - thresholds[f'{name}_hard_acc']
        
        return thresholds
    
    def run_batch_analysis(self, 
                          checkpoint_info: List[Tuple[int, str, str, str]],
                          difficulty_methods: List[str] = ['entropy', 'reconstruction']) -> None:
        """
        Run analysis for all checkpoints.
        
        Args:
            checkpoint_info: List of (epoch, backbone_ckpt, linear_ckpt, config_path)
            difficulty_methods: List of difficulty methods to analyze
        """
        
        for method in difficulty_methods:
            print(f"\n{'='*60}")
            print(f"Running {method} analysis for all epochs")
            print(f"{'='*60}")
            
            method_results = {}
            
            for epoch_num, backbone_ckpt, linear_ckpt, config_path in tqdm(checkpoint_info):
                results = self.run_single_difficulty_analysis(
                    backbone_ckpt, linear_ckpt, config_path, 
                    epoch_num, method
                )
                
                if results and '_raw_data' in results:
                    # Compute our custom thresholds
                    difficulties = results['_raw_data']['difficulties']
                    correct = results['_raw_data']['correct']
                    
                    threshold_results = self.compute_thresholds(difficulties, correct)
                    results['thresholds'] = threshold_results
                    
                    method_results[epoch_num] = results
            
            self.epoch_results[method] = method_results
        
        # Save all results
        self.save_results()
        
        # Create visualizations
        self.create_threshold_evolution_plots()
    
    def save_results(self):
        """Save aggregated results."""
        # Save without raw numpy arrays
        results_to_save = {}
        for method, epochs in self.epoch_results.items():
            results_to_save[method] = {}
            for epoch, data in epochs.items():
                # Remove raw data before saving
                data_copy = {k: v for k, v in data.items() if not k.startswith('_')}
                results_to_save[method][epoch] = data_copy
        
        with open(self.output_dir / 'aggregated_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)
    
    def create_threshold_evolution_plots(self):
        """Create comprehensive plots showing threshold evolution."""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # For each difficulty method
        for method in self.epoch_results:
            method_data = self.epoch_results[method]
            if not method_data:
                continue
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Threshold values over epochs
            ax1 = plt.subplot(3, 3, 1)
            self.plot_threshold_evolution(ax1, method_data, method)
            
            # 2. Accuracy differences over epochs
            ax2 = plt.subplot(3, 3, 2)
            self.plot_accuracy_differences(ax2, method_data, method)
            
            # 3. Easy vs Hard accuracy evolution
            ax3 = plt.subplot(3, 3, 3)
            self.plot_easy_hard_accuracy(ax3, method_data, method)
            
            # 4. Threshold comparison at specific epochs
            ax4 = plt.subplot(3, 3, 4)
            self.plot_threshold_comparison(ax4, method_data, method)
            
            # 5. Best threshold details
            ax5 = plt.subplot(3, 3, 5)
            self.plot_best_threshold_details(ax5, method_data, method)
            
            # 6. Overall accuracy evolution
            ax6 = plt.subplot(3, 3, 6)
            self.plot_overall_accuracy(ax6, method_data, method)
            
            # 7. Sample distribution
            ax7 = plt.subplot(3, 3, 7)
            self.plot_sample_distribution(ax7, method_data, method)
            
            # 8. Correlation evolution
            ax8 = plt.subplot(3, 3, 8)
            self.plot_correlation_evolution(ax8, method_data, method)
            
            # 9. Summary metrics
            ax9 = plt.subplot(3, 3, 9)
            self.plot_summary_metrics(ax9, method_data, method)
            
            plt.suptitle(f'Difficulty Analysis Evolution - {method.capitalize()} Method', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'threshold_evolution_{method}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_threshold_evolution(self, ax, method_data, method_name):
        """Plot how different thresholds evolve over epochs."""
        epochs = sorted(method_data.keys())
        
        threshold_types = ['mode', 'median', 'percentile_33', 'percentile_67', 'best_diff']
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for thresh_type, color in zip(threshold_types, colors):
            values = []
            for epoch in epochs:
                if 'thresholds' in method_data[epoch]:
                    values.append(method_data[epoch]['thresholds'].get(thresh_type, np.nan))
                else:
                    values.append(np.nan)
            
            ax.plot(epochs, values, marker='o', label=thresh_type, color=color)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Threshold Value')
        ax.set_title(f'Threshold Evolution - {method_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_accuracy_differences(self, ax, method_data, method_name):
        """Plot accuracy differences for each threshold type."""
        epochs = sorted(method_data.keys())
        
        threshold_types = ['mode', 'median', 'best_diff']
        colors = ['blue', 'green', 'purple']
        
        for thresh_type, color in zip(threshold_types, colors):
            acc_diffs = []
            for epoch in epochs:
                if 'thresholds' in method_data[epoch]:
                    acc_diffs.append(method_data[epoch]['thresholds'].get(f'{thresh_type}_acc_diff', 0))
                else:
                    acc_diffs.append(0)
            
            ax.plot(epochs, acc_diffs, marker='o', label=f'{thresh_type}', color=color, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy Difference (Easy - Hard)')
        ax.set_title('Accuracy Difference Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    def plot_easy_hard_accuracy(self, ax, method_data, method_name):
        """Plot easy vs hard accuracy for best threshold."""
        epochs = sorted(method_data.keys())
        
        easy_accs = []
        hard_accs = []
        
        for epoch in epochs:
            if 'thresholds' in method_data[epoch]:
                easy_accs.append(method_data[epoch]['thresholds'].get('best_diff_easy_acc', 0))
                hard_accs.append(method_data[epoch]['thresholds'].get('best_diff_hard_acc', 0))
            else:
                easy_accs.append(0)
                hard_accs.append(0)
        
        ax.plot(epochs, easy_accs, marker='o', label='Easy samples', color='green', linewidth=2)
        ax.plot(epochs, hard_accs, marker='s', label='Hard samples', color='red', linewidth=2)
        
        # Add shaded region to show the gap
        ax.fill_between(epochs, easy_accs, hard_accs, alpha=0.2, color='gray')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Easy vs Hard Sample Accuracy (Best Threshold)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def plot_threshold_comparison(self, ax, method_data, method_name):
        """Compare different thresholds at key epochs."""
        # Select key epochs (early, middle, late)
        all_epochs = sorted(method_data.keys())
        if len(all_epochs) >= 3:
            key_epochs = [all_epochs[0], all_epochs[len(all_epochs)//2], all_epochs[-1]]
        else:
            key_epochs = all_epochs
        
        threshold_types = ['mode', 'median', 'percentile_33', 'percentile_67', 'best_diff']
        x_pos = np.arange(len(threshold_types))
        width = 0.25
        
        for i, epoch in enumerate(key_epochs):
            if 'thresholds' in method_data[epoch]:
                values = []
                for thresh_type in threshold_types:
                    values.append(method_data[epoch]['thresholds'].get(thresh_type, 0))
                
                ax.bar(x_pos + i*width, values, width, label=f'Epoch {epoch}', alpha=0.8)
        
        ax.set_xlabel('Threshold Type')
        ax.set_ylabel('Threshold Value')
        ax.set_title('Threshold Comparison Across Epochs')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(threshold_types, rotation=45)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    def plot_best_threshold_details(self, ax, method_data, method_name):
        """Plot details about the best threshold selection."""
        epochs = sorted(method_data.keys())
        
        # Create text summary
        text_parts = []
        text_parts.append(f"Best Threshold Analysis ({method_name})\n")
        text_parts.append("="*40 + "\n\n")
        
        # Early epoch
        if epochs:
            early_epoch = epochs[0]
            if 'thresholds' in method_data[early_epoch]:
                thresh_data = method_data[early_epoch]['thresholds']
                text_parts.append(f"Epoch {early_epoch} (Early):\n")
                text_parts.append(f"  Best threshold: {thresh_data.get('best_diff', 0):.3f}\n")
                text_parts.append(f"  Accuracy diff: {thresh_data.get('best_diff_value', 0):.3f}\n")
                text_parts.append(f"  Easy acc: {thresh_data.get('best_diff_easy_acc', 0):.3f}\n")
                text_parts.append(f"  Hard acc: {thresh_data.get('best_diff_hard_acc', 0):.3f}\n\n")
        
        # Late epoch
        if len(epochs) > 1:
            late_epoch = epochs[-1]
            if 'thresholds' in method_data[late_epoch]:
                thresh_data = method_data[late_epoch]['thresholds']
                text_parts.append(f"Epoch {late_epoch} (Late):\n")
                text_parts.append(f"  Best threshold: {thresh_data.get('best_diff', 0):.3f}\n")
                text_parts.append(f"  Accuracy diff: {thresh_data.get('best_diff_value', 0):.3f}\n")
                text_parts.append(f"  Easy acc: {thresh_data.get('best_diff_easy_acc', 0):.3f}\n")
                text_parts.append(f"  Hard acc: {thresh_data.get('best_diff_hard_acc', 0):.3f}\n\n")
        
        # Trend analysis
        if len(epochs) > 2:
            best_thresholds = [method_data[e]['thresholds'].get('best_diff', 0) 
                             for e in epochs if 'thresholds' in method_data[e]]
            if best_thresholds:
                text_parts.append("Trend:\n")
                text_parts.append(f"  Threshold shift: {best_thresholds[-1] - best_thresholds[0]:.3f}\n")
                text_parts.append(f"  Direction: {'Increasing' if best_thresholds[-1] > best_thresholds[0] else 'Decreasing'}\n")
        
        ax.text(0.05, 0.95, ''.join(text_parts), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
    
    def plot_overall_accuracy(self, ax, method_data, method_name):
        """Plot overall accuracy evolution."""
        epochs = sorted(method_data.keys())
        
        overall_accs = []
        for epoch in epochs:
            overall_accs.append(method_data[epoch].get('overall_accuracy', 0))
        
        ax.plot(epochs, overall_accs, marker='o', linewidth=2, markersize=8, color='blue')
        ax.fill_between(epochs, overall_accs, alpha=0.3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Overall Accuracy')
        ax.set_title('Overall Model Accuracy Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def plot_sample_distribution(self, ax, method_data, method_name):
        """Plot distribution of easy/hard samples over epochs."""
        epochs = sorted(method_data.keys())
        
        easy_percents = []
        hard_percents = []
        
        for epoch in epochs:
            if 'thresholds' in method_data[epoch]:
                total = (method_data[epoch]['thresholds'].get('median_easy_count', 0) + 
                        method_data[epoch]['thresholds'].get('median_hard_count', 0))
                if total > 0:
                    easy_percent = method_data[epoch]['thresholds'].get('median_easy_count', 0) / total * 100
                    hard_percent = method_data[epoch]['thresholds'].get('median_hard_count', 0) / total * 100
                else:
                    easy_percent = hard_percent = 50
            else:
                easy_percent = hard_percent = 50
            
            easy_percents.append(easy_percent)
            hard_percents.append(hard_percent)
        
        ax.fill_between(epochs, 0, easy_percents, alpha=0.5, color='green', label='Easy')
        ax.fill_between(epochs, easy_percents, 100, alpha=0.5, color='red', label='Hard')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Percentage of Samples')
        ax.set_title('Sample Distribution (Median Threshold)')
        ax.legend()
        ax.set_ylim(0, 100)
    
    def plot_correlation_evolution(self, ax, method_data, method_name):
        """Plot correlation between difficulty and correctness."""
        epochs = sorted(method_data.keys())
        
        pearson_rs = []
        spearman_rhos = []
        
        for epoch in epochs:
            if 'correlations' in method_data[epoch]:
                pearson_rs.append(method_data[epoch]['correlations']['pearson']['r'])
                spearman_rhos.append(method_data[epoch]['correlations']['spearman']['rho'])
            else:
                pearson_rs.append(0)
                spearman_rhos.append(0)
        
        ax.plot(epochs, pearson_rs, marker='o', label='Pearson r', color='blue')
        ax.plot(epochs, spearman_rhos, marker='s', label='Spearman ρ', color='red')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Difficulty-Correctness Correlation Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)
    
    def plot_summary_metrics(self, ax, method_data, method_name):
        """Create a summary table of key metrics."""
        epochs = sorted(method_data.keys())
        
        if not epochs:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                   ha='center', va='center')
            ax.axis('off')
            return
        
        # Create summary data
        summary_data = []
        
        # Header
        summary_data.append(['Metric', f'Epoch {epochs[0]}', f'Epoch {epochs[-1]}', 'Change'])
        
        # Overall accuracy
        early_acc = method_data[epochs[0]].get('overall_accuracy', 0)
        late_acc = method_data[epochs[-1]].get('overall_accuracy', 0)
        summary_data.append(['Overall Acc', f'{early_acc:.3f}', f'{late_acc:.3f}', 
                           f'{late_acc - early_acc:+.3f}'])
        
        # Best threshold
        if 'thresholds' in method_data[epochs[0]] and 'thresholds' in method_data[epochs[-1]]:
            early_thresh = method_data[epochs[0]]['thresholds'].get('best_diff', 0)
            late_thresh = method_data[epochs[-1]]['thresholds'].get('best_diff', 0)
            summary_data.append(['Best Threshold', f'{early_thresh:.3f}', f'{late_thresh:.3f}',
                               f'{late_thresh - early_thresh:+.3f}'])
            
            # Accuracy difference
            early_diff = method_data[epochs[0]]['thresholds'].get('best_diff_value', 0)
            late_diff = method_data[epochs[-1]]['thresholds'].get('best_diff_value', 0)
            summary_data.append(['Acc Diff (E-H)', f'{early_diff:.3f}', f'{late_diff:.3f}',
                               f'{late_diff - early_diff:+.3f}'])
        
        # Create table
        table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.axis('off')
        ax.set_title('Summary Metrics', pad=20)


def get_checkpoint_info(base_dir: str, n_epochs: int = 20) -> List[Tuple[int, str, str, str]]:
    """
    Get checkpoint information for the first n epochs.
    Returns list of (epoch, backbone_checkpoint, linear_checkpoint, config_path)
    """
    checkpoint_info = []
    
    # Define paths based on your directory structure
    backbone_dir = Path(base_dir) / "selective_curriculum_mocov3" / "t60"
    linear_base_dir = Path(base_dir) / "linear" / "selective_curriculum_mocov3_t60"
    config_base_path = "scripts/linear_probe/core50/mocov3_linear.yaml"  # Your base config
    
    for epoch in range(n_epochs):
        # Find backbone checkpoint
        backbone_pattern = f"*-ep={epoch}-stp=0.ckpt"
        backbone_files = list(backbone_dir.glob(backbone_pattern))
        
        if not backbone_files:
            print(f"Warning: No backbone checkpoint found for epoch {epoch}")
            continue
        
        backbone_checkpoint = str(backbone_files[0])
        
        # Find linear checkpoint
        linear_dir = linear_base_dir / f"selective_curriculum_mocov3_t60_ep{epoch:02d}"
        linear_files = list(linear_dir.glob("*.ckpt"))
        
        if not linear_files:
            print(f"Warning: No linear checkpoint found for epoch {epoch}")
            continue
        
        linear_checkpoint = str(linear_files[0])
        
        checkpoint_info.append((epoch, backbone_checkpoint, linear_checkpoint, config_base_path))
    
    return checkpoint_info


def main():
    # Configuration
    base_dir = "/home/brothen/solo-learn/trained_models"
    output_dir = "batch_difficulty_analysis_t60"
    n_epochs = 20
    difficulty_methods = ['entropy', 'reconstruction']  # Add 'margin' if you want
    
    # Get checkpoint information
    checkpoint_info = get_checkpoint_info(base_dir, n_epochs)
    
    if not checkpoint_info:
        print("Error: No checkpoints found!")
        return 1
    
    print(f"Found {len(checkpoint_info)} checkpoint pairs to analyze")
    print(f"Difficulty methods: {difficulty_methods}")
    print(f"Output directory: {output_dir}")
    
    # Create analyzer and run batch analysis
    analyzer = BatchDifficultyAnalyzer(base_dir, output_dir)
    analyzer.run_batch_analysis(checkpoint_info, difficulty_methods)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Main plots: {output_dir}/threshold_evolution_*.png")
    print(f"Individual epoch results: {output_dir}/epoch_*/")
    
    return 0


if __name__ == "__main__":
    exit(main())