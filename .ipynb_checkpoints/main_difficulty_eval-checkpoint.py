"""
Standalone script to evaluate sample difficulty and classification accuracy.
This can be run after training to analyze the relationship between difficulty and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression

from solo.methods.base import BaseMethod
from solo.data.classification_dataloader import prepare_data
from omegaconf import OmegaConf


class DifficultyEvaluator:
    """Evaluates sample difficulty and classification performance."""
    
    def __init__(self, model_path: str, linear_path: str, cfg_path: str, 
                 device: str = 'cuda', difficulty_method: str = 'reconstruction'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.difficulty_method = difficulty_method
        
        # Load config
        self.cfg = OmegaConf.load(cfg_path)
        
        # Load pretrained model (CurriculumMoCoV3)
        self.model = self.load_pretrained_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load linear classifier
        self.classifier = self.load_linear_classifier(linear_path)
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # Storage for results
        self.results = {
            'difficulties': [],
            'predictions': [],
            'targets': [],
            'correct': [],
            'indices': [],
            'features': []  # Optional: store features for t-SNE
        }
    
    def load_pretrained_model(self, model_path):
        """Load the pretrained CurriculumMoCoV3 model."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Initialize model using proper import
        from solo.methods.curriculum_mocov3 import CurriculumMoCoV3
        model = CurriculumMoCoV3(self.cfg)
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def load_linear_classifier(self, linear_path):
        """Load the trained linear classifier."""
        checkpoint = torch.load(linear_path, map_location='cpu')
        
        # Extract classifier architecture from checkpoint
        # The LinearModel saves the classifier weights with 'classifier.' prefix
        classifier_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith('classifier.')]
        
        if not classifier_keys:
            raise ValueError("No classifier weights found in checkpoint. Make sure this is a linear model checkpoint.")
        
        # Get dimensions from weights
        weight_key = [k for k in classifier_keys if k.endswith('.weight')][0]
        bias_key = [k for k in classifier_keys if k.endswith('.bias')][0]
        
        num_features = checkpoint['state_dict'][weight_key].shape[1]
        num_classes = checkpoint['state_dict'][weight_key].shape[0]
        
        classifier = nn.Linear(num_features, num_classes)
        
        # Load classifier weights without prefix
        classifier_state = {
            'weight': checkpoint['state_dict'][weight_key],
            'bias': checkpoint['state_dict'][bias_key]
        }
        classifier.load_state_dict(classifier_state)
        
        return classifier
    
    def compute_reconstruction_difficulty(self, images, indices=None):
        """Compute difficulty based on reconstruction error."""
        with torch.no_grad():
            # Get features from first view
            out = self.model(images)
            
            if hasattr(self.model, 'is_vit') and self.model.is_vit:
                feats = out.get('feats_vit', out.get('feats'))
                
                # MAE-style reconstruction
                if self.model.curriculum_type == "mae":
                    # Create mask
                    B = feats.shape[0]
                    num_tokens = feats.shape[1]
                    mask_ratio = self.model.masking_ratio
                    
                    # Random mask for each sample
                    mask = torch.zeros(B, num_tokens, dtype=torch.bool, device=feats.device)
                    mask_length = int(mask_ratio * num_tokens)
                    
                    for i in range(B):
                        mask_indices = torch.randperm(num_tokens, device=feats.device)[:mask_length]
                        mask[i, mask_indices] = True
                    
                    # Compute reconstruction error
                    _, per_sample_error = self.model._compute_mae_reconstruction(
                        feats, mask, ([indices, [images]],)
                    )
                else:
                    # JEPA-style
                    # Create context and target masks
                    num_tokens = feats.shape[1]
                    target_ratio = self.model.masking_ratio
                    target_length = int(target_ratio * num_tokens)
                    
                    context_mask = torch.ones(B, num_tokens, dtype=torch.bool, device=feats.device)
                    target_mask = torch.zeros(B, num_tokens, dtype=torch.bool, device=feats.device)
                    
                    for i in range(B):
                        indices = torch.randperm(num_tokens, device=feats.device)
                        target_indices = indices[:target_length]
                        context_mask[i, target_indices] = False
                        target_mask[i, target_indices] = True
                    
                    _, per_sample_error = self.model._compute_jepa_reconstruction(
                        feats, context_mask, target_mask
                    )
            else:
                # ResNet-based
                feats = out.get('feats')
                if self.model.curriculum_type == "mae":
                    # Image reconstruction
                    pred_img = self.model.decoder(feats.detach())
                    per_sample_error = ((pred_img - images) ** 2).mean(dim=(1, 2, 3))
                else:
                    # Feature prediction - need second view
                    # For simplicity, use same image (you might want to augment)
                    out2 = self.model(images)
                    pred_feats = self.model.predictor_jepa(feats.detach())
                    per_sample_error = F.mse_loss(
                        pred_feats, out2['feats'].detach(), reduction='none'
                    ).mean(dim=1)
            
            # Normalize difficulties to [0, 1]
            per_sample_error = per_sample_error.cpu()
            min_err = per_sample_error.min()
            max_err = per_sample_error.max()
            
            if max_err > min_err:
                difficulties = (per_sample_error - min_err) / (max_err - min_err)
            else:
                difficulties = torch.zeros_like(per_sample_error)
            
            return difficulties, feats
    
    def compute_entropy_difficulty(self, logits):
        """Compute difficulty based on prediction entropy."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32))
            difficulty = entropy / max_entropy
        return difficulty.cpu()
    
    def evaluate_dataset(self, dataloader, max_batches=None):
        """Evaluate the entire dataset."""
        print(f"Evaluating with {self.difficulty_method} difficulty method...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Parse batch
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                if isinstance(batch[0], torch.Tensor) and batch[0].dim() == 1:
                    indices = batch[0]
                    if isinstance(batch[1], (list, tuple)):
                        images, targets = batch[1]
                    else:
                        images = batch[1]
                        targets = batch[2] if len(batch) > 2 else None
                else:
                    images, targets = batch
                    indices = torch.arange(len(images))
            else:
                continue
            
            images = images.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # Compute difficulties based on method
            if self.difficulty_method == 'reconstruction':
                difficulties, features = self.compute_reconstruction_difficulty(images, indices)
            else:
                # Get features for classification
                with torch.no_grad():
                    out = self.model(images)
                    features = out.get('feats')
            
            # Get predictions from linear classifier
            with torch.no_grad():
                if hasattr(features, 'shape') and len(features.shape) > 2:
                    # Pool features if needed (e.g., for ViT)
                    features = features.mean(dim=1)
                
                logits = self.classifier(features)
                predictions = torch.argmax(logits, dim=-1)
                
                if self.difficulty_method == 'entropy':
                    difficulties = self.compute_entropy_difficulty(logits)
            
            # Compute correctness
            if targets is not None:
                correct = (predictions == targets).cpu().numpy()
            else:
                correct = np.zeros(len(predictions))
            
            # Store results
            self.results['difficulties'].extend(difficulties.numpy())
            self.results['predictions'].extend(predictions.cpu().numpy())
            if targets is not None:
                self.results['targets'].extend(targets.cpu().numpy())
            self.results['correct'].extend(correct)
            self.results['indices'].extend(indices.cpu().numpy())
            
            # Optionally store features for visualization
            if batch_idx < 10:  # Only store first few batches
                self.results['features'].extend(features.cpu().numpy())
    
    def analyze_results(self):
        """Analyze the relationship between difficulty and accuracy."""
        # Convert to numpy arrays
        difficulties = np.array(self.results['difficulties'])
        correct = np.array(self.results['correct'])
        predictions = np.array(self.results['predictions'])
        targets = np.array(self.results['targets']) if self.results['targets'] else None
        
        # Overall statistics
        overall_acc = correct.mean()
        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        print(f"Difficulty Range: [{difficulties.min():.4f}, {difficulties.max():.4f}]")
        print(f"Difficulty Mean: {difficulties.mean():.4f} ± {difficulties.std():.4f}")
        
        # Compute accuracy by difficulty bins
        n_bins = 10
        bin_edges = np.linspace(difficulties.min(), difficulties.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_accuracies = []
        bin_counts = []
        bin_stds = []
        
        for i in range(n_bins):
            mask = (difficulties >= bin_edges[i]) & (difficulties < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (difficulties >= bin_edges[i]) & (difficulties <= bin_edges[i + 1])
            
            bin_samples = correct[mask]
            if len(bin_samples) > 0:
                bin_accuracies.append(bin_samples.mean())
                bin_counts.append(len(bin_samples))
                bin_stds.append(bin_samples.std() / np.sqrt(len(bin_samples)))  # SEM
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
                bin_stds.append(0)
        
        # Print bin statistics
        print("\nAccuracy by Difficulty Bins:")
        print("Bin\tDifficulty Range\tSamples\tAccuracy")
        for i in range(n_bins):
            print(f"{i+1}\t[{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}]\t"
                  f"{bin_counts[i]}\t{bin_accuracies[i]:.3f}")
        
        # Compute correlation
        corr_pearson, p_pearson = pearsonr(difficulties, correct)
        corr_spearman, p_spearman = spearmanr(difficulties, correct)
        
        print(f"\nCorrelation Analysis:")
        print(f"Pearson r = {corr_pearson:.4f} (p = {p_pearson:.4e})")
        print(f"Spearman ρ = {corr_spearman:.4f} (p = {p_spearman:.4e})")
        
        # Save detailed results
        results_dict = {
            'overall_accuracy': float(overall_acc),
            'difficulty_stats': {
                'min': float(difficulties.min()),
                'max': float(difficulties.max()),
                'mean': float(difficulties.mean()),
                'std': float(difficulties.std())
            },
            'correlations': {
                'pearson': {'r': float(corr_pearson), 'p': float(p_pearson)},
                'spearman': {'rho': float(corr_spearman), 'p': float(p_spearman)}
            },
            'bins': {
                'edges': bin_edges.tolist(),
                'centers': bin_centers.tolist(),
                'accuracies': [float(x) for x in bin_accuracies],
                'counts': [int(x) for x in bin_counts],
                'std_errors': [float(x) for x in bin_stds]
            }
        }
        
        save_dir = Path('difficulty_analysis')
        save_dir.mkdir(exist_ok=True)
        
        with open(save_dir / 'analysis_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save raw data for batch analysis
        np.savez(save_dir / 'raw_data.npz', 
                 difficulties=difficulties, 
                 correct=correct,
                 predictions=np.array(self.results['predictions']),
                 targets=np.array(self.results['targets']) if self.results['targets'] else None)
        
        # Create visualizations
        self.create_visualizations(difficulties, correct, bin_centers, 
                                   bin_accuracies, bin_counts, bin_stds)
        
        return results_dict
    
    def create_visualizations(self, difficulties, correct, bin_centers, 
                              bin_accuracies, bin_counts, bin_stds):
        """Create comprehensive visualization plots."""
        save_dir = Path('difficulty_analysis/plots')
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style - use more compatible approach
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                plt.style.use('default')
                # Set some basic styling manually
                plt.rcParams['axes.grid'] = True
                plt.rcParams['axes.axisbelow'] = True
        
        # 1. Main analysis figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Difficulty distribution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(difficulties[correct], bins=50, alpha=0.6, label='Correct', 
                 density=True, color='green')
        ax1.hist(difficulties[~correct], bins=50, alpha=0.6, label='Incorrect', 
                 density=True, color='red')
        ax1.set_xlabel('Difficulty Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Difficulty Scores by Correctness')
        ax1.legend()
        
        # KDE plot
        ax2 = fig.add_subplot(gs[0, 2])
        for label, mask, color in [('Correct', correct, 'green'), 
                                    ('Incorrect', ~correct, 'red')]:
            if mask.sum() > 0:
                sns.kdeplot(difficulties[mask], ax=ax2, label=label, color=color)
        ax2.set_xlabel('Difficulty Score')
        ax2.set_ylabel('Density')
        ax2.set_title('KDE of Difficulty Scores')
        ax2.legend()
        
        # Accuracy vs difficulty bins with error bars
        ax3 = fig.add_subplot(gs[1, :])
        bars = ax3.bar(bin_centers, bin_accuracies, width=0.8*(bin_centers[1]-bin_centers[0]),
                        yerr=bin_stds, capsize=5, alpha=0.7, edgecolor='black')
        
        # Color bars by accuracy
        norm = plt.Normalize(vmin=0, vmax=1)
        colors = plt.cm.RdYlGn(norm(bin_accuracies))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
        
        ax3.set_xlabel('Difficulty Score (Bin Center)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Classification Accuracy vs Difficulty Bins')
        ax3.set_ylim(0, 1.05)
        
        # Add sample count annotations
        for i, (x, y, n, err) in enumerate(zip(bin_centers, bin_accuracies, 
                                                bin_counts, bin_stds)):
            if n > 0:
                ax3.text(x, y + err + 0.02, f'n={n}', ha='center', fontsize=8)
        
        # Add overall accuracy line
        overall_acc = correct.mean()
        ax3.axhline(y=overall_acc, color='blue', linestyle='--', alpha=0.5,
                    label=f'Overall Accuracy: {overall_acc:.3f}')
        ax3.legend()
        
        # Scatter plot with regression
        ax4 = fig.add_subplot(gs[2, 0])
        # Subsample for clarity
        n_sample = min(5000, len(difficulties))
        idx = np.random.choice(len(difficulties), n_sample, replace=False)
        
        ax4.scatter(difficulties[idx], correct[idx], alpha=0.3, s=10)
        
        # Add logistic regression fit
        lr = LogisticRegression()
        lr.fit(difficulties.reshape(-1, 1), correct)
        x_range = np.linspace(difficulties.min(), difficulties.max(), 100)
        y_pred = lr.predict_proba(x_range.reshape(-1, 1))[:, 1]
        ax4.plot(x_range, y_pred, 'r-', linewidth=2, label='Logistic Fit')
        
        ax4.set_xlabel('Difficulty Score')
        ax4.set_ylabel('P(Correct)')
        ax4.set_title('Difficulty vs Correctness')
        ax4.legend()
        
        # Cumulative accuracy
        ax5 = fig.add_subplot(gs[2, 1])
        sorted_idx = np.argsort(difficulties)
        sorted_correct = correct[sorted_idx]
        cumulative_acc = np.cumsum(sorted_correct) / np.arange(1, len(sorted_correct) + 1)
        percentiles = np.linspace(0, 100, len(cumulative_acc))
        
        ax5.plot(percentiles, cumulative_acc, linewidth=2)
        ax5.fill_between(percentiles, cumulative_acc, alpha=0.3)
        ax5.set_xlabel('Percentile (Easy → Hard)')
        ax5.set_ylabel('Cumulative Accuracy')
        ax5.set_title('Cumulative Accuracy by Difficulty Percentile')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 100)
        ax5.set_ylim(0, 1.05)
        
        # Add quartile lines
        for q in [25, 50, 75]:
            idx = int(q * len(cumulative_acc) / 100)
            ax5.axvline(x=q, color='gray', linestyle=':', alpha=0.5)
            ax5.text(q, 0.02, f'{q}%', ha='center', fontsize=8)
            ax5.text(q, cumulative_acc[idx] + 0.02, 
                     f'{cumulative_acc[idx]:.3f}', ha='center', fontsize=8)
        
        # Percentile analysis
        ax6 = fig.add_subplot(gs[2, 2])
        percentile_ranges = ['0-10%', '10-25%', '25-50%', '50-75%', '75-90%', '90-100%']
        percentile_bounds = [0, 10, 25, 50, 75, 90, 100]
        percentile_accs = []
        
        for i in range(len(percentile_bounds) - 1):
            start_idx = int(percentile_bounds[i] * len(sorted_correct) / 100)
            end_idx = int(percentile_bounds[i+1] * len(sorted_correct) / 100)
            acc = sorted_correct[start_idx:end_idx].mean()
            percentile_accs.append(acc)
        
        bars = ax6.bar(percentile_ranges, percentile_accs, alpha=0.7)
        for bar, acc in zip(bars, percentile_accs):
            bar.set_color(plt.cm.RdYlGn(acc))
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax6.set_xlabel('Difficulty Percentile')
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Accuracy by Difficulty Percentiles')
        ax6.set_ylim(0, 1.1)
        ax6.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Difficulty Analysis - {self.difficulty_method.capitalize()} Method', 
                     fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'difficulty_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create a summary infographic
        self.create_summary_infographic(difficulties, correct, overall_acc)
    
    def create_summary_infographic(self, difficulties, correct, overall_acc):
        """Create a clean summary infographic."""
        save_dir = Path('difficulty_analysis/plots')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        # Key metrics
        ax = axes[0, 0]
        ax.axis('off')
        
        # Calculate key statistics
        easy_mask = difficulties <= np.percentile(difficulties, 33)
        medium_mask = (difficulties > np.percentile(difficulties, 33)) & \
                      (difficulties <= np.percentile(difficulties, 67))
        hard_mask = difficulties > np.percentile(difficulties, 67)
        
        easy_acc = correct[easy_mask].mean()
        medium_acc = correct[medium_mask].mean()
        hard_acc = correct[hard_mask].mean()
        
        metrics_text = f"""
Key Metrics

Overall Accuracy: {overall_acc:.1%}

Easy Samples (Bottom 33%): {easy_acc:.1%}
Medium Samples (Middle 33%): {medium_acc:.1%}
Hard Samples (Top 33%): {hard_acc:.1%}

Accuracy Drop (Easy→Hard): {(easy_acc - hard_acc):.1%}
Total Samples: {len(difficulties):,}
        """
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, 
                fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        
        # Pie chart of difficulty distribution
        ax = axes[0, 1]
        sizes = [easy_mask.sum(), medium_mask.sum(), hard_mask.sum()]
        labels = ['Easy', 'Medium', 'Hard']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        explode = (0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                           autopct='%1.1f%%', explode=explode,
                                           shadow=True, startangle=90)
        ax.set_title('Distribution of Sample Difficulties')
        
        # Accuracy comparison bar chart
        ax = axes[1, 0]
        categories = ['Easy\n(0-33%)', 'Medium\n(33-67%)', 'Hard\n(67-100%)', 'Overall']
        accuracies = [easy_acc, medium_acc, hard_acc, overall_acc]
        colors_bar = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
        
        bars = ax.bar(categories, accuracies, color=colors_bar, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy by Difficulty Category', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Difficulty score distribution (violin plot)
        ax = axes[1, 1]
        data_for_violin = []
        labels_violin = []
        
        if correct.sum() > 0:
            data_for_violin.append(difficulties[correct])
            labels_violin.append('Correct')
        
        if (~correct).sum() > 0:
            data_for_violin.append(difficulties[~correct])
            labels_violin.append('Incorrect')
        
        if data_for_violin:
            parts = ax.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                                  showmeans=True, showmedians=True)
            
            # Customize violin plot
            for pc, color in zip(parts['bodies'], ['green', 'red'][:len(data_for_violin)]):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            
            ax.set_xticks(range(len(labels_violin)))
            ax.set_xticklabels(labels_violin)
            ax.set_ylabel('Difficulty Score')
            ax.set_title('Difficulty Distribution by Correctness')
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('Sample Difficulty Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / 'difficulty_summary_infographic.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate sample difficulty and accuracy')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained CurriculumMoCoV3 checkpoint')
    parser.add_argument('--linear_path', type=str, required=True,
                        help='Path to trained linear classifier checkpoint')
    parser.add_argument('--cfg_path', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--difficulty_method', type=str, default='reconstruction',
                        choices=['reconstruction', 'entropy'],
                        help='Method to compute difficulty')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to evaluate (for debugging)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = DifficultyEvaluator(
        model_path=args.model_path,
        linear_path=args.linear_path,
        cfg_path=args.cfg_path,
        device=args.device,
        difficulty_method=args.difficulty_method
    )
    
    # Prepare data loader using the same logic as main_linear.py
    val_data_format = evaluator.cfg.data.format
    
    # Extract the use_categories parameter if present in dataset_kwargs
    use_categories = False
    if evaluator.cfg.data.get('dataset_kwargs') is not None and 'use_categories' in evaluator.cfg.data.dataset_kwargs:
        use_categories = evaluator.cfg.data.dataset_kwargs.use_categories

    # Only prepare validation data
    _, val_loader = prepare_data(
        evaluator.cfg.data.dataset,
        train_data_path=None,  # We don't need training data
        val_data_path=evaluator.cfg.data.val_path,
        data_format=val_data_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        auto_augment=False,  # No augmentation for evaluation
        train_backgrounds=getattr(evaluator.cfg.data, 'train_backgrounds', None),
        val_backgrounds=getattr(evaluator.cfg.data, 'val_backgrounds', None),
        use_categories=use_categories
    )
    
    # Run evaluation
    evaluator.evaluate_dataset(val_loader, max_batches=args.max_batches)
    
    # Analyze results
    results = evaluator.analyze_results()
    
    print("\nEvaluation complete! Results saved to 'difficulty_analysis/'")


if __name__ == '__main__':
    main()