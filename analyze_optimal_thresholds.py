#!/usr/bin/env python3
"""
Analyze Optimal Thresholds Across Training Epochs

This script analyzes the aggregated results to find optimal difficulty thresholds
for each method across training epochs using multiple approaches:

1. Maximum accuracy difference (easy vs hard)
2. Bimodal distribution analysis (peaks for correct vs incorrect)
3. Percentile-based thresholds
4. ROC-like threshold optimization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

class ThresholdAnalyzer:
    """Analyzes optimal difficulty thresholds across training epochs."""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = self.load_results()
        self.threshold_evolution = {}
        
    def load_results(self) -> Dict:
        """Load aggregated results from JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def find_optimal_threshold_max_diff(self, bins: Dict) -> Tuple[float, float, Dict]:
        """
        Method 1: Find threshold that maximizes accuracy difference between easy and hard samples.
        
        Returns:
            optimal_threshold, max_difference, analysis_details
        """
        centers = np.array(bins['centers'])
        accuracies = np.array(bins['accuracies'])
        counts = np.array(bins['counts'])
        
        # Remove bins with no samples
        valid_mask = counts > 0
        if not valid_mask.any():
            return np.nan, 0, {}
        
        centers = centers[valid_mask]
        accuracies = accuracies[valid_mask]
        counts = counts[valid_mask]
        
        if len(centers) < 3:  # Need at least 3 bins to find a meaningful threshold
            return np.nan, 0, {}
        
        best_threshold = np.nan
        max_diff = 0
        best_analysis = {}
        
        # Try each bin center as a potential threshold
        for i in range(1, len(centers) - 1):  # Skip first and last to ensure both sides have data
            threshold = centers[i]
            
            # Calculate weighted accuracy for easy (below threshold) and hard (above threshold)
            easy_mask = centers <= threshold
            hard_mask = centers > threshold
            
            if not easy_mask.any() or not hard_mask.any():
                continue
            
            # Weighted average accuracy
            easy_weights = counts[easy_mask]
            hard_weights = counts[hard_mask]
            
            easy_acc = np.average(accuracies[easy_mask], weights=easy_weights)
            hard_acc = np.average(accuracies[hard_mask], weights=hard_weights)
            
            diff = easy_acc - hard_acc
            
            if diff > max_diff:
                max_diff = diff
                best_threshold = threshold
                best_analysis = {
                    'easy_accuracy': easy_acc,
                    'hard_accuracy': hard_acc,
                    'easy_samples': easy_weights.sum(),
                    'hard_samples': hard_weights.sum(),
                    'threshold_bin_index': i
                }
        
        return best_threshold, max_diff, best_analysis
    
    def find_percentile_thresholds(self, bins: Dict, percentiles: List[float] = [25, 50, 75]) -> Dict:
        """
        Method 2: Find thresholds based on percentiles of difficulty distribution.
        
        Args:
            percentiles: List of percentiles to compute (e.g., [25, 50, 75] for quartiles)
        """
        centers = np.array(bins['centers'])
        counts = np.array(bins['counts'])
        
        # Create expanded array of difficulty values
        difficulties = []
        for center, count in zip(centers, counts):
            difficulties.extend([center] * int(count))
        
        if not difficulties:
            return {f'p{p}': np.nan for p in percentiles}
        
        difficulties = np.array(difficulties)
        
        threshold_dict = {}
        for p in percentiles:
            threshold_dict[f'p{p}'] = np.percentile(difficulties, p)
        
        return threshold_dict
    
    def find_bimodal_threshold(self, bins: Dict, correct_samples: np.ndarray, 
                              incorrect_samples: np.ndarray) -> Tuple[float, Dict]:
        """
        Method 3: Find threshold based on bimodal distribution analysis.
        
        This would require individual sample data, which we don't have in the aggregated results.
        Instead, we'll approximate using the accuracy curve shape.
        """
        centers = np.array(bins['centers'])
        accuracies = np.array(bins['accuracies'])
        counts = np.array(bins['counts'])
        
        valid_mask = counts > 0
        if not valid_mask.any() or len(centers[valid_mask]) < 3:
            return np.nan, {}
        
        centers = centers[valid_mask]
        accuracies = accuracies[valid_mask]
        counts = counts[valid_mask]
        
        # Find the steepest drop in accuracy (proxy for bimodal separation)
        acc_gradient = np.gradient(accuracies)
        steepest_drop_idx = np.argmin(acc_gradient)
        
        if steepest_drop_idx == 0 or steepest_drop_idx == len(centers) - 1:
            # Fallback to midpoint
            threshold = centers[len(centers) // 2]
        else:
            threshold = centers[steepest_drop_idx]
        
        analysis = {
            'gradient_min_idx': steepest_drop_idx,
            'gradient_min_value': acc_gradient[steepest_drop_idx],
            'accuracy_at_threshold': accuracies[steepest_drop_idx]
        }
        
        return threshold, analysis
    
    def analyze_epoch(self, method: str, epoch: int, epoch_data: Dict) -> Dict:
        """Analyze optimal thresholds for a single epoch and method."""
        
        bins = epoch_data['bins']
        overall_acc = epoch_data['overall_accuracy']
        
        # Method 1: Maximum accuracy difference
        opt_thresh, max_diff, diff_analysis = self.find_optimal_threshold_max_diff(bins)
        
        # Method 2: Percentile thresholds
        percentile_thresholds = self.find_percentile_thresholds(bins)
        
        # Method 3: Bimodal approximation
        bimodal_thresh, bimodal_analysis = self.find_bimodal_threshold(bins, None, None)
        
        # Additional metrics
        difficulty_range = epoch_data['difficulty_stats']['max'] - epoch_data['difficulty_stats']['min']
        difficulty_std = epoch_data['difficulty_stats']['std']
        correlation = epoch_data['correlations']['pearson']['r']
        
        return {
            'epoch': epoch,
            'method': method,
            'overall_accuracy': overall_acc,
            'optimal_threshold_max_diff': opt_thresh,
            'max_accuracy_difference': max_diff,
            'easy_accuracy': diff_analysis.get('easy_accuracy', np.nan),
            'hard_accuracy': diff_analysis.get('hard_accuracy', np.nan),
            'easy_samples': diff_analysis.get('easy_samples', 0),
            'hard_samples': diff_analysis.get('hard_samples', 0),
            'threshold_p25': percentile_thresholds.get('p25', np.nan),
            'threshold_p50': percentile_thresholds.get('p50', np.nan),
            'threshold_p75': percentile_thresholds.get('p75', np.nan),
            'bimodal_threshold': bimodal_thresh,
            'difficulty_range': difficulty_range,
            'difficulty_std': difficulty_std,
            'difficulty_correlation': correlation
        }
    
    def analyze_all_methods(self):
        """Analyze optimal thresholds for all methods and epochs."""
        
        print("🔍 Analyzing optimal thresholds across training epochs...")
        
        for method in ['entropy', 'reconstruction', 'margin', 'pixel_entropy']:
            if method not in self.results:
                print(f"⚠️  Method '{method}' not found in results")
                continue
            
            print(f"\n📊 Analyzing {method} method...")
            
            method_results = []
            epochs = sorted([int(k) for k in self.results[method].keys()])
            
            for epoch in epochs:
                epoch_data = self.results[method][str(epoch)]
                analysis = self.analyze_epoch(method, epoch, epoch_data)
                method_results.append(analysis)
                
                print(f"  Epoch {epoch:2d}: Optimal threshold = {analysis['optimal_threshold_max_diff']:.4f}, "
                      f"Max diff = {analysis['max_accuracy_difference']:.3f}, "
                      f"Overall acc = {analysis['overall_accuracy']:.3f}")
            
            self.threshold_evolution[method] = method_results
        
        print(f"\n✅ Analysis complete for {len(self.threshold_evolution)} methods")
    
    def create_threshold_evolution_plots(self, output_dir: Path):
        """Create comprehensive plots showing threshold evolution."""
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Main figure: Threshold evolution across epochs
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Optimal Difficulty Thresholds Evolution During Training', fontsize=16, fontweight='bold')
        
        methods = list(self.threshold_evolution.keys())
        colors = ['blue', 'red', 'green', 'orange'][:len(methods)]
        
        # Plot 1: Optimal threshold (max difference method)
        ax = axes[0, 0]
        for i, method in enumerate(methods):
            data = self.threshold_evolution[method]
            df = pd.DataFrame(data)
            
            # Filter out NaN values
            valid_mask = ~df['optimal_threshold_max_diff'].isna()
            if valid_mask.any():
                epochs = df[valid_mask]['epoch']
                thresholds = df[valid_mask]['optimal_threshold_max_diff']
                ax.plot(epochs, thresholds, marker='o', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Optimal Threshold')
        ax.set_title('Optimal Threshold (Max Accuracy Difference)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Maximum accuracy difference evolution
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            data = self.threshold_evolution[method]
            df = pd.DataFrame(data)
            
            epochs = df['epoch']
            max_diffs = df['max_accuracy_difference']
            ax.plot(epochs, max_diffs, marker='s', label=f'{method.title()}', 
                   color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Max Accuracy Difference (Easy - Hard)')
        ax.set_title('Difficulty Discrimination Power')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Percentile thresholds (median)
        ax = axes[0, 2]
        for i, method in enumerate(methods):
            data = self.threshold_evolution[method]
            df = pd.DataFrame(data)
            
            valid_mask = ~df['threshold_p50'].isna()
            if valid_mask.any():
                epochs = df[valid_mask]['epoch']
                p50_thresholds = df[valid_mask]['threshold_p50']
                ax.plot(epochs, p50_thresholds, marker='^', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Median Difficulty Threshold')
        ax.set_title('50th Percentile Threshold Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Easy vs Hard accuracy evolution
        ax = axes[1, 0]
        for i, method in enumerate(methods):
            data = self.threshold_evolution[method]
            df = pd.DataFrame(data)
            
            valid_mask = ~df['easy_accuracy'].isna() & ~df['hard_accuracy'].isna()
            if valid_mask.any():
                epochs = df[valid_mask]['epoch']
                easy_acc = df[valid_mask]['easy_accuracy']
                hard_acc = df[valid_mask]['hard_accuracy']
                
                ax.plot(epochs, easy_acc, marker='o', linestyle='-', 
                       label=f'{method.title()} Easy', color=colors[i], alpha=0.7)
                ax.plot(epochs, hard_acc, marker='s', linestyle='--', 
                       label=f'{method.title()} Hard', color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Easy vs Hard Sample Accuracy')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Difficulty range evolution
        ax = axes[1, 1]
        for i, method in enumerate(methods):
            data = self.threshold_evolution[method]
            df = pd.DataFrame(data)
            
            epochs = df['epoch']
            ranges = df['difficulty_range']
            ax.plot(epochs, ranges, marker='d', label=f'{method.title()}', 
                   color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Difficulty Range (Max - Min)')
        ax.set_title('Difficulty Value Range Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Correlation evolution
        ax = axes[1, 2]
        for i, method in enumerate(methods):
            data = self.threshold_evolution[method]
            df = pd.DataFrame(data)
            
            epochs = df['epoch']
            correlations = df['difficulty_correlation']
            ax.plot(epochs, correlations, marker='*', label=f'{method.title()}', 
                   color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Pearson Correlation (Difficulty vs Error)')
        ax.set_title('Difficulty-Accuracy Correlation')
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'threshold_evolution_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comprehensive plot saved: {output_dir / 'threshold_evolution_comprehensive.png'}")
        
        # Create individual method plots
        self.create_individual_threshold_plots(output_dir)
    
    def create_individual_threshold_plots(self, output_dir: Path):
        """Create detailed plots for each method individually."""
        
        for method in self.threshold_evolution.keys():
            data = self.threshold_evolution[method]
            df = pd.DataFrame(data)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Threshold Analysis: {method.title()} Method', fontsize=16, fontweight='bold')
            
            # Plot 1: Multiple threshold methods comparison
            ax = axes[0, 0]
            epochs = df['epoch']
            
            # Plot different threshold methods
            valid_opt = ~df['optimal_threshold_max_diff'].isna()
            if valid_opt.any():
                ax.plot(epochs[valid_opt], df[valid_opt]['optimal_threshold_max_diff'], 
                       marker='o', label='Max Diff', linewidth=2)
            
            valid_p50 = ~df['threshold_p50'].isna()
            if valid_p50.any():
                ax.plot(epochs[valid_p50], df[valid_p50]['threshold_p50'], 
                       marker='s', label='Median (P50)', linewidth=2)
            
            valid_bimodal = ~df['bimodal_threshold'].isna()
            if valid_bimodal.any():
                ax.plot(epochs[valid_bimodal], df[valid_bimodal]['bimodal_threshold'], 
                       marker='^', label='Bimodal Approx', linewidth=2)
            
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Threshold Value')
            ax.set_title('Threshold Methods Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Accuracy difference and overall accuracy
            ax = axes[0, 1]
            ax2 = ax.twinx()
            
            line1 = ax.plot(epochs, df['max_accuracy_difference'], 'b-o', 
                           label='Max Accuracy Difference', linewidth=2)
            line2 = ax2.plot(epochs, df['overall_accuracy'], 'r-s', 
                            label='Overall Accuracy', linewidth=2)
            
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Accuracy Difference (Easy - Hard)', color='b')
            ax2.set_ylabel('Overall Accuracy', color='r')
            ax.set_title('Discrimination vs Overall Performance')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Easy vs Hard accuracy with gap
            ax = axes[1, 0]
            valid_mask = ~df['easy_accuracy'].isna() & ~df['hard_accuracy'].isna()
            if valid_mask.any():
                epochs_valid = epochs[valid_mask]
                easy_acc = df[valid_mask]['easy_accuracy']
                hard_acc = df[valid_mask]['hard_accuracy']
                
                ax.plot(epochs_valid, easy_acc, 'g-o', label='Easy Samples', linewidth=2)
                ax.plot(epochs_valid, hard_acc, 'r-s', label='Hard Samples', linewidth=2)
                ax.fill_between(epochs_valid, easy_acc, hard_acc, alpha=0.2, color='gray')
            
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Easy vs Hard Sample Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Sample distribution (easy vs hard)
            ax = axes[1, 1]
            valid_mask = (df['easy_samples'] > 0) & (df['hard_samples'] > 0)
            if valid_mask.any():
                epochs_valid = epochs[valid_mask]
                easy_samples = df[valid_mask]['easy_samples']
                hard_samples = df[valid_mask]['hard_samples']
                total_samples = easy_samples + hard_samples
                
                easy_pct = easy_samples / total_samples * 100
                hard_pct = hard_samples / total_samples * 100
                
                ax.plot(epochs_valid, easy_pct, 'g-o', label='Easy Samples %', linewidth=2)
                ax.plot(epochs_valid, hard_pct, 'r-s', label='Hard Samples %', linewidth=2)
            
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Percentage of Samples')
            ax.set_title('Easy vs Hard Sample Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'threshold_analysis_{method}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Individual plot saved: {output_dir / f'threshold_analysis_{method}.png'}")
    
    def save_threshold_data(self, output_dir: Path):
        """Save threshold analysis data to CSV files."""
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for method, data in self.threshold_evolution.items():
            df = pd.DataFrame(data)
            csv_file = output_dir / f'threshold_evolution_{method}.csv'
            df.to_csv(csv_file, index=False)
            print(f"💾 Saved {method} data: {csv_file}")
        
        # Create summary table
        summary_data = []
        for method, data in self.threshold_evolution.items():
            df = pd.DataFrame(data)
            
            # Calculate summary statistics
            valid_thresholds = df[~df['optimal_threshold_max_diff'].isna()]
            if len(valid_thresholds) > 0:
                summary = {
                    'method': method,
                    'epochs_analyzed': len(df),
                    'valid_thresholds': len(valid_thresholds),
                    'initial_threshold': valid_thresholds.iloc[0]['optimal_threshold_max_diff'] if len(valid_thresholds) > 0 else np.nan,
                    'final_threshold': valid_thresholds.iloc[-1]['optimal_threshold_max_diff'] if len(valid_thresholds) > 0 else np.nan,
                    'threshold_change': (valid_thresholds.iloc[-1]['optimal_threshold_max_diff'] - 
                                       valid_thresholds.iloc[0]['optimal_threshold_max_diff']) if len(valid_thresholds) > 1 else 0,
                    'max_discrimination': df['max_accuracy_difference'].max(),
                    'final_overall_accuracy': df.iloc[-1]['overall_accuracy'],
                    'avg_correlation': df['difficulty_correlation'].mean()
                }
                summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / 'threshold_evolution_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"📋 Summary table saved: {summary_file}")
        
        return summary_df

def main():
    """Main function to analyze optimal thresholds."""
    
    print("🎯 Optimal Threshold Analysis")
    print("=" * 50)
    
    # Configuration
    results_file = "working_batch_difficulty_analysis/aggregated_results.json"
    output_dir = Path("threshold_analysis")
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("Please run the batch analysis first to generate aggregated results.")
        return 1
    
    # Initialize analyzer
    analyzer = ThresholdAnalyzer(results_file)
    
    # Analyze all methods and epochs
    analyzer.analyze_all_methods()
    
    # Create plots
    print(f"\n📊 Creating threshold evolution plots...")
    analyzer.create_threshold_evolution_plots(output_dir)
    
    # Save data
    print(f"\n💾 Saving threshold analysis data...")
    summary_df = analyzer.save_threshold_data(output_dir)
    
    # Print summary
    print(f"\n📋 Summary of Threshold Evolution:")
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    print(f"\n🎉 Threshold analysis complete!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"📊 Plots: threshold_evolution_comprehensive.png, threshold_analysis_<method>.png")
    print(f"📈 Data: threshold_evolution_<method>.csv, threshold_evolution_summary.csv")
    
    return 0

if __name__ == "__main__":
    exit(main()) 