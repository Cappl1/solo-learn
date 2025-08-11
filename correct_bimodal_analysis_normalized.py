#!/usr/bin/env python3
"""
CORRECTED Bimodal Analysis - Normalized Valley Threshold

PROBLEM: Previous valley calculation was biased toward minority class!
SOLUTION: Normalize both distributions first, then find true intersection point.

This finds where P(correct|difficulty) = P(incorrect|difficulty)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy import stats

def find_normalized_valley_threshold(bins, overall_accuracy):
    """
    Find the UNBIASED threshold by normalizing distributions first.
    
    This finds the intersection point where P(correct|difficulty) = P(incorrect|difficulty)
    instead of being biased toward the minority class.
    """
    centers = np.array(bins['centers'])
    accuracies = np.array(bins['accuracies'])
    counts = np.array(bins['counts'])
    
    # Remove empty bins
    valid_mask = counts > 0
    if not valid_mask.any():
        return np.nan, {}, {}
    
    centers = centers[valid_mask]
    accuracies = accuracies[valid_mask]
    counts = counts[valid_mask]
    
    # Reconstruct individual samples (approximately)
    difficulties = []
    correctness = []
    
    for center, accuracy, count in zip(centers, accuracies, counts):
        # Add samples for this bin
        num_correct = int(accuracy * count)
        num_incorrect = count - num_correct
        
        # Correct samples at this difficulty level
        difficulties.extend([center] * num_correct)
        correctness.extend([1] * num_correct)
        
        # Incorrect samples at this difficulty level
        difficulties.extend([center] * num_incorrect)
        correctness.extend([0] * num_incorrect)
    
    if len(difficulties) < 50:
        return np.nan, {}, {}
    
    difficulties = np.array(difficulties)
    correctness = np.array(correctness)
    
    # Separate correct and incorrect samples
    diff_correct = difficulties[correctness == 1]
    diff_incorrect = difficulties[correctness == 0]
    
    if len(diff_correct) < 10 or len(diff_incorrect) < 10:
        return np.nan, {}, {}
    
    # Create histograms for each group
    bin_range = (difficulties.min(), difficulties.max())
    hist_correct, bins_hist = np.histogram(diff_correct, bins=50, range=bin_range, density=True)
    hist_incorrect, _ = np.histogram(diff_incorrect, bins=50, range=bin_range, density=True)
    
    # Smooth histograms
    hist_correct_smooth = gaussian_filter1d(hist_correct, sigma=2)
    hist_incorrect_smooth = gaussian_filter1d(hist_incorrect, sigma=2)
    
    # CRITICAL FIX: Normalize both distributions to sum to 1
    # This removes class imbalance bias!
    hist_correct_norm = hist_correct_smooth / np.sum(hist_correct_smooth)
    hist_incorrect_norm = hist_incorrect_smooth / np.sum(hist_incorrect_smooth)
    
    # Find peaks (modes) for each distribution
    def find_peak(hist_smooth):
        peak_idx = np.argmax(hist_smooth)
        return peak_idx, hist_smooth[peak_idx]
    
    correct_peak_idx, correct_peak_height = find_peak(hist_correct_norm)
    incorrect_peak_idx, incorrect_peak_height = find_peak(hist_incorrect_norm)
    
    # Convert bin indices to actual difficulty values
    bin_centers = (bins_hist[:-1] + bins_hist[1:]) / 2
    correct_peak_difficulty = bin_centers[correct_peak_idx]
    incorrect_peak_difficulty = bin_centers[incorrect_peak_idx]
    
    # Find NORMALIZED valley between peaks
    valley_threshold = np.nan
    separation_quality = 0
    intersection_threshold = np.nan
    
    if correct_peak_idx != incorrect_peak_idx:
        # Method 1: Find valley in normalized combined distribution
        combined_norm = hist_correct_norm + hist_incorrect_norm
        start_idx = min(correct_peak_idx, incorrect_peak_idx)
        end_idx = max(correct_peak_idx, incorrect_peak_idx)
        
        if end_idx > start_idx:
            valley_idx = start_idx + np.argmin(combined_norm[start_idx:end_idx])
            valley_threshold = bin_centers[valley_idx]
            
            # Calculate separation quality
            peak_distance = abs(incorrect_peak_difficulty - correct_peak_difficulty)
            valley_depth = min(correct_peak_height, incorrect_peak_height) - combined_norm[valley_idx]
            separation_quality = peak_distance * valley_depth
        
        # Method 2: Find intersection point where P(correct) = P(incorrect)
        # This is the true unbiased threshold!
        diff_abs = np.abs(hist_correct_norm - hist_incorrect_norm)
        intersection_idx = np.argmin(diff_abs)
        intersection_threshold = bin_centers[intersection_idx]
    
    # Calculate overlap between NORMALIZED distributions
    overlap_norm = np.minimum(hist_correct_norm, hist_incorrect_norm).sum()
    
    correct_stats = {
        'peak_difficulty': correct_peak_difficulty,
        'peak_height': correct_peak_height,
        'mean_difficulty': diff_correct.mean(),
        'std_difficulty': diff_correct.std(),
        'count': len(diff_correct)
    }
    
    incorrect_stats = {
        'peak_difficulty': incorrect_peak_difficulty,
        'peak_height': incorrect_peak_height,
        'mean_difficulty': diff_incorrect.mean(),
        'std_difficulty': diff_incorrect.std(),
        'count': len(diff_incorrect)
    }
    
    analysis = {
        'valley_threshold': valley_threshold,
        'intersection_threshold': intersection_threshold,  # NEW: Unbiased threshold
        'separation_quality': separation_quality,
        'overlap_ratio': overlap_norm,
        'peak_distance': abs(incorrect_peak_difficulty - correct_peak_difficulty),
        'is_well_separated': separation_quality > 0.1 and overlap_norm < 0.7,
        'correct_stats': correct_stats,
        'incorrect_stats': incorrect_stats,
        'histograms': {
            'bins': bin_centers,
            'correct': hist_correct_norm,  # Normalized
            'incorrect': hist_incorrect_norm,  # Normalized
            'correct_raw': hist_correct_smooth,  # Raw for comparison
            'incorrect_raw': hist_incorrect_smooth  # Raw for comparison
        }
    }
    
    return intersection_threshold, correct_stats, incorrect_stats, analysis

def analyze_all_epochs_normalized():
    """Analyze bimodal separation with normalized valley calculation."""
    
    print("🎯 CORRECTED Bimodal Analysis: Normalized Valley Thresholds")
    print("=" * 70)
    print("🔧 FIX: Normalizing distributions to remove class imbalance bias")
    print("📊 Finding intersection where P(correct) = P(incorrect)")
    print("=" * 70)
    
    # Load results
    results_file = "working_batch_difficulty_analysis/aggregated_results.json"
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Analyze each method
    methods = ['entropy', 'margin', 'reconstruction', 'pixel_entropy']
    method_analyses = {}
    
    for method in methods:
        if method not in results:
            continue
        
        print(f"\n📊 Analyzing {method.upper()} method...")
        
        epoch_analyses = []
        
        for epoch_str in sorted(results[method].keys(), key=int):
            epoch = int(epoch_str)
            epoch_data = results[method][epoch_str]
            
            intersection_thresh, correct_stats, incorrect_stats, analysis = find_normalized_valley_threshold(
                epoch_data['bins'], epoch_data['overall_accuracy']
            )
            
            if not np.isnan(intersection_thresh):
                epoch_analysis = {
                    'epoch': epoch,
                    'intersection_threshold': intersection_thresh,
                    'valley_threshold': analysis['valley_threshold'],
                    'separation_quality': analysis['separation_quality'],
                    'overlap_ratio': analysis['overlap_ratio'],
                    'peak_distance': analysis['peak_distance'],
                    'is_well_separated': analysis['is_well_separated'],
                    'correct_peak': correct_stats['peak_difficulty'],
                    'incorrect_peak': incorrect_stats['peak_difficulty'],
                    'correct_mean': correct_stats['mean_difficulty'],
                    'incorrect_mean': incorrect_stats['mean_difficulty'],
                    'overall_accuracy': epoch_data['overall_accuracy']
                }
                epoch_analyses.append(epoch_analysis)
                
                status = "✅ SEPARATED" if analysis['is_well_separated'] else "❌ OVERLAPPED"
                bias_diff = abs(intersection_thresh - analysis['valley_threshold'])
                print(f"  Epoch {epoch:2d}: {status} | "
                      f"Correct: {correct_stats['peak_difficulty']:.3f}, "
                      f"Incorrect: {incorrect_stats['peak_difficulty']:.3f}")
                print(f"           Intersection: {intersection_thresh:.3f}, "
                      f"Valley: {analysis['valley_threshold']:.3f} "
                      f"(bias: {bias_diff:.3f})")
        
        method_analyses[method] = epoch_analyses
    
    return method_analyses

def create_normalized_comparison_plots(method_analyses):
    """Create plots comparing normalized vs biased thresholds."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Normalized vs Biased Threshold Comparison', fontsize=16, fontweight='bold')
    
    methods = list(method_analyses.keys())
    colors = ['blue', 'red', 'green', 'orange'][:len(methods)]
    
    for i, method in enumerate(methods):
        if not method_analyses[method]:
            continue
        
        data = method_analyses[method]
        epochs = [d['epoch'] for d in data]
        
        # Plot 1: Intersection vs Valley thresholds
        intersection_thresholds = [d['intersection_threshold'] for d in data]
        valley_thresholds = [d['valley_threshold'] for d in data]
        
        axes[0, 0].plot(epochs, intersection_thresholds, marker='o', label=f'{method.title()} (Intersection)', 
                       color=colors[i], linewidth=2, markersize=6)
        axes[0, 0].plot(epochs, valley_thresholds, marker='s', label=f'{method.title()} (Valley)', 
                       color=colors[i], linewidth=2, markersize=4, alpha=0.7, linestyle='--')
        
        # Plot 2: Bias magnitude (difference between methods)
        bias_magnitudes = [abs(d['intersection_threshold'] - d['valley_threshold']) for d in data]
        axes[0, 1].plot(epochs, bias_magnitudes, marker='^', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=6)
        
        # Plot 3: Peak separation
        peak_distances = [d['peak_distance'] for d in data]
        axes[1, 0].plot(epochs, peak_distances, marker='d', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=6)
        
        # Plot 4: Separation quality
        separation_qualities = [d['separation_quality'] for d in data]
        axes[1, 1].plot(epochs, separation_qualities, marker='*', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=8)
    
    # Style plots
    axes[0, 0].set_xlabel('Training Epoch')
    axes[0, 0].set_ylabel('Threshold Value')
    axes[0, 0].set_title('Intersection vs Valley Thresholds')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Training Epoch')
    axes[0, 1].set_ylabel('Bias Magnitude')
    axes[0, 1].set_title('Threshold Bias (|Intersection - Valley|)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Training Epoch')
    axes[1, 0].set_ylabel('Peak Distance')
    axes[1, 0].set_title('Distance Between Correct/Incorrect Peaks')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Training Epoch')
    axes[1, 1].set_ylabel('Separation Quality')
    axes[1, 1].set_title('Bimodal Separation Quality')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('normalized_threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Comparison plot saved: normalized_threshold_comparison.png")

def create_normalized_distribution_examples(method_analyses):
    """Show normalized vs raw distributions for key epochs."""
    
    # Load detailed data
    results_file = "working_batch_difficulty_analysis/aggregated_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if 'entropy' not in method_analyses or not method_analyses['entropy']:
        print("No entropy data for distribution examples")
        return
    
    # Plot for epochs 0, 14, 19 (early, breakthrough, late)
    epochs_to_plot = [14, 19]  # Focus on epochs with good separation
    fig, axes = plt.subplots(2, len(epochs_to_plot), figsize=(6*len(epochs_to_plot), 12))
    if len(epochs_to_plot) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Normalized vs Raw Distributions: Entropy Method', fontsize=16, fontweight='bold')
    
    for i, epoch in enumerate(epochs_to_plot):
        if str(epoch) not in results['entropy']:
            continue
        
        epoch_data = results['entropy'][str(epoch)]
        intersection_thresh, correct_stats, incorrect_stats, analysis = find_normalized_valley_threshold(
            epoch_data['bins'], epoch_data['overall_accuracy']
        )
        
        if 'histograms' not in analysis:
            continue
        
        bins = analysis['histograms']['bins']
        
        # Raw distributions (top row)
        ax_raw = axes[0, i]
        hist_correct_raw = analysis['histograms']['correct_raw']
        hist_incorrect_raw = analysis['histograms']['incorrect_raw']
        
        ax_raw.fill_between(bins, hist_correct_raw, alpha=0.6, color='green', label='Correct (Raw)')
        ax_raw.fill_between(bins, hist_incorrect_raw, alpha=0.6, color='red', label='Incorrect (Raw)')
        
        # Valley threshold (biased)
        ax_raw.axvline(analysis['valley_threshold'], color='orange', linestyle='--', linewidth=2, 
                      label=f'Valley: {analysis["valley_threshold"]:.3f}')
        
        ax_raw.set_xlabel('Entropy Difficulty')
        ax_raw.set_ylabel('Raw Density')
        ax_raw.set_title(f'Raw Distributions - Epoch {epoch}')
        ax_raw.legend()
        ax_raw.grid(True, alpha=0.3)
        
        # Normalized distributions (bottom row)
        ax_norm = axes[1, i]
        hist_correct_norm = analysis['histograms']['correct']
        hist_incorrect_norm = analysis['histograms']['incorrect']
        
        ax_norm.fill_between(bins, hist_correct_norm, alpha=0.6, color='green', label='Correct (Normalized)')
        ax_norm.fill_between(bins, hist_incorrect_norm, alpha=0.6, color='red', label='Incorrect (Normalized)')
        
        # Intersection threshold (unbiased)
        ax_norm.axvline(intersection_thresh, color='blue', linestyle='-', linewidth=2, 
                       label=f'Intersection: {intersection_thresh:.3f}')
        
        # Show both thresholds for comparison
        ax_norm.axvline(analysis['valley_threshold'], color='orange', linestyle='--', linewidth=1, alpha=0.7,
                       label=f'Valley: {analysis["valley_threshold"]:.3f}')
        
        ax_norm.set_xlabel('Entropy Difficulty')
        ax_norm.set_ylabel('Normalized Density')
        ax_norm.set_title(f'Normalized Distributions - Epoch {epoch}')
        ax_norm.legend()
        ax_norm.grid(True, alpha=0.3)
        
        # Add accuracy info
        acc = epoch_data['overall_accuracy']
        ax_raw.text(0.05, 0.95, f'Accuracy: {acc:.3f}', transform=ax_raw.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_norm.text(0.05, 0.95, f'Accuracy: {acc:.3f}', transform=ax_norm.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('normalized_vs_raw_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Distribution comparison saved: normalized_vs_raw_distributions.png")

def main():
    """Main function for normalized bimodal analysis."""
    
    # Run the corrected analysis
    method_analyses = analyze_all_epochs_normalized()
    
    if not method_analyses:
        print("❌ No analysis results generated")
        return
    
    # Create comparison plots
    print(f"\n📊 Creating normalized threshold comparison plots...")
    create_normalized_comparison_plots(method_analyses)
    
    print(f"\n📊 Creating distribution comparison examples...")
    create_normalized_distribution_examples(method_analyses)
    
    # Print summary with bias analysis
    print(f"\n📋 Summary of Normalized vs Biased Thresholds:")
    for method, data in method_analyses.items():
        if data:
            well_separated_epochs = sum(1 for d in data if d['is_well_separated'])
            avg_bias = np.mean([abs(d['intersection_threshold'] - d['valley_threshold']) for d in data])
            print(f"{method:12s}: {well_separated_epochs}/{len(data)} epochs separated, "
                  f"avg bias: {avg_bias:.3f}")
    
    print(f"\n🎯 Key Insights:")
    print(f"   - Intersection threshold = unbiased (where P(correct) = P(incorrect))")
    print(f"   - Valley threshold = biased toward minority class")
    print(f"   - Bias magnitude shows how much class imbalance affects threshold")
    print(f"   - Normalized analysis reveals true discrimination power")

if __name__ == "__main__":
    main() 