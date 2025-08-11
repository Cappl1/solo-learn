#!/usr/bin/env python3
"""
CORRECT Bimodal Analysis - Separate Correct vs Incorrect Samples!

This is what we SHOULD have been doing:
1. Correct samples → Find their difficulty peak
2. Incorrect samples → Find their difficulty peak  
3. Valley between peaks = Natural threshold!

This reveals if difficulty measures actually separate correct from incorrect predictions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy import stats

def find_correct_vs_incorrect_threshold(bins, overall_accuracy):
    """
    Find the natural threshold by analyzing correct vs incorrect sample distributions.
    
    This is the CORRECT way to do bimodal analysis!
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
    
    # Find peaks (modes) for each distribution
    def find_peak(hist_smooth):
        peak_idx = np.argmax(hist_smooth)
        return peak_idx, hist_smooth[peak_idx]
    
    correct_peak_idx, correct_peak_height = find_peak(hist_correct_smooth)
    incorrect_peak_idx, incorrect_peak_height = find_peak(hist_incorrect_smooth)
    
    # Convert bin indices to actual difficulty values
    bin_centers = (bins_hist[:-1] + bins_hist[1:]) / 2
    correct_peak_difficulty = bin_centers[correct_peak_idx]
    incorrect_peak_difficulty = bin_centers[incorrect_peak_idx]
    
    # Find valley between peaks
    valley_threshold = np.nan
    separation_quality = 0
    
    if correct_peak_idx != incorrect_peak_idx:
        # Find valley between the two peaks
        start_idx = min(correct_peak_idx, incorrect_peak_idx)
        end_idx = max(correct_peak_idx, incorrect_peak_idx)
        
        if end_idx > start_idx:
            # Look for minimum in the combined distribution
            combined_hist = hist_correct_smooth + hist_incorrect_smooth
            valley_idx = start_idx + np.argmin(combined_hist[start_idx:end_idx])
            valley_threshold = bin_centers[valley_idx]
            
            # Calculate separation quality
            peak_distance = abs(incorrect_peak_difficulty - correct_peak_difficulty)
            valley_depth = min(correct_peak_height, incorrect_peak_height) - combined_hist[valley_idx]
            separation_quality = peak_distance * valley_depth
    
    # Calculate overlap between distributions
    overlap = np.minimum(hist_correct_smooth, hist_incorrect_smooth).sum()
    total_area = (hist_correct_smooth.sum() + hist_incorrect_smooth.sum()) / 2
    overlap_ratio = overlap / total_area if total_area > 0 else 1.0
    
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
        'separation_quality': separation_quality,
        'overlap_ratio': overlap_ratio,
        'peak_distance': abs(incorrect_peak_difficulty - correct_peak_difficulty),
        'is_well_separated': separation_quality > 0.1 and overlap_ratio < 0.7,
        'correct_stats': correct_stats,
        'incorrect_stats': incorrect_stats,
        'histograms': {
            'bins': bin_centers,
            'correct': hist_correct_smooth,
            'incorrect': hist_incorrect_smooth
        }
    }
    
    return valley_threshold, correct_stats, incorrect_stats, analysis

def analyze_all_epochs():
    """Analyze bimodal separation for all epochs and methods."""
    
    print("🎯 CORRECT Bimodal Analysis: Separating Correct vs Incorrect Samples")
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
            
            valley_thresh, correct_stats, incorrect_stats, analysis = find_correct_vs_incorrect_threshold(
                epoch_data['bins'], epoch_data['overall_accuracy']
            )
            
            if not np.isnan(valley_thresh):
                epoch_analysis = {
                    'epoch': epoch,
                    'valley_threshold': valley_thresh,
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
                print(f"  Epoch {epoch:2d}: {status} | "
                      f"Correct peak: {correct_stats['peak_difficulty']:.3f}, "
                      f"Incorrect peak: {incorrect_stats['peak_difficulty']:.3f}, "
                      f"Valley: {valley_thresh:.3f}")
        
        method_analyses[method] = epoch_analyses
    
    return method_analyses

def create_bimodal_evolution_plots(method_analyses):
    """Create plots showing bimodal separation evolution."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bimodal Separation Evolution: Correct vs Incorrect Samples', fontsize=16, fontweight='bold')
    
    methods = list(method_analyses.keys())
    colors = ['blue', 'red', 'green', 'orange'][:len(methods)]
    
    for i, method in enumerate(methods):
        if not method_analyses[method]:
            continue
        
        data = method_analyses[method]
        epochs = [d['epoch'] for d in data]
        
        # Plot 1: Valley threshold evolution
        valley_thresholds = [d['valley_threshold'] for d in data]
        axes[0, 0].plot(epochs, valley_thresholds, marker='o', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=6)
        
        # Plot 2: Peak separation (distance between correct and incorrect peaks)
        peak_distances = [d['peak_distance'] for d in data]
        axes[0, 1].plot(epochs, peak_distances, marker='s', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=6)
        
        # Plot 3: Separation quality
        separation_qualities = [d['separation_quality'] for d in data]
        axes[1, 0].plot(epochs, separation_qualities, marker='^', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=6)
        
        # Plot 4: Overlap ratio (lower = better separation)
        overlap_ratios = [d['overlap_ratio'] for d in data]
        axes[1, 1].plot(epochs, overlap_ratios, marker='d', label=f'{method.title()}', 
                       color=colors[i], linewidth=2, markersize=6)
    
    # Style plots
    axes[0, 0].set_xlabel('Training Epoch')
    axes[0, 0].set_ylabel('Valley Threshold')
    axes[0, 0].set_title('Natural Threshold Between Correct/Incorrect')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Training Epoch')
    axes[0, 1].set_ylabel('Peak Distance')
    axes[0, 1].set_title('Distance Between Correct/Incorrect Peaks')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Training Epoch')
    axes[1, 0].set_ylabel('Separation Quality')
    axes[1, 0].set_title('How Well Separated Are The Distributions?')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Training Epoch')
    axes[1, 1].set_ylabel('Overlap Ratio (Lower = Better)')
    axes[1, 1].set_title('Distribution Overlap (0 = Perfect Separation)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correct_bimodal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Plot saved: correct_bimodal_analysis.png")

def create_example_distribution_plot(method_analyses):
    """Create example plots showing the actual distributions for a few epochs."""
    
    # Pick entropy method and a few representative epochs
    if 'entropy' not in method_analyses or not method_analyses['entropy']:
        print("No entropy data available for distribution plots")
        return
    
    # Load detailed data for distribution plotting
    results_file = "working_batch_difficulty_analysis/aggregated_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Plot distributions for epochs 0, 10, 19
    epochs_to_plot = [0, 10, 19]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Entropy Distributions: Correct vs Incorrect Samples', fontsize=16, fontweight='bold')
    
    for i, epoch in enumerate(epochs_to_plot):
        if str(epoch) not in results['entropy']:
            continue
        
        epoch_data = results['entropy'][str(epoch)]
        valley_thresh, correct_stats, incorrect_stats, analysis = find_correct_vs_incorrect_threshold(
            epoch_data['bins'], epoch_data['overall_accuracy']
        )
        
        if 'histograms' in analysis:
            bins = analysis['histograms']['bins']
            hist_correct = analysis['histograms']['correct']
            hist_incorrect = analysis['histograms']['incorrect']
            
            ax = axes[i]
            ax.fill_between(bins, hist_correct, alpha=0.6, color='green', label='Correct Samples')
            ax.fill_between(bins, hist_incorrect, alpha=0.6, color='red', label='Incorrect Samples')
            
            if not np.isnan(valley_thresh):
                ax.axvline(valley_thresh, color='black', linestyle='--', linewidth=2, 
                          label=f'Valley Threshold: {valley_thresh:.3f}')
            
            ax.axvline(correct_stats['peak_difficulty'], color='green', linestyle='-', alpha=0.8,
                      label=f'Correct Peak: {correct_stats["peak_difficulty"]:.3f}')
            ax.axvline(incorrect_stats['peak_difficulty'], color='red', linestyle='-', alpha=0.8,
                      label=f'Incorrect Peak: {incorrect_stats["peak_difficulty"]:.3f}')
            
            ax.set_xlabel('Entropy Difficulty')
            ax.set_ylabel('Density')
            ax.set_title(f'Epoch {epoch} (Acc: {epoch_data["overall_accuracy"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entropy_distributions_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Distribution plot saved: entropy_distributions_evolution.png")

def main():
    """Main function for correct bimodal analysis."""
    
    # Run the analysis
    method_analyses = analyze_all_epochs()
    
    if not method_analyses:
        print("❌ No analysis results generated")
        return
    
    # Create plots
    print(f"\n📊 Creating bimodal evolution plots...")
    create_bimodal_evolution_plots(method_analyses)
    
    print(f"\n📊 Creating distribution examples...")
    create_example_distribution_plot(method_analyses)
    
    # Print summary
    print(f"\n📋 Summary of Bimodal Separation:")
    for method, data in method_analyses.items():
        if data:
            well_separated_epochs = sum(1 for d in data if d['is_well_separated'])
            print(f"{method:12s}: {well_separated_epochs}/{len(data)} epochs show good separation")
    
    print(f"\n🎯 What this reveals:")
    print(f"   - Whether difficulty measures actually separate correct from incorrect")
    print(f"   - How separation quality evolves during training")
    print(f"   - Which methods create meaningful bimodal distributions")
    print(f"   - Natural thresholds based on sample correctness")

if __name__ == "__main__":
    main() 