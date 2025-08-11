#!/usr/bin/env python3
"""
Simple Threshold Evolution Plot - Just the essentials!

Two plots only:
1. Epochs vs Optimal Threshold (how threshold changes during training)
2. Epochs vs Accuracy Difference (how well we can separate easy/hard samples)

Now includes mode-based and bimodal valley detection methods!
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def find_optimal_threshold(bins):
    """Find threshold that maximizes accuracy difference between easy and hard samples."""
    
    centers = np.array(bins['centers'])
    accuracies = np.array(bins['accuracies'])
    counts = np.array(bins['counts'])
    
    # Remove empty bins
    valid_mask = counts > 0
    if not valid_mask.any() or len(centers[valid_mask]) < 3:
        return np.nan, 0
    
    centers = centers[valid_mask]
    accuracies = accuracies[valid_mask]
    counts = counts[valid_mask]
    
    best_threshold = np.nan
    max_diff = 0
    
    # Try each threshold
    for i in range(1, len(centers) - 1):
        threshold = centers[i]
        
        # Easy samples (below threshold)
        easy_mask = centers <= threshold
        hard_mask = centers > threshold
        
        if not easy_mask.any() or not hard_mask.any():
            continue
        
        # Calculate weighted accuracies
        easy_acc = np.average(accuracies[easy_mask], weights=counts[easy_mask])
        hard_acc = np.average(accuracies[hard_mask], weights=counts[hard_mask])
        
        diff = easy_acc - hard_acc
        
        if diff > max_diff:
            max_diff = diff
            best_threshold = threshold
    
    return best_threshold, max_diff

def find_mode_and_valley_thresholds(bins):
    """Find thresholds based on mode and bimodal valley detection."""
    
    centers = np.array(bins['centers'])
    counts = np.array(bins['counts'])
    
    # Remove empty bins
    valid_mask = counts > 0
    if not valid_mask.any():
        return np.nan, np.nan
    
    centers = centers[valid_mask]
    counts = counts[valid_mask]
    
    # Create expanded array of difficulty values for histogram
    difficulties = []
    for center, count in zip(centers, counts):
        difficulties.extend([center] * int(count))
    
    if len(difficulties) < 50:
        return np.nan, np.nan
    
    difficulties = np.array(difficulties)
    
    # 1. Mode-based threshold
    hist, bins_hist = np.histogram(difficulties, bins=50)
    mode_idx = np.argmax(hist)
    mode_value = (bins_hist[mode_idx] + bins_hist[mode_idx + 1]) / 2
    
    # 2. Bimodal valley detection
    valley_threshold = np.nan
    
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
            valley_start = min(peak1, peak2)
            valley_end = max(peak1, peak2)
            if valley_end > valley_start:
                valley_idx = np.argmin(hist_smooth[valley_start:valley_end]) + valley_start
                valley_threshold = (bins_hist[valley_idx] + bins_hist[valley_idx + 1]) / 2
    
    return mode_value, valley_threshold

def main():
    print("📊 Simple Threshold Evolution Analysis (with Mode & Valley Detection)")
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
    colors = ['blue', 'red', 'green', 'orange']
    
    # Create the plots - now with 3 threshold methods
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Difficulty Threshold Evolution During Training', fontsize=16, fontweight='bold')
    
    for i, method in enumerate(methods):
        if method not in results:
            continue
        
        epochs = []
        max_diff_thresholds = []
        mode_thresholds = []
        valley_thresholds = []
        accuracy_diffs = []
        
        # Process each epoch
        for epoch_str in sorted(results[method].keys(), key=int):
            epoch = int(epoch_str)
            epoch_data = results[method][epoch_str]
            
            # Method 1: Max difference threshold
            threshold, acc_diff = find_optimal_threshold(epoch_data['bins'])
            
            # Method 2 & 3: Mode and valley thresholds
            mode_thresh, valley_thresh = find_mode_and_valley_thresholds(epoch_data['bins'])
            
            if not np.isnan(threshold):
                epochs.append(epoch)
                max_diff_thresholds.append(threshold)
                mode_thresholds.append(mode_thresh)
                valley_thresholds.append(valley_thresh)
                accuracy_diffs.append(acc_diff)
        
        if epochs:
            # Plot 1: Max Difference Threshold (original)
            ax1.plot(epochs, max_diff_thresholds, marker='o', label=f'{method.title()}', 
                    color=colors[i], linewidth=2, markersize=6)
            
            # Plot 2: Mode-based Threshold
            valid_mode = [not np.isnan(x) for x in mode_thresholds]
            if any(valid_mode):
                epochs_mode = [e for e, v in zip(epochs, valid_mode) if v]
                mode_vals = [x for x, v in zip(mode_thresholds, valid_mode) if v]
                ax2.plot(epochs_mode, mode_vals, marker='s', label=f'{method.title()}', 
                        color=colors[i], linewidth=2, markersize=6)
            
            # Plot 3: Valley-based Threshold (Bimodal)
            valid_valley = [not np.isnan(x) for x in valley_thresholds]
            if any(valid_valley):
                epochs_valley = [e for e, v in zip(epochs, valid_valley) if v]
                valley_vals = [x for x, v in zip(valley_thresholds, valid_valley) if v]
                ax3.plot(epochs_valley, valley_vals, marker='^', label=f'{method.title()}', 
                        color=colors[i], linewidth=2, markersize=6)
            
            # Plot 4: Discrimination Power (same as before)
            ax4.plot(epochs, accuracy_diffs, marker='d', label=f'{method.title()}', 
                    color=colors[i], linewidth=2, markersize=6)
            
            print(f"{method:12s}: Max-diff {max_diff_thresholds[0]:.3f} → {max_diff_thresholds[-1]:.3f} "
                  f"(Δ: {max_diff_thresholds[-1] - max_diff_thresholds[0]:+.3f})")
    
    # Style Plot 1: Max Difference Method
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Optimal Difficulty Threshold')
    ax1.set_title('Method 1: Max Accuracy Difference')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Style Plot 2: Mode Method
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Mode Difficulty Threshold')
    ax2.set_title('Method 2: Distribution Mode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Style Plot 3: Valley Method
    ax3.set_xlabel('Training Epoch')
    ax3.set_ylabel('Valley Difficulty Threshold')
    ax3.set_title('Method 3: Bimodal Valley')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Style Plot 4: Discrimination Power
    ax4.set_xlabel('Training Epoch')
    ax4.set_ylabel('Accuracy Difference (Easy - Hard)')
    ax4.set_title('Discrimination Power')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Plot saved: threshold_methods_comparison.png")
    print(f"\n🎯 What the four plots show:")
    print(f"   Top Left:    Method 1 - Max accuracy difference threshold")
    print(f"   Top Right:   Method 2 - Mode of difficulty distribution")
    print(f"   Bottom Left: Method 3 - Valley between bimodal peaks")
    print(f"   Bottom Right: How well each method separates easy/hard")

if __name__ == "__main__":
    main() 