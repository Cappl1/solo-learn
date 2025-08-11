#!/usr/bin/env python3
"""
Clean Batch Difficulty Analysis Script

This script runs difficulty analysis across multiple checkpoint pairs and creates
threshold evolution plots to track how difficulty thresholds change during training.

Usage: python clean_batch_difficulty_analysis.py
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
import pandas as pd

class CleanBatchDifficultyAnalyzer:
    """Clean batch difficulty analyzer that uses the working evaluate_difficulty.py script."""
    
    def __init__(self, base_dir: str, output_dir: str = "batch_difficulty_analysis"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for results across epochs
        self.results = {
            'entropy': {},
            'reconstruction': {},
            'margin': {}
        }
        
        # Threshold metrics to track
        self.threshold_metrics = {}
    
    def find_checkpoint_pairs(self, max_epochs: int = 20) -> List[Tuple[int, str, str]]:
        """Find available checkpoint pairs (backbone + linear classifier)."""
        
        print("🔍 Searching for checkpoint pairs...")
        
        # Based on the YAML structure, the paths are:
        # Backbone: trained_models/selective_curriculum_mocov3/t60/mocov3-selective-curriculum-jepa-core50-5j35ltq7-ep=X-stp=0.ckpt
        # Linear: trained_models/linear/selective_curriculum_mocov3_t60/selective_curriculum_mocov3_t60_epXX/linear/<wandb_id>/*.ckpt
        
        backbone_dir = self.base_dir / "selective_curriculum_mocov3" / "t60"
        linear_base = self.base_dir / "linear" / "simclr_base"
        
        checkpoint_pairs = []
        
        if not backbone_dir.exists():
            print(f"❌ Backbone directory not found: {backbone_dir}")
            return []
        
        if not linear_base.exists():
            print(f"❌ Linear directory not found: {linear_base}")
            return []
        
        print(f"Checking backbone directory: {backbone_dir}")
        print(f"Checking linear directory: {linear_base}")
        
        # Find backbone checkpoints with stp=0 (epoch end checkpoints)
        for ckpt_file in backbone_dir.glob("*-ep=*-stp=0.ckpt"):
            filename = ckpt_file.name
            
            # Extract epoch number from filename
            try:
                epoch_part = filename.split("-ep=")[1].split("-stp=")[0]
                epoch_num = int(epoch_part)
                
                if epoch_num <= max_epochs:
                    # Look for corresponding linear checkpoint
                    linear_checkpoint = self.find_linear_checkpoint(epoch_num, linear_base)
                    
                    if linear_checkpoint:
                        checkpoint_pairs.append((epoch_num, str(ckpt_file), linear_checkpoint))
                        print(f"  ✅ Found pair for epoch {epoch_num}")
                    else:
                        print(f"  ❌ No linear checkpoint for epoch {epoch_num}")
                        
            except (IndexError, ValueError) as e:
                print(f"  ❌ Could not parse epoch from {filename}: {e}")
                continue
        
        # Sort by epoch number
        checkpoint_pairs.sort(key=lambda x: x[0])
        
        print(f"\n📊 Found {len(checkpoint_pairs)} valid checkpoint pairs")
        for epoch, backbone, linear in checkpoint_pairs:
            print(f"  Epoch {epoch:2d}: {Path(backbone).name}")
            print(f"            {Path(linear).name}")
        
        return checkpoint_pairs
    
    def find_linear_checkpoint(self, epoch_num: int, linear_base: Path) -> Optional[str]:
        """Find the linear checkpoint for a given epoch."""
        
        # Based on the YAML structure, linear checkpoints are in:
        # trained_models/linear/selective_curriculum_mocov3_t60/selective_curriculum_mocov3_t60_epXX/linear/<wandb_id>/*.ckpt
        
        # Look for the epoch directory
        epoch_dir = linear_base / f"selective_curriculum_mocov3_t60_ep{epoch_num:02d}"
        
        if not epoch_dir.exists():
            return None
        
        # Look in the linear subdirectory
        linear_dir = epoch_dir / "linear"
        if not linear_dir.exists():
            return None
        
        # Look for wandb subdirectories
        for wandb_dir in linear_dir.iterdir():
            if wandb_dir.is_dir():
                # Look for .ckpt files in this directory
                ckpt_files = list(wandb_dir.glob("*.ckpt"))
                if ckpt_files:
                    # Prefer "last" checkpoint if available
                    last_ckpts = [f for f in ckpt_files if "last" in f.name]
                    if last_ckpts:
                        return str(last_ckpts[0])
                    else:
                        # Otherwise, take the most recent checkpoint
                        ckpt_files.sort(key=lambda x: x.stat().st_mtime)
                        return str(ckpt_files[-1])
        
        return None
    
    def get_config_path(self) -> str:
        """Get the config path for difficulty evaluation."""
        
        # Use the working difficulty_cur.yaml config
        config_path = "/home/brothen/solo-learn/scripts/linear_probe/core50/difficulty_cur.yaml"
        
        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            print("Please ensure the config file exists or update the path")
            sys.exit(1)
        
        return config_path
    
    def run_difficulty_analysis(self, backbone_path: str, linear_path: str, 
                              config_path: str, epoch_num: int, 
                              difficulty_method: str = 'entropy') -> Optional[Dict]:
        """Run difficulty analysis for a single checkpoint pair."""
        
        print(f"  Running {difficulty_method} analysis for epoch {epoch_num}...")
        
        # Create unique output directory for this epoch and method
        epoch_output_dir = self.output_dir / f"epoch_{epoch_num:02d}_{difficulty_method}"
        epoch_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Change to epoch directory to capture outputs
        original_cwd = os.getcwd()
        os.chdir(epoch_output_dir)
        
        try:
            # Run the working evaluate_difficulty.py script
            cmd = [
                sys.executable, "evaluate_difficulty.py",
                "--model_path", backbone_path,
                "--linear_path", linear_path,
                "--cfg_path", config_path,
                "--difficulty_method", difficulty_method,
                "--batch_size", "64",
                "--num_workers", "4",
                "--device", "cuda"
            ]
            
            result = subprocess.run(cmd, cwd="/home/brothen/solo-learn", 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load the results
                results_file = Path("difficulty_analysis") / f"analysis_results_{difficulty_method}.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        analysis_results = json.load(f)
                    
                    print(f"    ✅ {difficulty_method} analysis completed (acc: {analysis_results['overall_accuracy']:.3f})")
                    return analysis_results
                else:
                    print(f"    ❌ Results file not found: {results_file}")
                    return None
            else:
                print(f"    ❌ {difficulty_method} analysis failed:")
                print(f"    stdout: {result.stdout}")
                print(f"    stderr: {result.stderr}")
                return None
                
        finally:
            os.chdir(original_cwd)
    
    def compute_threshold_metrics(self, analysis_results: Dict, epoch_num: int, method: str):
        """Compute threshold-based metrics from analysis results."""
        
        difficulties = np.array(analysis_results['bins']['centers'])
        accuracies = np.array(analysis_results['bins']['accuracies'])
        counts = np.array(analysis_results['bins']['counts'])
        
        # Remove bins with no samples
        valid_mask = counts > 0
        if not valid_mask.any():
            return
        
        difficulties = difficulties[valid_mask]
        accuracies = accuracies[valid_mask]
        counts = counts[valid_mask]
        
        overall_acc = analysis_results['overall_accuracy']
        
        # Compute various threshold metrics
        metrics = {
            'epoch': epoch_num,
            'overall_accuracy': overall_acc,
            'difficulty_range': difficulties.max() - difficulties.min(),
            'accuracy_drop': accuracies.max() - accuracies.min(),
        }
        
        # Find thresholds where accuracy drops below certain levels
        thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
        for threshold in thresholds:
            target_acc = overall_acc * threshold
            # Find first difficulty level where accuracy drops below target
            below_threshold = accuracies < target_acc
            if below_threshold.any():
                threshold_idx = np.where(below_threshold)[0][0]
                metrics[f'threshold_acc_{int(threshold*100)}'] = difficulties[threshold_idx]
            else:
                metrics[f'threshold_acc_{int(threshold*100)}'] = difficulties.max()
        
        # Easy vs hard accuracy (split at median difficulty)
        median_idx = len(difficulties) // 2
        easy_acc = accuracies[:median_idx].mean() if median_idx > 0 else overall_acc
        hard_acc = accuracies[median_idx:].mean() if median_idx < len(accuracies) else overall_acc
        
        metrics['easy_accuracy'] = easy_acc
        metrics['hard_accuracy'] = hard_acc
        metrics['easy_hard_gap'] = easy_acc - hard_acc
        
        # Correlation metrics
        correlation = analysis_results['correlations']['pearson']['r']
        metrics['difficulty_correlation'] = correlation
        
        # Store metrics
        if method not in self.threshold_metrics:
            self.threshold_metrics[method] = []
        
        self.threshold_metrics[method].append(metrics)
    
    def run_batch_analysis(self, checkpoint_pairs: List[Tuple[int, str, str]], 
                          difficulty_methods: List[str] = ['entropy', 'reconstruction', 'margin']):
        """Run difficulty analysis across all checkpoint pairs."""
        
        config_path = self.get_config_path()
        
        print(f"\n🚀 Starting batch analysis...")
        print(f"📊 Checkpoint pairs: {len(checkpoint_pairs)}")
        print(f"🔬 Difficulty methods: {difficulty_methods}")
        print(f"📁 Output directory: {self.output_dir}")
        
        for epoch_num, backbone_path, linear_path in tqdm(checkpoint_pairs, desc="Processing epochs"):
            print(f"\n📈 Processing Epoch {epoch_num}")
            print(f"  Backbone: {Path(backbone_path).name}")
            print(f"  Linear: {Path(linear_path).name}")
            
            for method in difficulty_methods:
                # Run analysis for this method
                results = self.run_difficulty_analysis(
                    backbone_path, linear_path, config_path, epoch_num, method
                )
                
                if results:
                    # Store results
                    self.results[method][epoch_num] = results
                    
                    # Compute threshold metrics
                    self.compute_threshold_metrics(results, epoch_num, method)
                else:
                    print(f"    ❌ Failed to get results for {method}")
        
        print(f"\n✅ Batch analysis complete!")
        self.save_summary_results()
    
    def save_summary_results(self):
        """Save summary results and create plots."""
        
        print(f"\n💾 Saving summary results...")
        
        # Save threshold metrics
        for method, metrics_list in self.threshold_metrics.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                df = df.sort_values('epoch')
                
                output_file = self.output_dir / f"threshold_metrics_{method}.csv"
                df.to_csv(output_file, index=False)
                print(f"  Saved {method} metrics: {output_file}")
        
        # Create summary plots
        self.create_threshold_evolution_plots()
        
        # Save complete results
        results_file = self.output_dir / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'results': self.results,
                'threshold_metrics': self.threshold_metrics
            }, f, indent=2)
        print(f"  Saved complete results: {results_file}")
    
    def create_threshold_evolution_plots(self):
        """Create comprehensive threshold evolution plots."""
        
        print(f"📊 Creating threshold evolution plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        methods = ['entropy', 'reconstruction', 'margin']
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Overall accuracy evolution
        ax1 = fig.add_subplot(gs[0, :])
        for i, method in enumerate(methods):
            if method in self.threshold_metrics and self.threshold_metrics[method]:
                df = pd.DataFrame(self.threshold_metrics[method])
                ax1.plot(df['epoch'], df['overall_accuracy'], 
                        marker='o', linewidth=2, label=f'{method.title()}', color=colors[i])
        
        ax1.set_xlabel('Training Epoch')
        ax1.set_ylabel('Overall Accuracy')
        ax1.set_title('Overall Accuracy Evolution During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2-4: Threshold evolution for each method
        for i, method in enumerate(methods):
            if method not in self.threshold_metrics or not self.threshold_metrics[method]:
                continue
            
            ax = fig.add_subplot(gs[1, i])
            df = pd.DataFrame(self.threshold_metrics[method])
            
            # Plot different accuracy thresholds
            threshold_cols = [col for col in df.columns if col.startswith('threshold_acc_')]
            for col in threshold_cols:
                threshold_pct = col.split('_')[-1]
                ax.plot(df['epoch'], df[col], marker='o', 
                       label=f'{threshold_pct}% Acc Threshold', alpha=0.7)
            
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Difficulty Threshold')
            ax.set_title(f'{method.title()}: Accuracy Threshold Evolution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Plot 5-7: Easy vs Hard accuracy gap
        for i, method in enumerate(methods):
            if method not in self.threshold_metrics or not self.threshold_metrics[method]:
                continue
            
            ax = fig.add_subplot(gs[2, i])
            df = pd.DataFrame(self.threshold_metrics[method])
            
            ax.plot(df['epoch'], df['easy_accuracy'], marker='o', label='Easy Samples', color='green')
            ax.plot(df['epoch'], df['hard_accuracy'], marker='s', label='Hard Samples', color='red')
            ax.fill_between(df['epoch'], df['easy_accuracy'], df['hard_accuracy'], 
                           alpha=0.2, color='gray', label='Accuracy Gap')
            
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{method.title()}: Easy vs Hard Sample Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 8-10: Difficulty correlation evolution
        for i, method in enumerate(methods):
            if method not in self.threshold_metrics or not self.threshold_metrics[method]:
                continue
            
            ax = fig.add_subplot(gs[3, i])
            df = pd.DataFrame(self.threshold_metrics[method])
            
            ax.plot(df['epoch'], df['difficulty_correlation'], marker='o', 
                   linewidth=2, color=colors[i])
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title(f'{method.title()}: Difficulty-Accuracy Correlation')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Difficulty Analysis: Threshold Evolution During Training', 
                     fontsize=16, fontweight='bold')
        
        # Save the plot
        plot_file = self.output_dir / "threshold_evolution_comprehensive.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved comprehensive plot: {plot_file}")
        
        # Create individual method plots
        self.create_individual_method_plots()
    
    def create_individual_method_plots(self):
        """Create detailed plots for each method individually."""
        
        for method in ['entropy', 'reconstruction', 'margin']:
            if method not in self.threshold_metrics or not self.threshold_metrics[method]:
                continue
            
            df = pd.DataFrame(self.threshold_metrics[method])
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Detailed Analysis: {method.title()} Method', fontsize=16, fontweight='bold')
            
            # Plot 1: Threshold evolution
            ax = axes[0, 0]
            threshold_cols = [col for col in df.columns if col.startswith('threshold_acc_')]
            for col in threshold_cols:
                threshold_pct = col.split('_')[-1]
                ax.plot(df['epoch'], df[col], marker='o', label=f'{threshold_pct}%')
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Difficulty Threshold')
            ax.set_title('Accuracy Threshold Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Accuracy gap evolution
            ax = axes[0, 1]
            ax.plot(df['epoch'], df['easy_hard_gap'], marker='o', linewidth=2, color='purple')
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Easy - Hard Accuracy Gap')
            ax.set_title('Easy vs Hard Sample Gap')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Overall metrics
            ax = axes[1, 0]
            ax2 = ax.twinx()
            
            line1 = ax.plot(df['epoch'], df['overall_accuracy'], marker='o', color='blue', label='Overall Accuracy')
            line2 = ax2.plot(df['epoch'], df['difficulty_correlation'], marker='s', color='red', label='Correlation')
            
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Overall Accuracy', color='blue')
            ax2.set_ylabel('Difficulty Correlation', color='red')
            ax.set_title('Accuracy vs Correlation Evolution')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right')
            
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Summary statistics
            ax = axes[1, 1]
            ax.plot(df['epoch'], df['difficulty_range'], marker='o', label='Difficulty Range')
            ax.plot(df['epoch'], df['accuracy_drop'], marker='s', label='Accuracy Drop')
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Value')
            ax.set_title('Difficulty and Accuracy Range Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_file = self.output_dir / f"detailed_analysis_{method}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved detailed plot: {plot_file}")


def main():
    """Main function to run the batch difficulty analysis."""
    
    print("🧪 Clean Batch Difficulty Analysis")
    print("=" * 50)
    
    # Configuration
    base_dir = "/home/brothen/solo-learn/trained_models"
    output_dir = "clean_batch_difficulty_analysis"
    max_epochs = 20
    
    # Initialize analyzer
    analyzer = CleanBatchDifficultyAnalyzer(base_dir, output_dir)
    
    # Find checkpoint pairs
    checkpoint_pairs = analyzer.find_checkpoint_pairs(max_epochs)
    
    if not checkpoint_pairs:
        print("❌ No valid checkpoint pairs found!")
        print("Please ensure you have both backbone and linear checkpoints.")
        return 1
    
    # Run batch analysis
    difficulty_methods = ['entropy', 'reconstruction', 'margin']
    analyzer.run_batch_analysis(checkpoint_pairs, difficulty_methods)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"📊 Plots available:")
    print(f"  - threshold_evolution_comprehensive.png")
    print(f"  - detailed_analysis_<method>.png")
    print(f"📈 Metrics saved to: threshold_metrics_<method>.csv")
    
    return 0


if __name__ == "__main__":
    exit(main()) 