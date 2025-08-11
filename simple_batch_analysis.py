#!/usr/bin/env python3
"""
Simple Batch Difficulty Analysis

This script just runs the working evaluate_difficulty.py approach multiple times
for different checkpoint pairs and collects the results.
"""

import os
import sys
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import pandas as pd

def find_checkpoint_pairs(base_dir: str, max_epochs: int = 20) -> List[Tuple[int, str, str]]:
    """Find available checkpoint pairs."""
    
    base_path = Path(base_dir)
    backbone_dir = base_path / "selective_curriculum_mocov3" / "t60"
    linear_base = base_path / "linear" / "selective_curriculum_mocov3_t60"
    
    checkpoint_pairs = []
    
    # Find backbone checkpoints with stp=0 (epoch end checkpoints)
    for ckpt_file in backbone_dir.glob("*-ep=*-stp=0.ckpt"):
        filename = ckpt_file.name
        
        try:
            epoch_part = filename.split("-ep=")[1].split("-stp=")[0]
            epoch_num = int(epoch_part)
            
            if epoch_num <= max_epochs:
                # Look for corresponding linear checkpoint
                epoch_dir = linear_base / f"selective_curriculum_mocov3_t60_ep{epoch_num:02d}"
                
                if epoch_dir.exists():
                    linear_dir = epoch_dir / "linear"
                    if linear_dir.exists():
                        # Find the wandb subdirectory and checkpoint
                        for wandb_dir in linear_dir.iterdir():
                            if wandb_dir.is_dir():
                                ckpt_files = list(wandb_dir.glob("*.ckpt"))
                                if ckpt_files:
                                    # Prefer "last" checkpoint
                                    last_ckpts = [f for f in ckpt_files if "last" in f.name]
                                    if last_ckpts:
                                        linear_checkpoint = str(last_ckpts[0])
                                    else:
                                        linear_checkpoint = str(ckpt_files[0])
                                    
                                    checkpoint_pairs.append((epoch_num, str(ckpt_file), linear_checkpoint))
                                    break
                        
        except (IndexError, ValueError):
            continue
    
    # Sort by epoch number
    checkpoint_pairs.sort(key=lambda x: x[0])
    return checkpoint_pairs

def run_single_analysis(backbone_path: str, linear_path: str, epoch_num: int, 
                       difficulty_method: str, output_dir: Path) -> bool:
    """Run analysis for a single checkpoint pair and method."""
    
    # Create output directory for this epoch and method
    epoch_output_dir = output_dir / f"epoch_{epoch_num:02d}_{difficulty_method}"
    epoch_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Change to output directory
    original_cwd = os.getcwd()
    os.chdir(epoch_output_dir)
    
    try:
        print(f"    Running {difficulty_method} analysis for epoch {epoch_num}...")
        print(f"    Working directory: {epoch_output_dir}")
        
        # Use the current Python environment directly
        cmd = [
            sys.executable, "/home/brothen/solo-learn/evaluate_difficulty.py",
            "--model_path", backbone_path,
            "--linear_path", linear_path,
            "--cfg_path", "/home/brothen/solo-learn/scripts/linear_probe/core50/difficulty_cur.yaml",
            "--difficulty_method", difficulty_method,
            "--batch_size", "64",
            "--num_workers", "4",
            "--device", "cuda"
        ]
        
        print(f"    Running command: {' '.join(cmd[:2])} [model and linear paths] {cmd[4:]}")
        
        # Run using current environment
        env = os.environ.copy()
        env['PYTHONPATH'] = '/home/brothen/solo-learn'
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              env=env, cwd="/home/brothen/solo-learn", timeout=300)  # 5 min timeout
        
        print(f"    Return code: {result.returncode}")
        
        if result.returncode == 0:
            # Check if results file was created in the main directory and move it
            main_results_dir = Path("/home/brothen/solo-learn/difficulty_analysis")
            results_file = main_results_dir / f"analysis_results_{difficulty_method}.json"
            local_results_dir = Path("difficulty_analysis")
            local_results_file = local_results_dir / f"analysis_results_{difficulty_method}.json"
            
            print(f"    Looking for results file: {results_file}")
            print(f"    File exists: {results_file.exists()}")
            
            if results_file.exists():
                # Move the results to the current epoch directory
                local_results_dir.mkdir(exist_ok=True)
                import shutil
                shutil.move(str(results_file), str(local_results_file))
                print(f"    Moved results to: {local_results_file}")
                
                # Also move any plots if they exist
                main_plots_dir = main_results_dir / "plots"
                if main_plots_dir.exists():
                    local_plots_dir = local_results_dir / "plots"
                    if local_plots_dir.exists():
                        shutil.rmtree(str(local_plots_dir))
                    shutil.move(str(main_plots_dir), str(local_plots_dir))
                    print(f"    Moved plots to: {local_plots_dir}")
                
                # Clean up only if the directory is empty after moving files
                if main_results_dir.exists():
                    remaining_files = list(main_results_dir.glob("*"))
                    if not remaining_files:  # Only delete if directory is empty
                        shutil.rmtree(str(main_results_dir))
                
                print(f"    ✅ {difficulty_method} analysis completed")
                return True
            else:
                print(f"    ❌ {difficulty_method} analysis failed - no results file")
                print(f"    Expected file: {results_file}")
                
                # List the main directory contents to see what was created
                main_dir = Path("/home/brothen/solo-learn")
                print(f"    Contents of {main_dir}:")
                for item in main_dir.iterdir():
                    if item.name.startswith('difficulty') or item.name.endswith('.json') or item.name.endswith('.png'):
                        print(f"      {item.name}")
                
                if result.stdout:
                    print(f"    stdout (last 500 chars): {result.stdout[-500:]}")
                if result.stderr:
                    print(f"    stderr (last 500 chars): {result.stderr[-500:]}")
                return False
        else:
            print(f"    ❌ {difficulty_method} analysis failed - return code {result.returncode}")
            if result.stdout:
                print(f"    stdout: {result.stdout}")
            if result.stderr:
                print(f"    stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ❌ {difficulty_method} analysis timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"    ❌ {difficulty_method} analysis failed with exception: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def collect_results(output_dir: Path, checkpoint_pairs: List[Tuple[int, str, str]], 
                   difficulty_methods: List[str]) -> Dict:
    """Collect all results into a single structure."""
    
    results = {}
    
    for method in difficulty_methods:
        results[method] = {}
        
        for epoch_num, _, _ in checkpoint_pairs:
            epoch_output_dir = output_dir / f"epoch_{epoch_num:02d}_{method}"
            results_file = epoch_output_dir / "difficulty_analysis" / f"analysis_results_{method}.json"
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results[method][epoch_num] = json.load(f)
                except Exception as e:
                    print(f"Error loading results for epoch {epoch_num}, method {method}: {e}")
    
    return results

def create_summary_plots(results: Dict, output_dir: Path):
    """Create summary plots showing threshold evolution."""
    
    plt.style.use('default')
    
    # Extract data for plotting
    methods = ['entropy', 'reconstruction', 'margin', 'pixel_entropy']
    colors = ['blue', 'red', 'green', 'orange']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Difficulty Analysis: Evolution Across Training Epochs', fontsize=16)
    
    # Plot 1: Overall accuracy evolution
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        if method in results and results[method]:
            epochs = sorted(results[method].keys())
            accuracies = [results[method][epoch]['overall_accuracy'] for epoch in epochs]
            ax.plot(epochs, accuracies, marker='o', label=f'{method.title()}', 
                   color=colors[i], linewidth=2)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Overall Accuracy')
    ax.set_title('Overall Accuracy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Difficulty range evolution
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        if method in results and results[method]:
            epochs = sorted(results[method].keys())
            ranges = []
            for epoch in epochs:
                bins = results[method][epoch]['bins']
                diff_range = max(bins['centers']) - min(bins['centers'])
                ranges.append(diff_range)
            ax.plot(epochs, ranges, marker='o', label=f'{method.title()}', 
                   color=colors[i], linewidth=2)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Difficulty Range')
    ax.set_title('Difficulty Range Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy spread (max - min accuracy across difficulty bins)
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        if method in results and results[method]:
            epochs = sorted(results[method].keys())
            spreads = []
            for epoch in epochs:
                bins = results[method][epoch]['bins']
                accuracies = [acc for acc in bins['accuracies'] if acc > 0]  # Remove empty bins
                if accuracies:
                    spread = max(accuracies) - min(accuracies)
                    spreads.append(spread)
                else:
                    spreads.append(0)
            ax.plot(epochs, spreads, marker='o', label=f'{method.title()}', 
                   color=colors[i], linewidth=2)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Accuracy Spread (Easy - Hard)')
    ax.set_title('Easy vs Hard Accuracy Gap Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Correlation evolution
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        if method in results and results[method]:
            epochs = sorted(results[method].keys())
            correlations = []
            for epoch in epochs:
                if 'correlations' in results[method][epoch]:
                    corr = results[method][epoch]['correlations']['pearson']['r']
                    correlations.append(corr)
                else:
                    correlations.append(0)
            ax.plot(epochs, correlations, marker='o', label=f'{method.title()}', 
                   color=colors[i], linewidth=2)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Difficulty-Accuracy Correlation Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_evolution_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary plot saved: {output_dir / 'threshold_evolution_summary.png'}")

def main():
    print("🧪 Simple Batch Difficulty Analysis")
    print("=" * 50)
    
    # Configuration
    base_dir = "/home/brothen/solo-learn/trained_models"
    output_dir = Path("simple_batch_difficulty_analysis")
    output_dir.mkdir(exist_ok=True)
    max_epochs = 20
    difficulty_methods = ['entropy', 'reconstruction', 'margin', 'pixel_entropy']
    
    # Find checkpoint pairs
    checkpoint_pairs = find_checkpoint_pairs(base_dir, max_epochs)
    
    if not checkpoint_pairs:
        print("❌ No checkpoint pairs found!")
        return 1
    
    print(f"Found {len(checkpoint_pairs)} checkpoint pairs")
    
    # Run analysis for each pair and method
    print(f"\n🚀 Running batch analysis...")
    
    total_runs = len(checkpoint_pairs) * len(difficulty_methods)
    success_count = 0
    
    for epoch_num, backbone_path, linear_path in tqdm(checkpoint_pairs, desc="Processing epochs"):
        print(f"\n📈 Processing Epoch {epoch_num}")
        print(f"  Backbone: {Path(backbone_path).name}")
        print(f"  Linear: {Path(linear_path).name}")
        
        for method in difficulty_methods:
            success = run_single_analysis(backbone_path, linear_path, epoch_num, 
                                        method, output_dir)
            if success:
                success_count += 1
    
    print(f"\n✅ Batch analysis complete!")
    print(f"📊 Success rate: {success_count}/{total_runs} ({success_count/total_runs*100:.1f}%)")
    
    # Collect and analyze results
    print(f"\n📈 Collecting results...")
    results = collect_results(output_dir, checkpoint_pairs, difficulty_methods)
    
    # Save aggregated results
    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary plots
    print(f"\n📊 Creating summary plots...")
    create_summary_plots(results, output_dir)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"📊 Summary plot: threshold_evolution_summary.png")
    print(f"📈 Individual results: epoch_XX_<method>/difficulty_analysis/")
    
    return 0

if __name__ == "__main__":
    exit(main()) 