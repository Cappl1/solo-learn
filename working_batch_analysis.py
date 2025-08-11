#!/usr/bin/env python3
"""
Working Batch Analysis - No subprocess BS, just direct imports
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# Add the project to Python path
sys.path.insert(0, '/home/brothen/solo-learn')

# Now import the working evaluator directly
from evaluate_difficulty import DifficultyEvaluator

def find_checkpoint_pairs(base_dir: str, max_epochs: int = 20) -> List[Tuple[int, str, str]]:
    """Find available checkpoint pairs."""
    
    base_path = Path(base_dir)
    backbone_dir = base_path / "selective_curriculum_mocov3" / "t60"
    linear_base = base_path / "linear" / "selective_curriculum_mocov3_t60"
    
    checkpoint_pairs = []
    
    # Find backbone checkpoints with stp=0 
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
                       difficulty_method: str, output_dir: Path) -> Dict:
    """Run analysis for a single checkpoint pair and method using direct import."""
    
    config_path = "/home/brothen/solo-learn/scripts/linear_probe/core50/difficulty_cur.yaml"
    
    # Create output directory for this epoch and method
    epoch_output_dir = output_dir / f"epoch_{epoch_num:02d}_{difficulty_method}"
    epoch_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Change to output directory
    original_cwd = os.getcwd()
    os.chdir(epoch_output_dir)
    
    try:
        print(f"    Running {difficulty_method} analysis for epoch {epoch_num}...")
        
        # Use the DifficultyEvaluator directly - no subprocess!
        evaluator = DifficultyEvaluator(
            model_path=backbone_path,
            linear_path=linear_path,
            cfg_path=config_path,
            device='cuda',
            difficulty_method=difficulty_method
        )
        
        # Load config and finalize model loading
        from omegaconf import OmegaConf
        evaluator.cfg = OmegaConf.load(config_path)
        evaluator.finalize_model_loading()
        
        # Create dataloader - copy from evaluate_difficulty.py main function
        eval_cfg = OmegaConf.create(evaluator.cfg.data)
        eval_cfg.dataset = "temporal_core50"
        
        if 'dataset_kwargs' not in eval_cfg:
            eval_cfg.dataset_kwargs = {}
        
        eval_cfg.dataset_kwargs.time_window = 15
        eval_cfg.dataset_kwargs.backgrounds = eval_cfg.get('val_backgrounds', ["s3", "s7", "s10"])
        
        # Use instance labels (50 classes) as the classifier expects
        classifier_num_classes = evaluator.classifier.out_features
        use_categories = (classifier_num_classes == 10)
        eval_cfg.dataset_kwargs.use_categories = use_categories
        
        # Create dataset and dataloader
        from solo.data.custom.temporal_core50 import TemporalCore50
        from torch.utils.data import DataLoader
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        eval_dataset = TemporalCore50(
            h5_path=eval_cfg.get('val_path', eval_cfg.get('train_path')),
            transform=lambda img1, img2: (transform(img1), transform(img2)),
            time_window=eval_cfg.dataset_kwargs.time_window,
            backgrounds=eval_cfg.dataset_kwargs.backgrounds,
            use_categories=eval_cfg.dataset_kwargs.use_categories
        )
        
        val_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        
        # Run evaluation
        evaluator.evaluate_dataset(val_loader)
        
        # Analyze results
        results = evaluator.analyze_results()
        
        print(f"    ✅ {difficulty_method} analysis completed (acc: {results['overall_accuracy']:.3f})")
        return results
        
    except Exception as e:
        print(f"    ❌ {difficulty_method} analysis failed: {e}")
        return None
        
    finally:
        os.chdir(original_cwd)

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
    
    # Plot 3: Accuracy spread evolution
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        if method in results and results[method]:
            epochs = sorted(results[method].keys())
            spreads = []
            for epoch in epochs:
                bins = results[method][epoch]['bins']
                accuracies = [acc for acc in bins['accuracies'] if acc > 0]
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
    print("🧪 Working Batch Difficulty Analysis (No Subprocess BS)")
    print("=" * 60)
    
    # Configuration
    base_dir = "/home/brothen/solo-learn/trained_models"
    output_dir = Path("working_batch_difficulty_analysis")
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
    
    results = {}
    success_count = 0
    total_runs = len(checkpoint_pairs) * len(difficulty_methods)
    
    for epoch_num, backbone_path, linear_path in tqdm(checkpoint_pairs, desc="Processing epochs"):
        print(f"\n📈 Processing Epoch {epoch_num}")
        print(f"  Backbone: {Path(backbone_path).name}")
        print(f"  Linear: {Path(linear_path).name}")
        
        for method in difficulty_methods:
            analysis_results = run_single_analysis(backbone_path, linear_path, epoch_num, 
                                        method, output_dir)
            if analysis_results:
                if method not in results:
                    results[method] = {}
                results[method][epoch_num] = analysis_results
                success_count += 1
    
    print(f"\n✅ Batch analysis complete!")
    print(f"📊 Success rate: {success_count}/{total_runs} ({success_count/total_runs*100:.1f}%)")
    
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