#!/usr/bin/env python3
"""
Test script to verify checkpoint pairing and run batch difficulty analysis with available checkpoints.
Usage: python test_batch_difficulty_analysis.py
"""

import os
import sys
from pathlib import Path
import yaml
import torch

# Import the batch analyzer
from improved_batch_difficulty_analysis import BatchDifficultyAnalyzer, get_checkpoint_info

# Try to import data preparation
try:
    from solo.data.classification_dataloader import prepare_data
except ImportError:
    print("Warning: Could not import prepare_data")

def check_checkpoint_structure(base_dir: str):
    """Check what checkpoint files actually exist."""
    base_path = Path(base_dir)
    print("Checking checkpoint structure...")
    print(f"Base directory: {base_path}")
    
    # Look for backbone checkpoints
    possible_backbone_dirs = [
        base_path / "selective_curriculum_mocov3" / "t60",
        base_path / "curriculum_mocov3" / "t60", 
        base_path / "selective_jepa_curriculum_mocov3" / "t60",
        # Add more possibilities based on your actual structure
    ]
    
    backbone_dir = None
    for dir_path in possible_backbone_dirs:
        if dir_path.exists():
            backbone_dir = dir_path
            print(f"✅ Found backbone directory: {backbone_dir}")
            break
    
    if not backbone_dir:
        print("❌ No backbone directory found. Checking all subdirectories:")
        for item in base_path.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        return None, None
    
    # List backbone checkpoints
    backbone_files = list(backbone_dir.glob("*.ckpt"))
    print(f"Found {len(backbone_files)} backbone checkpoint files:")
    for f in sorted(backbone_files)[:5]:  # Show first 5
        print(f"  - {f.name}")
    
    # Look for linear checkpoints
    linear_base_dir = base_path / "linear" / "selective_curriculum_mocov3_t60"
    if linear_base_dir.exists():
        print(f"✅ Found linear base directory: {linear_base_dir}")
        linear_subdirs = [d for d in linear_base_dir.iterdir() if d.is_dir()]
        print(f"Found {len(linear_subdirs)} linear checkpoint directories:")
        for d in sorted(linear_subdirs)[:5]:  # Show first 5
            ckpt_files = list(d.glob("*.ckpt"))
            print(f"  - {d.name}: {len(ckpt_files)} .ckpt files")
    else:
        print(f"❌ Linear directory not found: {linear_base_dir}")
        # Check alternative locations
        linear_alt = base_path / "linear"
        if linear_alt.exists():
            print(f"Alternative linear directories found in {linear_alt}:")
            for item in linear_alt.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
    
    return backbone_dir, linear_base_dir

def get_available_checkpoint_pairs(base_dir: str, max_epochs: int = 20) -> list:
    """Get actually available checkpoint pairs, regardless of naming convention."""
    
    backbone_dir, linear_base_dir = check_checkpoint_structure(base_dir)
    if not backbone_dir:
        return []
    
    checkpoint_pairs = []
    
    # Look for backbone checkpoints with epoch info
    backbone_files = {}
    for ckpt_file in backbone_dir.glob("*.ckpt"):
        filename = ckpt_file.name
        # Try to extract epoch number from various patterns
        try:
            if "-ep=" in filename:
                epoch_part = filename.split("-ep=")[1].split("-stp=")[0]
                epoch_num = int(epoch_part)
                if epoch_num < max_epochs:
                    backbone_files[epoch_num] = str(ckpt_file)
        except (IndexError, ValueError):
            continue
    
    print(f"\nFound backbone checkpoints for epochs: {sorted(backbone_files.keys())}")
    
    # Look for corresponding linear checkpoints
    if linear_base_dir and linear_base_dir.exists():
        for epoch_num in sorted(backbone_files.keys()):
            # Try different naming patterns for linear checkpoints
            possible_linear_dirs = [
                linear_base_dir / f"selective_curriculum_mocov3_t60_ep{epoch_num:02d}",
                linear_base_dir / f"ep{epoch_num:02d}",
                linear_base_dir / f"epoch_{epoch_num:02d}",
            ]
            
            linear_checkpoint = None
            for linear_dir in possible_linear_dirs:
                if linear_dir.exists():
                    # Look for .ckpt files recursively (they're nested deeper)
                    ckpt_files = list(linear_dir.rglob("*.ckpt"))  # rglob for recursive search
                    if ckpt_files:
                        # Prefer files with "last" in the name, otherwise take first
                        last_ckpt = [f for f in ckpt_files if "last" in f.name]
                        if last_ckpt:
                            linear_checkpoint = str(last_ckpt[0])
                        else:
                            linear_checkpoint = str(ckpt_files[0])
                        break
            
            if linear_checkpoint:
                checkpoint_pairs.append((epoch_num, backbone_files[epoch_num], linear_checkpoint))
                print(f"✅ Epoch {epoch_num:2d}: Found pair")
                print(f"    Linear: {Path(linear_checkpoint).name}")
            else:
                print(f"❌ Epoch {epoch_num:2d}: Backbone found, but no linear checkpoint")
    
    return checkpoint_pairs

def create_test_config(base_config_path: str = None) -> str:
    """Create a test config file if the specified one doesn't exist."""
    
    # Try to find an existing config suitable for difficulty evaluation
    possible_configs = [
        "configs/difficulty_eval_selective_curriculum.yaml",  # Our new config
        "scripts/pretrain/core50/selective_curriculum_mocov3.yaml",  # Original training config
        "scripts/linear_probe/core50/mocov3_linear.yaml",
        "scripts/linear_probe/core50/mocov3_batch_base.yaml",
        "configs/linear_batch/linear_ep00.yaml",  # From batch training
    ]
    
    if base_config_path:
        possible_configs.insert(0, base_config_path)
    
    for config_path in possible_configs:
        if Path(config_path).exists():
            print(f"✅ Using existing config: {config_path}")
            return config_path
    
    # Create a minimal test config with ALL required parameters
    config_dir = Path("configs/difficulty_test")
    config_dir.mkdir(exist_ok=True, parents=True)
    config_path = config_dir / "test_config.yaml"
    
    # Use a complete config that matches your training setup and includes ALL required fields
    test_config = {
        'defaults': ['_self_'],
        'name': 'difficulty_analysis_test',
        'method': 'selective_curriculum_mocov3',  # Match your training method
        'backbone': {'name': 'resnet18'},
        
        # CRITICAL: Add method_kwargs that match your training
        'method_kwargs': {
            'proj_hidden_dim': 4096,
            'proj_output_dim': 256,
            'pred_hidden_dim': 4096,
            'temperature': 0.1,
            'curriculum_type': 'jepa',
            'curriculum_reverse': False,
            'curriculum_warmup_epochs': 10,
            'curriculum_weight': 1.0,
            'reconstruction_masking_ratio': 0.75,
            'num_candidates': 8,
            'curriculum_only_for_epochs': 100
        },
        
        # CRITICAL: Add momentum section (this was missing!)
        'momentum': {
            'base_tau': 0.996,
            'final_tau': 0.996
        },
        
        'data': {
            'dataset': 'core50_categories',
            'train_path': '/home/brothen/core50_arr.h5',
            'val_path': '/home/brothen/core50_arr.h5',
            'format': 'h5',
            'num_classes': 10,
            'num_workers': 4,
            # Add the missing num_large_crops field
            'num_large_crops': 2,
            'num_small_crops': 0,
            'train_backgrounds': ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11'],
            'val_backgrounds': ['s3', 's7', 's10'],
            'dataset_kwargs': {'use_categories': True}
        },
        'optimizer': {
            'name': 'lars',
            'batch_size': 64,
            'lr': 1.6,
            'weight_decay': 1e-6
        },
        # CRITICAL: Add complete scheduler section (this was missing!)
        'scheduler': {
            'name': 'warmup_cosine',
            'warmup_epochs': 0.01,
            'warmup_start_lr': 0.0,
            'eta_min': 0.0,
            'max_epochs': 100,
            'interval': 'step',
            'frequency': 1,
            'lr_decay_steps': None,
            'min_lr': 0.0
        },
        
        # Additional required parameters that solo-learn expects
        'no_validation': False,
        'accumulate_grad_batches': 1,
        'log_every_n_steps': 50,
        'max_epochs': 100,
        'check_val_every_n_epoch': 1,
        
        # Add augmentation parameters that might be expected
        'augmentations': [],
        'transform_kwargs': {},
        
        # Add loss parameters 
        'loss': {
            'name': 'mocov3',
            'temperature': 0.1
        },
        
        # Training parameters
        'devices': [0],
        'accelerator': 'gpu',
        'precision': 32,
        'sync_batchnorm': True,
        'strategy': 'auto',
        'enable_checkpointing': False,
        'logger': False,
        'callbacks': None
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    print(f"✅ Created test config: {config_path}")
    return str(config_path)

def main():
    # Configuration - adjust these paths to match your setup
    base_dir = "/home/brothen/solo-learn/trained_models"
    output_dir = "test_difficulty_analysis_t60"
    max_test_epochs = 5  # Test with first 5 epochs only
    
    print("="*60)
    print("TESTING BATCH DIFFICULTY ANALYSIS SETUP")
    print("="*60)
    
    # Step 1: Check checkpoint structure
    checkpoint_pairs = get_available_checkpoint_pairs(base_dir, max_test_epochs)
    
    if not checkpoint_pairs:
        print("\n❌ No valid checkpoint pairs found!")
        print("Please check your checkpoint directories and ensure linear classifiers have been trained.")
        return 1
    
    print(f"\n✅ Found {len(checkpoint_pairs)} valid checkpoint pairs for testing")
    
    # Step 2: Create/find config
    config_path = create_test_config()
    
    # Step 3: Add config path to checkpoint pairs
    checkpoint_info = [(epoch, backbone, linear, config_path) 
                      for epoch, backbone, linear in checkpoint_pairs]
    
    # Step 4: Test with a single checkpoint first
    print(f"\n" + "="*60)
    print("TESTING WITH SINGLE CHECKPOINT")
    print("="*60)
    
    test_epoch, test_backbone, test_linear, test_config = checkpoint_info[0]
    print(f"Testing epoch {test_epoch}:")
    print(f"  Backbone: {Path(test_backbone).name}")
    print(f"  Linear: {Path(test_linear).name}")
    print(f"  Config: {test_config}")
    
    # Initialize analyzer
    analyzer = BatchDifficultyAnalyzer(base_dir, output_dir)
    
    # Test direct import
    try:
        from flexible_main_difficulty_eval import DifficultyEvaluator
        print("✅ Direct import of DifficultyEvaluator successful")
        use_flexible = False
    except ImportError:
        try:
            from main_difficulty_eval import DifficultyEvaluator
            print("✅ Direct import of DifficultyEvaluator successful")
            use_flexible = False
        except ImportError as e:
            print(f"❌ Direct import failed: {e}")
            print("Will use subprocess method")
            use_flexible = None
    
    # Test single analysis
    print(f"\nRunning test analysis for epoch {test_epoch}...")
    try:
        if use_flexible:
            # Use the flexible evaluator
            evaluator = DifficultyEvaluator(
                model_path=test_backbone,
                linear_path=test_linear,
                cfg_path=test_config,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                difficulty_method='entropy'
            )
            
            # Prepare data loader using the evaluator's modified config
            _, val_loader = prepare_data(
                "core50_categories",
                train_data_path=None,
                val_data_path=evaluator.cfg.data.val_path,
                data_format=evaluator.cfg.data.format,
                batch_size=32,
                num_workers=2,
                auto_augment=False,
                train_backgrounds=getattr(evaluator.cfg.data, 'train_backgrounds', None),
                val_backgrounds=getattr(evaluator.cfg.data, 'val_backgrounds', ["s3", "s7", "s10"]),
                use_categories=True
            )
            
            # Run evaluation
            evaluator.evaluate_dataset(val_loader, max_batches=5)  # Just 5 batches for testing
            result = evaluator.analyze_results()
            
        elif use_flexible is False:
            result = analyzer.run_single_difficulty_analysis_direct(
                test_backbone, test_linear, test_config, test_epoch,
                difficulty_method='entropy', batch_size=32
            )
        else:
            result = analyzer.run_single_difficulty_analysis_subprocess(
                test_backbone, test_linear, test_config, test_epoch,
                difficulty_method='entropy', batch_size=32
            )
        
        if result:
            print("✅ Single analysis completed successfully!")
            print(f"Overall accuracy: {result.get('overall_accuracy', 'N/A'):.3f}")
            
            # Don't run full batch analysis yet - just confirm the single test worked
            print(f"\n✅ Test completed successfully!")
            print(f"You can now run the full batch analysis with available checkpoints.")
            print(f"To do this, modify the batch_difficulty_analysis.py to use DifficultyEvaluator")
            
        else:
            print("❌ Single analysis failed")
            return 1
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())