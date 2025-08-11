#!/usr/bin/env python3
"""
Fixed test script to verify checkpoint pairing and run batch difficulty analysis.
Usage: python fixed_test_batch_difficulty_analysis.py
"""

import os
import sys
from pathlib import Path
import yaml
import torch

# Import the configuration fixer
from fixed_config_utils import create_complete_config, extract_config_from_checkpoint, fix_config_for_evaluation

# Import the batch analyzer
try:
    from improved_batch_difficulty_analysis import BatchDifficultyAnalyzer, get_checkpoint_info
except ImportError:
    print("Warning: Could not import BatchDifficultyAnalyzer")

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

def create_config_from_checkpoint(backbone_checkpoint: str, output_config_path: str = None) -> str:
    """Create a config by extracting from checkpoint and fixing it."""
    
    if output_config_path is None:
        epoch_str = Path(backbone_checkpoint).stem.split('-ep=')[1].split('-')[0] if '-ep=' in Path(backbone_checkpoint).stem else "unknown"
        output_config_path = f"configs/difficulty_eval_ep{epoch_str}.yaml"
    
    print(f"Creating config from checkpoint: {backbone_checkpoint}")
    
    # First, try to extract config from checkpoint
    extracted_config = extract_config_from_checkpoint(backbone_checkpoint, output_config_path + ".extracted")
    
    if extracted_config and Path(extracted_config).exists():
        print("Using extracted config from checkpoint")
        # Fix the extracted config
        fixed_config = fix_config_for_evaluation(extracted_config, output_config_path)
        return fixed_config
    else:
        print("Could not extract config from checkpoint, creating complete config")
        # Create a complete config
        return create_complete_config(output_config_path)

def test_single_analysis(test_epoch: int, test_backbone: str, test_linear: str, test_config: str):
    """Test difficulty analysis with a single checkpoint."""
    
    print(f"\nRunning test analysis for epoch {test_epoch}...")
    print(f"Modified config for evaluation:")
    
    # Load and modify config for evaluation
    with open(test_config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Show key config details
    print(f"  Dataset: {cfg['data']['dataset']}")
    print(f"  Num classes: {cfg['data']['num_classes']}")
    print(f"  Use categories: {cfg['data']['dataset_kwargs'].get('use_categories', False)}")
    
    try:
        # Try to import the fixed evaluator
        from fixed_flexible_evaluator import FixedFlexibleDifficultyEvaluator
        print("✅ Direct import of FixedFlexibleDifficultyEvaluator successful")
        
        # Test initialization
        evaluator = FixedFlexibleDifficultyEvaluator(
            model_path=test_backbone,
            linear_path=test_linear,
            cfg_path=test_config,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            difficulty_method='entropy'
        )
        
        print("✅ FixedFlexibleDifficultyEvaluator initialized successfully")
        
        # Prepare a small test dataset
        _, val_loader = prepare_data(
            cfg['data']['dataset'],
            train_data_path=None,
            val_data_path=cfg['data']['val_path'],
            data_format=cfg['data']['format'],
            batch_size=32,  # Small batch for testing
            num_workers=2,  # Reduced workers for testing
            auto_augment=False,
            train_backgrounds=cfg['data'].get('train_backgrounds', None),
            val_backgrounds=cfg['data'].get('val_backgrounds', ["s3", "s7", "s10"]),
            use_categories=cfg['data']['dataset_kwargs'].get('use_categories', True)
        )
        
        print("✅ Data loader created successfully")
        
        # Run evaluation on just a few batches
        evaluator.evaluate_dataset(val_loader, max_batches=3)  # Just 3 batches for testing
        result = evaluator.analyze_results()
        
        if result:
            print("✅ Analysis completed successfully!")
            print(f"Overall accuracy: {result.get('overall_accuracy', 'N/A'):.3f}")
            return True
        else:
            print("❌ Analysis returned no results")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"FixedFlexibleDifficultyEvaluator not available, will need to use subprocess method")
        return False
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Configuration - adjust these paths to match your setup
    base_dir = "/home/brothen/solo-learn/trained_models"
    output_dir = "test_difficulty_analysis_t60_fixed"
    max_test_epochs = 5  # Test with first 5 epochs only
    
    print("="*60)
    print("TESTING BATCH DIFFICULTY ANALYSIS SETUP (FIXED)")
    print("="*60)
    
    # Step 1: Check checkpoint structure
    checkpoint_pairs = get_available_checkpoint_pairs(base_dir, max_test_epochs)
    
    if not checkpoint_pairs:
        print("\n❌ No valid checkpoint pairs found!")
        print("Please check your checkpoint directories and ensure linear classifiers have been trained.")
        return 1
    
    print(f"\n✅ Found {len(checkpoint_pairs)} valid checkpoint pairs for testing")
    
    # Step 2: Create config from the first checkpoint
    test_epoch, test_backbone, test_linear = checkpoint_pairs[0]
    
    print(f"\n" + "="*60)
    print("CREATING PROPER CONFIGURATION")
    print("="*60)
    
    # Create a proper config by extracting from checkpoint or creating complete one
    config_path = create_config_from_checkpoint(
        test_backbone, 
        f"configs/difficulty_eval_fixed_ep{test_epoch:02d}.yaml"
    )
    
    if not config_path or not Path(config_path).exists():
        print("❌ Failed to create config file")
        return 1
    
    print(f"✅ Using config: {config_path}")
    
    # Step 3: Test with a single checkpoint
    print(f"\n" + "="*60)
    print("TESTING WITH SINGLE CHECKPOINT")
    print("="*60)
    
    print(f"Testing epoch {test_epoch}:")
    print(f"  Backbone: {Path(test_backbone).name}")
    print(f"  Linear: {Path(test_linear).name}")
    print(f"  Config: {config_path}")
    
    # Test the analysis
    success = test_single_analysis(test_epoch, test_backbone, test_linear, config_path)
    
    if success:
        print(f"\n✅ Test completed successfully!")
        print(f"Configuration is working properly.")
        print(f"You can now run the full batch analysis.")
        
        # Optionally create configs for all epochs
        print(f"\nCreating configs for all available epochs...")
        all_configs = []
        for epoch_num, backbone_path, linear_path in checkpoint_pairs:
            epoch_config = create_config_from_checkpoint(
                backbone_path,
                f"configs/difficulty_eval_fixed_ep{epoch_num:02d}.yaml"
            )
            if epoch_config:
                all_configs.append((epoch_num, backbone_path, linear_path, epoch_config))
        
        print(f"✅ Created {len(all_configs)} config files for batch analysis")
        
        # Save the checkpoint info for batch processing
        checkpoint_info_file = Path("checkpoint_info_fixed.yaml")
        with open(checkpoint_info_file, 'w') as f:
            yaml.dump({
                'base_dir': base_dir,
                'output_dir': output_dir,
                'checkpoint_pairs': [
                    {
                        'epoch': epoch,
                        'backbone': backbone,
                        'linear': linear,
                        'config': config
                    }
                    for epoch, backbone, linear, config in all_configs
                ]
            }, f)
        
        print(f"✅ Checkpoint info saved to: {checkpoint_info_file}")
        
    else:
        print("❌ Test failed - please check the error messages above")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())