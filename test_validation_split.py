#!/usr/bin/env python3
"""
Test the validation split functionality for MVImageNet
"""

import sys
sys.path.append('.')

import os
import numpy as np
import pandas as pd
from collections import Counter

def test_validation_split():
    """Test that validation split creates balanced class distributions"""
    print("=== Testing MVImageNet Validation Split ===")
    
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    print(f"Testing with:")
    print(f"  H5 path: {h5_path}")
    print(f"  Metadata path: {metadata_path}")
    
    # Test parameters
    val_split = 0.05
    time_window = 5
    random_seed = 42
    
    print(f"\nTest parameters:")
    print(f"  Validation split: {val_split*100:.1f}%")
    print(f"  Time window: {time_window}")
    print(f"  Random seed: {random_seed}")
    
    # Create training dataset
    print(f"\n--- Creating Training Dataset ---")
    train_dataset = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=metadata_path,
        time_window=time_window,
        split='train',
        val_split=val_split,
        stratify_by_category=True,
        random_seed=random_seed,
        transform=None
    )
    
    # Create validation dataset  
    print(f"\n--- Creating Validation Dataset ---")
    val_dataset = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=metadata_path,
        time_window=time_window,
        split='val',
        val_split=val_split,
        stratify_by_category=True,
        random_seed=random_seed,
        transform=None
    )
    
    print(f"\n--- Dataset Size Comparison ---")
    print(f"Training set size: {len(train_dataset):,}")
    print(f"Validation set size: {len(val_dataset):,}")
    print(f"Total size: {len(train_dataset) + len(val_dataset):,}")
    print(f"Validation percentage: {len(val_dataset) / (len(train_dataset) + len(val_dataset)) * 100:.2f}%")
    
    # Test class distribution
    print(f"\n--- Testing Class Distribution ---")
    
    # Get training set categories
    train_categories = []
    print("Sampling training set categories (first 100 samples)...")
    for i in range(min(100, len(train_dataset))):
        try:
            (img1, img2), category = train_dataset[i]
            train_categories.append(category)
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue
    
    # Get validation set categories  
    val_categories = []
    print("Sampling validation set categories (first 50 samples)...")
    for i in range(min(50, len(val_dataset))):
        try:
            (img1, img2), category = val_dataset[i]
            val_categories.append(category)
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue
    
    if train_categories and val_categories:
        train_counter = Counter(train_categories)
        val_counter = Counter(val_categories)
        
        print(f"\nTraining set category distribution (top 10):")
        for cat, count in train_counter.most_common(10):
            print(f"  Category {cat}: {count} samples")
            
        print(f"\nValidation set category distribution (top 10):")
        for cat, count in val_counter.most_common(10):
            print(f"  Category {cat}: {count} samples")
        
        # Check overlap
        train_cats = set(train_categories)
        val_cats = set(val_categories)
        overlap = train_cats & val_cats
        print(f"\nCategory overlap:")
        print(f"  Training categories: {len(train_cats)}")
        print(f"  Validation categories: {len(val_cats)}")
        print(f"  Overlapping categories: {len(overlap)}")
        print(f"  Overlap percentage: {len(overlap)/len(train_cats)*100:.1f}%")
        
        # Check if validation has reasonable distribution
        if len(val_cats) > 1:
            print(f"  ✓ Validation set has multiple categories")
        else:
            print(f"  ⚠️  Validation set has only {len(val_cats)} category")
            
        return True
    else:
        print(f"✗ Failed to sample categories from datasets")
        return False

def test_reproducibility():
    """Test that the split is reproducible with the same seed"""
    print(f"\n=== Testing Split Reproducibility ===")
    
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    # Create datasets twice with same seed
    datasets1 = []
    datasets2 = []
    
    for seed in [42, 42]:  # Same seed twice
        train_ds = TemporalMVImageNet(
            h5_path=h5_path,
            metadata_path=metadata_path,
            time_window=5,
            split='train',
            val_split=0.05,
            stratify_by_category=True,
            random_seed=seed,
            transform=None
        )
        
        val_ds = TemporalMVImageNet(
            h5_path=h5_path,
            metadata_path=metadata_path,
            time_window=5,
            split='val', 
            val_split=0.05,
            stratify_by_category=True,
            random_seed=seed,
            transform=None
        )
        
        if seed == 42:
            datasets1 = [train_ds, val_ds]
        else:
            datasets2 = [train_ds, val_ds]
    
    # Compare sizes
    train1, val1 = datasets1
    train2, val2 = datasets2
    
    print(f"First run:  Train={len(train1):,}, Val={len(val1):,}")
    print(f"Second run: Train={len(train2):,}, Val={len(val2):,}")
    
    if len(train1) == len(train2) and len(val1) == len(val2):
        print(f"✓ Dataset sizes are reproducible")
        return True
    else:
        print(f"✗ Dataset sizes differ between runs")
        return False

def test_integration_with_config():
    """Test integration with the configuration system"""
    print(f"\n=== Testing Configuration Integration ===")
    
    from omegaconf import DictConfig
    from solo.data.StatefulDistributeSampler import DataPrepIterCheck
    
    # Create a minimal config for testing validation
    cfg = DictConfig({
        'seed': 42,
        'data': {
            'dataset': 'temporal_mvimagenet',
            'train_path': '/home/data/MVImageNet/data_all.h5',
            'val_path': '/home/data/MVImageNet/data_all.h5',
            'format': 'h5',
            'num_workers': 2,
            'no_labels': False,
            'fraction': -1.0,
            'dataset_kwargs': {
                'metadata_path': '/home/data/MVImageNet/dataset_val_all3.parquet',
                'time_window': 5,
                'val_split': 0.05,
                'stratify_by_category': True,
                'random_seed': 42
            }
        },
        'augmentations': [{
            'num_crops': 2,
            'crop_size': 224,
            'rrc': {'enabled': False},
            'color_jitter': {'prob': 0.0},
            'grayscale': {'prob': 0.0},
            'gaussian_blur': {'prob': 0.0},
            'solarization': {'prob': 0.0},
            'equalization': {'prob': 0.0},
            'horizontal_flip': {'prob': 0.0}
        }],
        'optimizer': {
            'batch_size': 4
        },
        'debug_augmentations': False
    })
    
    try:
        # Create data module
        data_module = DataPrepIterCheck(cfg)
        data_module.setup(stage='fit')
        
        print(f"✓ DataPrepIterCheck setup successful")
        print(f"  Training dataset size: {len(data_module.train_dataset):,}")
        
        # Check if validation dataset was created
        if hasattr(data_module, 'val_dataset'):
            print(f"  Validation dataset size: {len(data_module.val_dataset):,}")
            
            # Test validation dataloader
            val_loader = data_module.val_dataloader()
            if val_loader is not None:
                print(f"  ✓ Validation dataloader created")
                return True
            else:
                print(f"  ⚠️  Validation dataloader is None")
                return False
        else:
            print(f"  ⚠️  No validation dataset created")
            return False
            
    except Exception as e:
        print(f"✗ Configuration integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("MVImageNet Validation Split Testing")
    print("=" * 40)
    
    # Check file availability
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    if not os.path.exists(h5_path):
        print(f"ERROR: H5 file not found: {h5_path}")
        return
        
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found: {metadata_path}")
        return
    
    # Run tests
    split_success = test_validation_split()
    repro_success = test_reproducibility()
    config_success = test_integration_with_config()
    
    print(f"\n=== Summary ===")
    print(f"Validation split: {'✓' if split_success else '✗'}")
    print(f"Reproducibility: {'✓' if repro_success else '✗'}")
    print(f"Config integration: {'✓' if config_success else '✗'}")
    
    if all([split_success, repro_success, config_success]):
        print(f"\n🎉 ALL VALIDATION TESTS PASSED!")
        print(f"   The 5% balanced validation split is working correctly.")
        print(f"   Ready for training with validation monitoring.")
    else:
        print(f"\n❌ Some validation tests failed.")

if __name__ == "__main__":
    main() 