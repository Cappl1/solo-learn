#!/usr/bin/env python3
"""
Test script to compare the two MVImageNet dataset implementations
"""

import sys
import os
sys.path.append('.')

import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import h5py

# Import both implementations
from solo.data.custom.temporal_mvimagnet import TemporalMVImageNet as TemporalMVImageNet_v1
from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet as TemporalMVImageNet_v2
from solo.data.custom.temporal_mvimagnet2 import create_temporal_mvimagenet_splits

def simple_dual_transform(img1, img2):
    """Simple transform for testing that handles dual images"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img1), transform(img2)

def explore_h5_structure(h5_path, max_show=3):
    """Explore H5 file structure to understand data organization"""
    print(f"\n=== Exploring H5 structure: {h5_path} ===")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"Top-level keys: {list(f.keys())[:max_show]}... (total: {len(f.keys())})")
            
            # Explore first few keys
            for i, key in enumerate(list(f.keys())[:max_show]):
                print(f"\nKey '{key}':")
                if isinstance(f[key], h5py.Group):
                    print(f"  Type: Group")
                    print(f"  Sub-keys: {list(f[key].keys())[:max_show]}... (total: {len(f[key].keys())})")
                    
                    # Go one level deeper
                    if len(f[key].keys()) > 0:
                        first_subkey = list(f[key].keys())[0]
                        print(f"  Exploring '{key}/{first_subkey}':")
                        subgroup = f[key][first_subkey]
                        if isinstance(subgroup, h5py.Group):
                            print(f"    Type: Group")
                            print(f"    Sub-keys: {list(subgroup.keys())[:max_show]}...")
                            
                            # Go one more level deeper
                            if len(subgroup.keys()) > 0:
                                first_subsubkey = list(subgroup.keys())[0]
                                print(f"    Exploring '{key}/{first_subkey}/{first_subsubkey}':")
                                data = subgroup[first_subsubkey]
                                if isinstance(data, h5py.Dataset):
                                    print(f"      Type: Dataset")
                                    print(f"      Shape: {data.shape}")
                                    print(f"      Dtype: {data.dtype}")
                                    print(f"      Size: {data.size}")
                        elif isinstance(subgroup, h5py.Dataset):
                            print(f"    Type: Dataset")
                            print(f"    Shape: {subgroup.shape}")
                            print(f"    Dtype: {subgroup.dtype}")
                elif isinstance(f[key], h5py.Dataset):
                    print(f"  Type: Dataset")
                    print(f"  Shape: {f[key].shape}")
                    print(f"  Dtype: {f[key].dtype}")
    except Exception as e:
        print(f"Error exploring H5 file: {e}")

def test_implementation_v1(h5_path, metadata_path=None):
    """Test the first implementation (temporal_mvimagnet.py)"""
    print(f"\n=== Testing Implementation V1 (original) ===")
    
    try:
        # Test without metadata first
        print("Testing without metadata...")
        dataset = TemporalMVImageNet_v1(
            h5_path=h5_path,
            metadata_path=None,  # No metadata
            time_window=10,
            categories=None,  # Use all categories
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test a few samples
        print("Testing sample loading:")
        for i in range(min(3, len(dataset))):
            start_time = time.time()
            try:
                (img1, img2), metadata = dataset[i]
                load_time = time.time() - start_time
                print(f"  Sample {i}: images {img1.size} & {img2.size}, metadata keys: {list(metadata.keys())}, time={load_time:.3f}s")
            except Exception as e:
                print(f"  Sample {i}: ERROR - {e}")
        
        # Test with metadata if provided
        if metadata_path:
            print(f"\nTesting with metadata from {metadata_path}...")
            dataset_with_meta = TemporalMVImageNet_v1(
                h5_path=h5_path,
                metadata_path=metadata_path,
                time_window=10,
                categories=None,
            )
            
            print(f"Dataset with metadata size: {len(dataset_with_meta)}")
            
            # Test one sample
            try:
                (img1, img2), metadata = dataset_with_meta[0]
                print(f"  Sample with metadata: images {img1.size} & {img2.size}, metadata keys: {list(metadata.keys())}")
            except Exception as e:
                print(f"  Sample with metadata: ERROR - {e}")
        
        return dataset
        
    except Exception as e:
        print(f"Implementation V1 failed: {e}")
        return None

def test_implementation_v2(h5_path, metadata_path):
    """Test the second implementation (temporal_mvimagnet2.py)"""
    print(f"\n=== Testing Implementation V2 (metadata-based) ===")
    
    if not metadata_path:
        print("V2 requires metadata_path, skipping...")
        return None
    
    try:
        dataset = TemporalMVImageNet_v2(
            h5_path=h5_path,
            metadata_path=metadata_path,
            time_window=10,
            split='train',  # Specify split
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test a few samples
        print("Testing sample loading:")
        for i in range(min(3, len(dataset))):
            start_time = time.time()
            try:
                (img1, img2), label = dataset[i]
                load_time = time.time() - start_time
                print(f"  Sample {i}: images {img1.size} & {img2.size}, label: {label}, time={load_time:.3f}s")
            except Exception as e:
                print(f"  Sample {i}: ERROR - {e}")
        
        return dataset
        
    except Exception as e:
        print(f"Implementation V2 failed: {e}")
        return None

def compare_performance(dataset1, dataset2, num_samples=10):
    """Compare loading performance between implementations"""
    print(f"\n=== Performance Comparison ({num_samples} samples) ===")
    
    if dataset1 is None or dataset2 is None:
        print("Cannot compare - one or both datasets failed to load")
        return
    
    # Test V1 performance
    v1_times = []
    print("Testing V1 performance...")
    for i in range(min(num_samples, len(dataset1))):
        start_time = time.time()
        try:
            _ = dataset1[i]
            v1_times.append(time.time() - start_time)
        except Exception as e:
            print(f"  V1 sample {i} failed: {e}")
    
    # Test V2 performance
    v2_times = []
    print("Testing V2 performance...")
    for i in range(min(num_samples, len(dataset2))):
        start_time = time.time()
        try:
            _ = dataset2[i]
            v2_times.append(time.time() - start_time)
        except Exception as e:
            print(f"  V2 sample {i} failed: {e}")
    
    # Report results
    if v1_times and v2_times:
        print(f"V1 avg time: {np.mean(v1_times):.4f}s (±{np.std(v1_times):.4f})")
        print(f"V2 avg time: {np.mean(v2_times):.4f}s (±{np.std(v2_times):.4f})")
        print(f"Ratio (V2/V1): {np.mean(v2_times)/np.mean(v1_times):.2f}x")

def test_with_transforms(dataset, name):
    """Test dataset with transforms applied"""
    print(f"\n=== Testing {name} with transforms ===")
    
    if dataset is None:
        print(f"{name} is None, skipping transform test")
        return
    
    # Create a version with transforms (need to create new instance)
    # This is tricky because we need to recreate the dataset with transforms
    print("Note: Transform testing would require recreating dataset instances")
    print("Skipping detailed transform testing for now")

def test_dataloader_compatibility(dataset, name):
    """Test if dataset works with PyTorch DataLoader"""
    print(f"\n=== Testing {name} DataLoader compatibility ===")
    
    if dataset is None:
        print(f"{name} is None, skipping DataLoader test")
        return
    
    try:
        # Create DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=0  # Use 0 to avoid multiprocessing issues in testing
        )
        
        # Try to get one batch
        batch = next(iter(dataloader))
        
        # Handle different return formats
        if isinstance(batch, tuple) and len(batch) == 2:
            data, labels = batch
            if isinstance(data, tuple) and len(data) == 2:
                # V1 format: ((img1_batch, img2_batch), metadata_or_labels)
                img1_batch, img2_batch = data
                print(f"  {name} DataLoader successful:")
                print(f"    Image1 batch shape: {img1_batch.shape if hasattr(img1_batch, 'shape') else type(img1_batch)}")
                print(f"    Image2 batch shape: {img2_batch.shape if hasattr(img2_batch, 'shape') else type(img2_batch)}")
                print(f"    Labels/metadata type: {type(labels)}")
            else:
                # V2 format might be different
                print(f"  {name} DataLoader batch format: {type(data)}, {type(labels)}")
        else:
            print(f"  {name} DataLoader unexpected batch format: {type(batch)}")
            
    except Exception as e:
        print(f"  {name} DataLoader failed: {e}")

def main():
    """Main test function"""
    print("MVImageNet Dataset Implementation Comparison")
    print("=" * 50)
    
    # Define test paths - update these according to your system
    h5_path = "/home/data/MVImageNet/data_all.h5"  # Update this path
    metadata_paths = [
        "/home/data/MVImageNet/dataset_val_all3.parquet",  # V2 metadata
        "/home/data/MVImageNet/dataset_test_all3.parquet",  # Alternative metadata for V1
    ]
    
    # Check if files exist
    print("Checking file availability:")
    print(f"  H5 file: {os.path.exists(h5_path)} - {h5_path}")
    
    available_metadata = []
    for meta_path in metadata_paths:
        exists = os.path.exists(meta_path)
        print(f"  Metadata: {exists} - {meta_path}")
        if exists:
            available_metadata.append(meta_path)
    
    if not os.path.exists(h5_path):
        print("\nERROR: H5 file not found. Please update the h5_path in the script.")
        return
    
    # Explore H5 structure
    explore_h5_structure(h5_path)
    
    # Test both implementations
    metadata_path = available_metadata[0] if available_metadata else None
    
    dataset_v1 = test_implementation_v1(h5_path, metadata_path)
    dataset_v2 = test_implementation_v2(h5_path, metadata_path)
    
    # Compare performance
    compare_performance(dataset_v1, dataset_v2)
    
    # Test transforms
    test_with_transforms(dataset_v1, "V1")
    test_with_transforms(dataset_v2, "V2")
    
    # Test DataLoader compatibility
    test_dataloader_compatibility(dataset_v1, "V1")
    test_dataloader_compatibility(dataset_v2, "V2")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"V1 (original): {'✓' if dataset_v1 is not None else '✗'} - Direct H5 structure reading")
    print(f"V2 (metadata): {'✓' if dataset_v2 is not None else '✗'} - Metadata-based indexing")
    
    if dataset_v1 is not None and dataset_v2 is not None:
        print(f"V1 dataset size: {len(dataset_v1)}")
        print(f"V2 dataset size: {len(dataset_v2)}")
    
    # Test the create_temporal_mvimagenet_splits function
    print(f"\n=== Testing create_temporal_mvimagenet_splits ===")
    try:
        if len(available_metadata) >= 3:  # Need train, val, test
            splits = create_temporal_mvimagenet_splits(
                h5_path=h5_path,
                train_csv=available_metadata[0],
                val_parquet=available_metadata[1] if len(available_metadata) > 1 else available_metadata[0],
                test_parquet=available_metadata[2] if len(available_metadata) > 2 else available_metadata[0],
                time_window=10
            )
            print(f"Splits created successfully:")
            for split_name, split_dataset in splits.items():
                print(f"  {split_name}: {len(split_dataset)} samples")
        else:
            print("Not enough metadata files to test splits creation")
    except Exception as e:
        print(f"Splits creation failed: {e}")

if __name__ == "__main__":
    main() 