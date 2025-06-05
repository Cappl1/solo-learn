#!/usr/bin/env python3
"""
Investigate MVImageNet H5 file structure for dataset implementation
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import io
from PIL import Image

def investigate_h5_structure():
    """Investigate the structure of H5 files to understand how to build dataset classes"""
    
    data_dir = Path('/home/data/MVImageNet')
    
    # First, let's check data_all.h5 if it exists
    main_h5 = data_dir / 'data_all.h5'
    if main_h5.exists():
        print("=== Investigating data_all.h5 ===")
        try:
            with h5py.File(main_h5, 'r') as f:
                print(f"Top-level keys: {list(f.keys())}")
                
                # Check the structure of the first few keys
                for key in list(f.keys())[:3]:
                    print(f"\n--- Key: {key} ---")
                    item = f[key]
                    print(f"Type: {type(item)}")
                    
                    if isinstance(item, h5py.Dataset):
                        print(f"Dataset shape: {item.shape}")
                        print(f"Dataset dtype: {item.dtype}")
                        
                        # If it looks like image data, try to understand format
                        if len(item.shape) >= 2:
                            print(f"Possible image data format")
                            sample = item[0] if len(item) > 0 else None
                            if sample is not None:
                                print(f"Sample shape: {sample.shape if hasattr(sample, 'shape') else type(sample)}")
                                print(f"Sample dtype: {sample.dtype if hasattr(sample, 'dtype') else type(sample)}")
                        
                    elif isinstance(item, h5py.Group):
                        print(f"Group with keys: {list(item.keys())}")
                        
                        # Check subgroups/datasets
                        for subkey in list(item.keys())[:3]:
                            subitem = item[subkey]
                            print(f"  {subkey}: {type(subitem)}")
                            if isinstance(subitem, h5py.Dataset):
                                print(f"    Shape: {subitem.shape}, dtype: {subitem.dtype}")
                print()
                
        except Exception as e:
            print(f"Error reading data_all.h5: {e}")
            print()
    
    # Check a few individual H5 files
    h5_files = list(data_dir.glob('data*.h5'))[:3]
    for h5_file in h5_files:
        if h5_file.name != 'data_all.h5':
            print(f"=== Investigating {h5_file.name} ===")
            try:
                with h5py.File(h5_file, 'r') as f:
                    print(f"File size: {h5_file.stat().st_size / 1024**2:.2f} MB")
                    print(f"Top-level keys: {list(f.keys())}")
                    
                    for key in list(f.keys())[:2]:
                        print(f"\n--- Key: {key} ---")
                        item = f[key]
                        print(f"Type: {type(item)}")
                        
                        if isinstance(item, h5py.Dataset):
                            print(f"Dataset shape: {item.shape}")
                            print(f"Dataset dtype: {item.dtype}")
                            
                            # Try to extract a sample
                            if len(item.shape) >= 2 and len(item) > 0:
                                sample = item[0]
                                print(f"First sample shape: {sample.shape if hasattr(sample, 'shape') else type(sample)}")
                                
                                # If it looks like an image, try to understand the format
                                if hasattr(sample, 'shape') and len(sample.shape) >= 2:
                                    if sample.shape[-1] == 3 or len(sample.shape) == 3:
                                        print("Looks like RGB image data")
                                    elif len(sample.shape) == 2:
                                        print("Looks like grayscale image data")
                                
                        elif isinstance(item, h5py.Group):
                            print(f"Group with keys: {list(item.keys())}")
                    print()
                    
            except Exception as e:
                print(f"Error reading {h5_file}: {e}")
                print()

def investigate_parquet_h5_mapping():
    """Try to understand how parquet metadata maps to H5 data"""
    
    print("=== Investigating Parquet-H5 Mapping ===")
    
    data_dir = Path('/home/data/MVImageNet')
    
    # Load validation parquet (we know this one works)
    val_parquet = data_dir / 'dataset_val_all3.parquet'
    if val_parquet.exists():
        print("Loading validation parquet...")
        try:
            df = pd.read_parquet(val_parquet, engine='fastparquet')
            print(f"Validation set shape: {df.shape}")
            print(f"Sample paths:")
            for path in df['path'].head(10):
                print(f"  {path}")
            print()
            
            # Check if paths correspond to any H5 file structure
            print("Analyzing path patterns...")
            paths = df['path'].values
            
            # Extract directory patterns
            dirs = [Path(p).parent for p in paths[:100]]
            unique_dirs = pd.Series([str(d) for d in dirs]).value_counts()
            print("Top directory patterns:")
            print(unique_dirs.head(10))
            print()
            
            # Check if any H5 files contain keys that match these patterns
            h5_files = list(data_dir.glob('data*.h5'))[:3]
            for h5_file in h5_files:
                try:
                    with h5py.File(h5_file, 'r') as f:
                        h5_keys = list(f.keys())
                        print(f"\n{h5_file.name} has keys like: {h5_keys[:5]}")
                        
                        # Check if any keys match directory patterns
                        for key in h5_keys[:10]:
                            if any(str(d) in key or key in str(d) for d in list(unique_dirs.index)[:5]):
                                print(f"  Possible match: {key}")
                                
                except Exception as e:
                    print(f"Error checking {h5_file}: {e}")
            
        except Exception as e:
            print(f"Error reading validation parquet: {e}")

def test_image_loading():
    """Test if we can actually load images from H5 files"""
    
    print("=== Testing Image Loading ===")
    
    data_dir = Path('/home/data/MVImageNet')
    
    # Try to load images from different H5 files
    h5_files = list(data_dir.glob('data*.h5'))[:2]
    
    for h5_file in h5_files:
        print(f"\nTesting {h5_file.name}...")
        try:
            with h5py.File(h5_file, 'r') as f:
                keys = list(f.keys())
                
                for key in keys[:2]:  # Test first 2 keys
                    item = f[key]
                    if isinstance(item, h5py.Dataset) and len(item.shape) >= 2:
                        print(f"  Testing dataset {key}...")
                        
                        try:
                            # Try to load first sample
                            sample = item[0]
                            print(f"    Sample shape: {sample.shape}")
                            
                            # Try different ways to convert to image
                            if isinstance(sample, bytes):
                                print("    Data is bytes - trying PIL")
                                img = Image.open(io.BytesIO(sample))
                                print(f"    Successfully loaded image: {img.size}, {img.mode}")
                                
                            elif hasattr(sample, 'shape'):
                                print("    Data is array - trying PIL fromarray")
                                if len(sample.shape) == 3 and sample.shape[-1] == 3:
                                    img = Image.fromarray(sample.astype('uint8'))
                                    print(f"    Successfully loaded image: {img.size}, {img.mode}")
                                elif len(sample.shape) == 2:
                                    img = Image.fromarray(sample.astype('uint8'), mode='L')
                                    print(f"    Successfully loaded grayscale image: {img.size}, {img.mode}")
                                    
                        except Exception as e:
                            print(f"    Error loading sample: {e}")
                            
        except Exception as e:
            print(f"Error testing {h5_file}: {e}")

if __name__ == "__main__":
    investigate_h5_structure()
    investigate_parquet_h5_mapping()
    test_image_loading()
    
    print("\n=== Summary ===")
    print("Run this script to understand:")
    print("1. H5 file internal structure (keys, groups, datasets)")
    print("2. How parquet file paths map to H5 file organization")
    print("3. Image data format and loading methods")
    print("4. How to build the index mapping for dataset classes") 