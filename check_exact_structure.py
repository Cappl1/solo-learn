#!/usr/bin/env python3
import h5py
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    data_dir = Path('/home/data/MVImageNet')
    
    # Load parquet data
    val_parquet = data_dir / 'dataset_val_all3.parquet'
    df = pd.read_parquet(val_parquet, engine='fastparquet')
    
    # Get sample row
    sample = df.iloc[0]
    partition = sample['partition']
    category = str(sample['category'])
    obj = sample['object']
    frame = sample['frame']
    
    print(f"Testing access to:")
    print(f"  partition: {partition}")
    print(f"  category: {category}")
    print(f"  object: {obj}")
    print(f"  frame: {frame}")
    
    main_h5 = data_dir / 'data_all.h5'
    with h5py.File(main_h5, 'r') as f:
        # Navigate step by step
        print(f"\n1. Partition exists: {partition in f}")
        if partition in f:
            partition_group = f[partition]
            print(f"   Type: {type(partition_group)}")
            
            print(f"\n2. Category {category} exists: {category in partition_group}")
            if category in partition_group:
                category_group = partition_group[category]
                print(f"   Type: {type(category_group)}")
                
                print(f"\n3. Object {obj} exists: {obj in category_group}")
                if obj in category_group:
                    object_item = category_group[obj]
                    print(f"   Type: {type(object_item)}")
                    
                    if isinstance(object_item, h5py.Group):
                        print(f"   Object group keys: {list(object_item.keys())}")
                        
                        # Check different possible keys
                        for key in object_item.keys():
                            item = object_item[key]
                            print(f"   {key}: type={type(item)}")
                            if isinstance(item, h5py.Dataset):
                                print(f"     shape={item.shape}, dtype={item.dtype}")
                                
                    elif isinstance(object_item, h5py.Dataset):
                        print(f"   Direct dataset: shape={object_item.shape}, dtype={object_item.dtype}")
                        
                        # Try to access the frame
                        if len(object_item.shape) >= 1 and frame < object_item.shape[0]:
                            img = object_item[frame]
                            print(f"   Frame {frame}: shape={img.shape}, dtype={img.dtype}")
                        
    # Check a few more samples to understand the pattern
    print(f"\n=== Checking multiple samples ===")
    for i in range(min(3, len(df))):
        sample = df.iloc[i]
        partition = sample['partition']
        category = str(sample['category'])
        obj = sample['object']
        frame = sample['frame']
        
        with h5py.File(main_h5, 'r') as f:
            try:
                # Try different access patterns
                if partition in f and category in f[partition] and obj in f[partition][category]:
                    object_item = f[partition][category][obj]
                    
                    if isinstance(object_item, h5py.Dataset):
                        print(f"Sample {i}: DATASET access - shape={object_item.shape}")
                        if frame < object_item.shape[0]:
                            img = object_item[frame]
                            print(f"  Frame {frame} shape: {img.shape}")
                    elif isinstance(object_item, h5py.Group):
                        print(f"Sample {i}: GROUP access - keys={list(object_item.keys())}")
                        
            except Exception as e:
                print(f"Sample {i}: Error - {e}")

if __name__ == "__main__":
    main() 