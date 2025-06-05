#!/usr/bin/env python3
import h5py
import pandas as pd
from pathlib import Path
import urllib.parse

def main():
    data_dir = Path('/home/data/MVImageNet')
    
    # Load parquet data to understand the mapping
    val_parquet = data_dir / 'dataset_val_all3.parquet'
    df = pd.read_parquet(val_parquet, engine='fastparquet')
    
    print("=== Understanding Data Structure ===")
    print(f"Validation samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Key insight: partition column contains URL-encoded UUIDs that match H5 keys
    print(f"\nPartition examples:")
    partitions = df['partition'].unique()
    for i, partition in enumerate(partitions[:3]):
        decoded = urllib.parse.unquote(partition)
        print(f"  {partition} -> {decoded}")
    
    # Object column contains object IDs that are referenced in paths
    print(f"\nObject examples:")
    objects = df['object'].unique()[:5]
    for obj in objects:
        print(f"  {obj}")
    
    # Path structure analysis
    print(f"\nPath structure analysis:")
    paths = df['path'].head(10)
    for path in paths:
        parts = Path(path).parts
        print(f"  {path} -> category: {parts[0]}, object: {parts[1]}, file: {parts[2]}")
    
    # Now check H5 structure with this understanding
    main_h5 = data_dir / 'data_all.h5'
    with h5py.File(main_h5, 'r') as f:
        # Test one partition
        test_partition = partitions[0]
        if test_partition in f:
            print(f"\n=== H5 Structure for partition {test_partition} ===")
            partition_group = f[test_partition]
            category_keys = list(partition_group.keys())
            print(f"Category keys: {category_keys[:10]}")
            
            # Check one category
            test_category = category_keys[0]
            category_group = partition_group[test_category]
            
            if isinstance(category_group, h5py.Group):
                object_keys = list(category_group.keys())
                print(f"Objects in category {test_category}: {len(object_keys)} objects")
                print(f"Sample object keys: {object_keys[:5]}")
                
                # Check one object
                test_object = object_keys[0]
                object_group = category_group[test_object]
                
                if isinstance(object_group, h5py.Group):
                    print(f"Object {test_object} structure: {list(object_group.keys())}")
                    
                    # Check if 'images' exists
                    if 'images' in object_group:
                        images = object_group['images']
                        print(f"Images dataset: shape={images.shape}, dtype={images.dtype}")
                        
                        # Try to load one image
                        if len(images) > 0:
                            img = images[0]
                            print(f"First image shape: {img.shape}")
    
    # Create mapping summary
    print(f"\n=== Data Organization Summary ===")
    print("Structure: H5[partition][category][object]['images'][frame_index]")
    print("- partition: URL-encoded UUID (matches parquet 'partition' column)")
    print("- category: Numeric category ID (matches parquet 'category' column)")  
    print("- object: Object ID (matches parquet 'object' column)")
    print("- images: Array of images for this object")
    print("- frame_index: Specific frame (matches parquet 'frame' column)")
    
    # Test this understanding
    print(f"\n=== Testing Understanding ===")
    sample = df.iloc[0]
    print(f"Sample row: {dict(sample)}")
    
    partition = sample['partition']
    category = str(sample['category'])
    obj = sample['object']
    frame = sample['frame']
    
    with h5py.File(main_h5, 'r') as f:
        try:
            images = f[partition][category][obj]['images']
            img = images[frame]
            print(f"✓ Successfully accessed image: shape={img.shape}")
            print(f"  Path: H5[{partition}][{category}][{obj}]['images'][{frame}]")
        except Exception as e:
            print(f"✗ Error accessing image: {e}")

if __name__ == "__main__":
    main() 