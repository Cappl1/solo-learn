#!/usr/bin/env python3
"""
Debug script to understand MVImageNet H5 file structure
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def explore_h5_structure(h5_path, max_levels=3, max_items_per_level=10):
    """Recursively explore H5 file structure"""
    
    def explore_group(group, level=0, max_level=3, path=""):
        if level > max_level:
            return
            
        items = list(group.keys())[:max_items_per_level]
        
        for i, key in enumerate(items):
            current_path = f"{path}/{key}" if path else key
            print("  " * level + f"[{i}] {key}")
            
            try:
                item = group[key]
                if hasattr(item, 'shape'):
                    print("  " * level + f"    Shape: {item.shape}, Dtype: {item.dtype}")
                elif hasattr(item, 'keys'):
                    print("  " * level + f"    Group with {len(item.keys())} items")
                    if level < max_level:
                        explore_group(item, level + 1, max_level, current_path)
                else:
                    print("  " * level + f"    Type: {type(item)}")
                    
            except Exception as e:
                print("  " * level + f"    Error accessing: {e}")
        
        if len(group.keys()) > max_items_per_level:
            print("  " * level + f"... and {len(group.keys()) - max_items_per_level} more items")
    
    print(f"Exploring H5 file: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        print(f"Root contains {len(f.keys())} top-level items")
        explore_group(f, max_level=max_levels)

def analyze_metadata_vs_h5(h5_path, metadata_path):
    """Compare metadata expectations vs H5 reality"""
    
    print(f"\n=== Analyzing Metadata vs H5 Structure ===")
    
    # Load metadata
    print(f"Loading metadata from: {metadata_path}")
    if metadata_path.endswith('.parquet'):
        df = pd.read_parquet(metadata_path)
    else:
        df = pd.read_csv(metadata_path)
    
    print(f"Metadata shape: {df.shape}")
    print(f"Metadata columns: {list(df.columns)}")
    print(f"Sample metadata rows:")
    print(df.head())
    
    # Get unique values from metadata
    if 'partition' in df.columns:
        unique_partitions = df['partition'].unique()
        print(f"\nUnique partitions in metadata: {unique_partitions[:10]}")
    
    if 'category' in df.columns:
        unique_categories = df['category'].unique()
        print(f"Unique categories in metadata: {unique_categories[:10]} (total: {len(unique_categories)})")
    
    if 'object' in df.columns:
        unique_objects = df['object'].unique()
        print(f"Sample unique objects: {unique_objects[:10]} (total: {len(unique_objects)})")
    
    # Now check what's actually in the H5 file
    print(f"\n=== Checking H5 File Structure ===")
    with h5py.File(h5_path, 'r') as f:
        h5_top_keys = list(f.keys())
        print(f"H5 top-level keys: {h5_top_keys[:10]}")
        
        # Check if metadata partitions exist in H5
        if 'partition' in df.columns:
            print(f"\nChecking if metadata partitions exist in H5:")
            for partition in unique_partitions[:5]:
                exists = partition in f
                print(f"  {partition}: {'✓' if exists else '✗'}")
                
                if exists:
                    partition_group = f[partition]
                    partition_keys = list(partition_group.keys())
                    print(f"    Contains {len(partition_keys)} items: {partition_keys[:5]}")
                    
                    # Check categories within partition
                    if 'category' in df.columns:
                        sample_categories = df[df['partition'] == partition]['category'].unique()[:3]
                        for cat in sample_categories:
                            cat_str = str(cat)
                            cat_exists = cat_str in partition_group
                            print(f"    Category {cat_str}: {'✓' if cat_exists else '✗'}")
                            
                            if cat_exists:
                                cat_group = partition_group[cat_str]
                                cat_objects = list(cat_group.keys())
                                print(f"      Objects: {cat_objects[:3]} (total: {len(cat_objects)})")

def test_specific_sample_access(h5_path, metadata_path):
    """Test accessing specific samples that are failing"""
    
    print(f"\n=== Testing Specific Sample Access ===")
    
    # Load metadata 
    if metadata_path.endswith('.parquet'):
        df = pd.read_parquet(metadata_path)
    else:
        df = pd.read_csv(metadata_path)
    
    # Get first few samples
    print("Testing access to first 5 metadata samples:")
    
    with h5py.File(h5_path, 'r') as f:
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            print(f"\nSample {i}:")
            print(f"  Metadata: partition={row.get('partition', 'N/A')}, category={row.get('category', 'N/A')}, object={row.get('object', 'N/A')}, frame={row.get('frame', 'N/A')}")
            
            # Try to access this sample in H5
            try:
                partition = row['partition']
                category = str(row['category'])
                obj = row['object']
                frame = row['frame']
                
                # Check each level
                print(f"  Checking partition '{partition}': {'✓' if partition in f else '✗'}")
                if partition in f:
                    partition_group = f[partition]
                    print(f"  Checking category '{category}': {'✓' if category in partition_group else '✗'}")
                    if category in partition_group:
                        category_group = partition_group[category]
                        print(f"  Checking object '{obj}': {'✓' if obj in category_group else '✗'}")
                        if obj in category_group:
                            object_group = category_group[obj]
                            print(f"  Object shape: {object_group.shape if hasattr(object_group, 'shape') else 'Not a dataset'}")
                            print(f"  Checking frame {frame}: {'✓' if frame < len(object_group) else f'✗ (max: {len(object_group)})'}")
                            
                            if frame < len(object_group):
                                img_data = object_group[frame]
                                print(f"  Frame data type: {type(img_data)}, size: {len(img_data) if hasattr(img_data, '__len__') else 'Unknown'}")
                        else:
                            available_objects = list(category_group.keys())[:5]
                            print(f"  Available objects: {available_objects}")
                    else:
                        available_categories = list(partition_group.keys())[:5]
                        print(f"  Available categories: {available_categories}")
                else:
                    available_partitions = list(f.keys())[:5]
                    print(f"  Available partitions: {available_partitions}")
                    
            except Exception as e:
                print(f"  Error: {e}")

def main():
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    if not Path(h5_path).exists():
        print(f"H5 file not found: {h5_path}")
        return
        
    if not Path(metadata_path).exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    try:
        print("=== H5 Structure Exploration ===")
        explore_h5_structure(h5_path, max_levels=3, max_items_per_level=5)
        
        analyze_metadata_vs_h5(h5_path, metadata_path)
        
        test_specific_sample_access(h5_path, metadata_path)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 