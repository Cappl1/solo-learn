#!/usr/bin/env python3

import h5py
import pandas as pd
import sys
from pathlib import Path

def debug_h5_structure():
    """Detailed debugging of H5 structure vs metadata expectations"""
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    print("=== H5 FILE STRUCTURE ANALYSIS ===")
    
    # 1. Check H5 file structure
    try:
        with h5py.File(h5_path, 'r') as h5_file:
            print(f"\nH5 File: {h5_path}")
            print(f"Top-level keys (first 10): {list(h5_file.keys())[:10]}")
            print(f"Total top-level keys: {len(h5_file.keys())}")
            
            # Check first partition structure
            first_key = list(h5_file.keys())[0]
            print(f"\nFirst partition: {first_key}")
            first_partition = h5_file[first_key]
            print(f"Categories in first partition: {list(first_partition.keys())[:10]}")
            
            # Check first category in first partition
            if len(first_partition.keys()) > 0:
                first_category = list(first_partition.keys())[0]
                print(f"First category: {first_category}")
                category_group = first_partition[first_category]
                print(f"Objects in first category: {list(category_group.keys())[:5]}")
                
                # Check first object
                if len(category_group.keys()) > 0:
                    first_object = list(category_group.keys())[0]
                    print(f"First object: {first_object}")
                    object_data = category_group[first_object]
                    print(f"Object data shape: {object_data.shape}")
                    print(f"Object data dtype: {object_data.dtype}")
                    print(f"Number of frames: {len(object_data)}")
    
    except Exception as e:
        print(f"Error reading H5 file: {e}")
        return
    
    print("\n=== METADATA ANALYSIS ===")
    
    # 2. Check metadata structure
    try:
        df = pd.read_parquet(metadata_path)
        print(f"\nMetadata file: {metadata_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check unique values in key columns
        if 'partition' in df.columns:
            print(f"\nUnique partitions (first 10): {df['partition'].unique()[:10]}")
            print(f"Total partitions: {df['partition'].nunique()}")
        
        if 'category' in df.columns:
            print(f"\nUnique categories: {sorted(df['category'].unique())[:20]}")
            print(f"Total categories: {df['category'].nunique()}")
        
        if 'object' in df.columns:
            print(f"\nSample objects: {df['object'].unique()[:10]}")
            print(f"Total objects: {df['object'].nunique()}")
        
        if 'frame' in df.columns:
            print(f"\nFrame range: {df['frame'].min()} - {df['frame'].max()}")
    
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return
    
    print("\n=== MISMATCH ANALYSIS ===")
    
    # 3. Try to match metadata to H5 structure
    try:
        with h5py.File(h5_path, 'r') as h5_file:
            print(f"\nTesting metadata-to-H5 mapping...")
            
            # Take first few samples from metadata
            sample_rows = df.head(5)
            
            for idx, row in sample_rows.iterrows():
                partition = row['partition']
                category = str(row['category'])
                obj = row['object']
                frame = row['frame']
                
                print(f"\nSample {idx}:")
                print(f"  Metadata: partition={partition}, category={category}, object={obj}, frame={frame}")
                
                # Check if path exists in H5
                path_exists = []
                
                # Check partition
                if partition in h5_file:
                    path_exists.append(f"✓ Partition '{partition}' exists")
                    partition_group = h5_file[partition]
                    
                    # Check category
                    if category in partition_group:
                        path_exists.append(f"✓ Category '{category}' exists")
                        category_group = partition_group[category]
                        
                        # Check object
                        if obj in category_group:
                            path_exists.append(f"✓ Object '{obj}' exists")
                            object_data = category_group[obj]
                            
                            # Check frame
                            if frame < len(object_data):
                                path_exists.append(f"✓ Frame {frame} exists (dataset length: {len(object_data)})")
                                
                                # Try to load the actual frame
                                try:
                                    frame_data = object_data[frame]
                                    path_exists.append(f"✓ Frame data loaded: {type(frame_data)}, size: {len(frame_data) if hasattr(frame_data, '__len__') else 'N/A'}")
                                except Exception as e:
                                    path_exists.append(f"✗ Frame loading failed: {e}")
                            else:
                                path_exists.append(f"✗ Frame {frame} out of range (dataset length: {len(object_data)})")
                        else:
                            path_exists.append(f"✗ Object '{obj}' not found. Available: {list(category_group.keys())[:5]}")
                    else:
                        path_exists.append(f"✗ Category '{category}' not found. Available: {list(partition_group.keys())[:5]}")
                else:
                    path_exists.append(f"✗ Partition '{partition}' not found. Available: {list(h5_file.keys())[:5]}")
                
                for status in path_exists:
                    print(f"    {status}")
                    
                # Stop at first failure for detailed analysis
                if any("✗" in status for status in path_exists):
                    break
    
    except Exception as e:
        print(f"Error in mismatch analysis: {e}")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. If partitions don't match, update metadata file or fix partition mapping")
    print("2. If categories are numeric in metadata but string in H5, add str() conversion")
    print("3. If objects don't exist, check for object naming mismatches")
    print("4. If frames are out of range, check for frame indexing issues")


if __name__ == "__main__":
    debug_h5_structure() 