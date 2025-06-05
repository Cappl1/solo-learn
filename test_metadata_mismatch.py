#!/usr/bin/env python3
"""
Test to verify metadata vs H5 content mismatch
"""

import h5py
import pandas as pd

def test_metadata_vs_h5():
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    # Load metadata
    df = pd.read_parquet(metadata_path)
    print(f"Loaded metadata with {len(df)} rows")
    
    # Get first metadata entry
    first_row = df.iloc[0]
    metadata_partition = first_row['partition']
    metadata_category = str(first_row['category'])
    metadata_object = first_row['object']
    metadata_frame = first_row['frame']
    
    print(f"\nFirst metadata entry:")
    print(f"  Partition: {metadata_partition}")
    print(f"  Category: {metadata_category}")
    print(f"  Object: {metadata_object}")
    print(f"  Frame: {metadata_frame}")
    
    # Check what's actually in H5
    with h5py.File(h5_path, 'r') as f:
        h5_partitions = list(f.keys())
        print(f"\nH5 file partitions ({len(h5_partitions)} total):")
        for i, partition in enumerate(h5_partitions[:5]):
            print(f"  [{i}] {partition}")
        
        # Check if metadata partition exists
        metadata_exists = metadata_partition in h5_partitions
        print(f"\nMetadata partition '{metadata_partition}' exists in H5: {metadata_exists}")
        
        if metadata_exists:
            print("✓ Partition match found!")
            partition_group = f[metadata_partition]
            categories = list(partition_group.keys())
            print(f"  Partition has {len(categories)} categories: {categories[:10]}")
            
            category_exists = metadata_category in categories
            print(f"  Category '{metadata_category}' exists: {category_exists}")
            
            if category_exists:
                category_group = partition_group[metadata_category]
                objects = list(category_group.keys())
                print(f"  Category has {len(objects)} objects: {objects[:10]}")
                
                object_exists = metadata_object in objects
                print(f"  Object '{metadata_object}' exists: {object_exists}")
                
                if object_exists:
                    object_data = category_group[metadata_object]
                    print(f"  Object has {len(object_data)} frames")
                    frame_valid = metadata_frame < len(object_data)
                    print(f"  Frame {metadata_frame} valid: {frame_valid}")
                    
                    if frame_valid:
                        print("✓ Complete path is valid!")
                    else:
                        print(f"✗ Frame index {metadata_frame} >= {len(object_data)}")
                else:
                    print(f"✗ Object '{metadata_object}' not found")
                    print(f"    Available objects: {objects[:10]}")
            else:
                print(f"✗ Category '{metadata_category}' not found")
                print(f"    Available categories: {categories[:10]}")
        else:
            print("✗ Partition mismatch!")
            print("This explains why the dataset is failing!")
    
    # Let's check a few more metadata entries to see the pattern
    print(f"\n=== Checking Multiple Metadata Entries ===")
    
    with h5py.File(h5_path, 'r') as f:
        h5_partitions_set = set(f.keys())
        
        valid_count = 0
        total_checked = min(100, len(df))
        
        for i in range(total_checked):
            row = df.iloc[i]
            partition = row['partition']
            if partition in h5_partitions_set:
                valid_count += 1
        
        print(f"Out of first {total_checked} metadata entries:")
        print(f"  Valid partitions: {valid_count}")
        print(f"  Invalid partitions: {total_checked - valid_count}")
        print(f"  Validity rate: {valid_count/total_checked*100:.1f}%")
        
        if valid_count == 0:
            print("\n⚠️  NO metadata entries match H5 partitions!")
            print("This suggests the metadata file is for a different H5 file or version.")
        elif valid_count < total_checked:
            print(f"\n⚠️  Only {valid_count/total_checked*100:.1f}% of metadata entries are valid.")
            print("This suggests partial mismatch between metadata and H5 file.")

if __name__ == "__main__":
    test_metadata_vs_h5() 