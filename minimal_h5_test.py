#!/usr/bin/env python3
"""
Minimal H5 test to avoid hanging
"""

import h5py
import sys
import os

def test_h5_basics():
    h5_path = "/home/data/MVImageNet/data_all.h5"
    
    print(f"Testing H5 file: {h5_path}")
    print(f"File exists: {os.path.exists(h5_path)}")
    
    if not os.path.exists(h5_path):
        print("File doesn't exist!")
        return
    
    # Check file size
    file_size = os.path.getsize(h5_path)
    print(f"File size: {file_size / (1024**3):.2f} GB")
    
    # Try to open H5 file
    try:
        print("Opening H5 file...")
        with h5py.File(h5_path, 'r') as f:
            print(f"✓ H5 file opened successfully")
            print(f"Number of top-level groups: {len(f.keys())}")
            
            # Get first few keys
            keys = list(f.keys())[:3]
            print(f"First 3 keys: {keys}")
            
            # Test accessing first group
            if keys:
                first_key = keys[0]
                print(f"Accessing first group: {first_key}")
                group = f[first_key]
                subkeys = list(group.keys())[:3]
                print(f"First group has {len(group.keys())} items, first 3: {subkeys}")
                
                # Test accessing first subgroup
                if subkeys:
                    first_subkey = subkeys[0]
                    print(f"Accessing first category: {first_subkey}")
                    subgroup = group[first_subkey]
                    sub_subkeys = list(subgroup.keys())[:3]
                    print(f"First category has {len(subgroup.keys())} objects, first 3: {sub_subkeys}")
                    
                    # Test accessing first object
                    if sub_subkeys:
                        first_obj = sub_subkeys[0]
                        print(f"Accessing first object: {first_obj}")
                        obj_data = subgroup[first_obj]
                        print(f"Object data shape: {obj_data.shape}, dtype: {obj_data.dtype}")
                        
                        # Test accessing first frame
                        if len(obj_data) > 0:
                            print(f"Accessing first frame...")
                            frame_data = obj_data[0]
                            print(f"Frame data type: {type(frame_data)}")
                            if hasattr(frame_data, '__len__'):
                                print(f"Frame data size: {len(frame_data)} bytes")
                            
                            # Try to decode as image
                            try:
                                import io
                                from PIL import Image
                                
                                if isinstance(frame_data, bytes):
                                    img_bytes = frame_data
                                else:
                                    img_bytes = frame_data.tobytes()
                                
                                img = Image.open(io.BytesIO(img_bytes))
                                print(f"✓ Successfully decoded image: {img.size}, mode: {img.mode}")
                                
                            except Exception as e:
                                print(f"✗ Image decoding failed: {e}")
                        
    except Exception as e:
        print(f"✗ H5 file access failed: {e}")
        import traceback
        traceback.print_exc()

def test_metadata_load():
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    print(f"\nTesting metadata file: {metadata_path}")
    print(f"File exists: {os.path.exists(metadata_path)}")
    
    if not os.path.exists(metadata_path):
        print("Metadata file doesn't exist!")
        return
    
    # Check file size
    file_size = os.path.getsize(metadata_path)
    print(f"Metadata file size: {file_size / (1024**2):.2f} MB")
    
    try:
        print("Loading metadata...")
        import pandas as pd
        
        # Try to load just the first few rows
        df = pd.read_parquet(metadata_path, engine='fastparquet')
        print(f"✓ Metadata loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print("First row:")
        print(df.iloc[0])
        
    except Exception as e:
        print(f"✗ Metadata loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Minimal H5 Test ===")
    test_h5_basics()
    test_metadata_load() 