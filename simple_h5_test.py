#!/usr/bin/env python3
"""
Simple test to understand H5 access pattern issues
"""

import h5py
import pandas as pd
import io
from PIL import Image

def test_h5_access():
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    # Load metadata
    df = pd.read_parquet(metadata_path)
    print(f"Loaded {len(df)} metadata entries")
    print("First few rows:")
    print(df.head())
    
    # Test access to first sample
    print(f"\n=== Testing First Sample Access ===")
    row = df.iloc[0]
    print(f"Sample 0 metadata:")
    for col in ['partition', 'category', 'object', 'frame']:
        print(f"  {col}: {row[col]} (type: {type(row[col])})")
    
    # Try to access this sample
    with h5py.File(h5_path, 'r') as f:
        partition = row['partition']
        category = str(row['category'])
        obj = row['object']
        frame = row['frame']
        
        print(f"\nTrying to access: {partition}/{category}/{obj}[{frame}]")
        
        # Step by step access
        print(f"1. Partition '{partition}' exists: {partition in f}")
        if partition in f:
            p_group = f[partition]
            print(f"2. Category '{category}' exists: {category in p_group}")
            if category in p_group:
                c_group = p_group[category]
                print(f"3. Object '{obj}' exists: {obj in c_group}")
                if obj in c_group:
                    o_data = c_group[obj]
                    print(f"4. Object data shape: {o_data.shape}")
                    print(f"5. Frame {frame} valid: {frame < len(o_data)}")
                    if frame < len(o_data):
                        img_data = o_data[frame]
                        print(f"6. Image data type: {type(img_data)}, size: {len(img_data) if hasattr(img_data, '__len__') else 'unknown'}")
                        
                        # Try to decode image
                        try:
                            if isinstance(img_data, bytes):
                                img_bytes = img_data
                            else:
                                img_bytes = img_data.tobytes()
                            
                            img = Image.open(io.BytesIO(img_bytes))
                            print(f"7. Successfully decoded image: {img.size}, mode: {img.mode}")
                            
                        except Exception as e:
                            print(f"7. Image decoding failed: {e}")
                else:
                    available = list(c_group.keys())[:10]
                    print(f"   Available objects: {available}")
            else:
                available = list(p_group.keys())[:10]
                print(f"   Available categories: {available}")
        else:
            available = list(f.keys())[:10]
            print(f"   Available partitions: {available}")

def test_dataset_class():
    """Test the actual dataset class behavior"""
    
    print(f"\n=== Testing Dataset Class ===")
    
    import sys
    sys.path.append('.')
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    # Create dataset
    try:
        dataset = TemporalMVImageNet(
            h5_path=h5_path,
            metadata_path=metadata_path,
            time_window=5,
            split='train'
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Try to get first sample
        try:
            sample = dataset[0]
            print(f"Sample 0 loaded successfully: {type(sample)}")
            if isinstance(sample, tuple) and len(sample) >= 2:
                data, target = sample
                print(f"  Data type: {type(data)}")
                print(f"  Target: {target}")
                
                if isinstance(data, tuple) and len(data) >= 2:
                    img1, img2 = data
                    print(f"  Image 1: {img1.size if hasattr(img1, 'size') else type(img1)}")
                    print(f"  Image 2: {img2.size if hasattr(img2, 'size') else type(img2)}")
        except Exception as e:
            print(f"Sample loading failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_h5_access()
    test_dataset_class() 