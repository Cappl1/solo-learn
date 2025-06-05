#!/usr/bin/env python3
"""
Step-by-step test of the dataset class to find the exact failure point
"""

import sys
sys.path.append('.')

import h5py
import pandas as pd
import io
from PIL import Image

def test_dataset_creation():
    """Test just the dataset creation without sample access"""
    
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    print("=== Testing Dataset Creation ===")
    
    try:
        print("Creating dataset...")
        dataset = TemporalMVImageNet(
            h5_path=h5_path,
            metadata_path=metadata_path,
            time_window=5,
            split='train',
            transform=None  # No transform for testing
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Sequence map size: {len(dataset.sequence_map)}")
        print(f"  Flat indices size: {len(dataset.flat_indices)}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_manual_sample_access(dataset):
    """Test manual sample access following the dataset code path"""
    
    if dataset is None:
        print("Cannot test - dataset is None")
        return
    
    print(f"\n=== Testing Manual Sample Access ===")
    
    h5_path = dataset.h5_path
    
    # Get the first sample's metadata
    df_idx = dataset.flat_indices[0]
    sample = dataset.df.iloc[df_idx]
    
    print(f"Sample 0 info:")
    print(f"  df_idx: {df_idx}")
    print(f"  partition: {sample['partition']}")
    print(f"  category: {sample['category']}")
    print(f"  object: {sample['object']}")
    print(f"  frame: {sample['frame']}")
    
    # Manual H5 access following dataset code
    try:
        print(f"\nManual H5 access:")
        with h5py.File(h5_path, 'r', swmr=True, libver='latest') as h5_file:
            # Extract H5 path components
            partition = sample['partition']
            category = str(sample['category'])
            obj = sample['object']
            frame = sample['frame']
            
            print(f"  Accessing: {partition}/{category}/{obj}[{frame}]")
            
            # Get the image data
            img_data = h5_file[partition][category][obj][frame]
            print(f"  Image data type: {type(img_data)}")
            
            # Convert to bytes
            if isinstance(img_data, bytes):
                img_bytes = img_data
            else:
                img_bytes = img_data.tobytes()
            
            print(f"  Image bytes length: {len(img_bytes)}")
            
            # Decode image
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            print(f"  ✓ Image decoded: {image.size}, mode: {image.mode}")
            
            return True
            
    except Exception as e:
        print(f"  ✗ Manual access failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_getitem(dataset):
    """Test the dataset __getitem__ method"""
    
    if dataset is None:
        print("Cannot test - dataset is None")
        return
    
    print(f"\n=== Testing Dataset __getitem__ ===")
    
    try:
        print("Calling dataset[0]...")
        result = dataset[0]
        print(f"✓ dataset[0] succeeded")
        print(f"  Result type: {type(result)}")
        
        if isinstance(result, tuple) and len(result) >= 2:
            data, target = result
            print(f"  Data type: {type(data)}")
            print(f"  Target: {target}")
            
            if isinstance(data, tuple) and len(data) >= 2:
                img1, img2 = data
                print(f"  Image 1: {img1.size if hasattr(img1, 'size') else type(img1)}")
                print(f"  Image 2: {img2.size if hasattr(img2, 'size') else type(img2)}")
        
        return True
        
    except Exception as e:
        print(f"✗ dataset[0] failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_transforms():
    """Test with transforms to see if that's the issue"""
    
    print(f"\n=== Testing With Transforms ===")
    
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    import torchvision.transforms as transforms
    
    # Simple transform
    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    def dual_transform(img1, img2):
        return [simple_transform(img1), simple_transform(img2)]
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    try:
        print("Creating dataset with transforms...")
        dataset = TemporalMVImageNet(
            h5_path=h5_path,
            metadata_path=metadata_path,
            time_window=5,
            split='train',
            transform=dual_transform
        )
        
        print(f"✓ Dataset with transforms created")
        
        # Test sample access
        print("Testing sample access with transforms...")
        result = dataset[0]
        print(f"✓ Sample access with transforms succeeded")
        print(f"  Result type: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Step-by-step Dataset Testing")
    print("=" * 40)
    
    # Test 1: Dataset creation
    dataset = test_dataset_creation()
    
    # Test 2: Manual sample access
    manual_success = test_manual_sample_access(dataset)
    
    # Test 3: Dataset __getitem__
    getitem_success = test_dataset_getitem(dataset)
    
    # Test 4: With transforms
    transform_success = test_with_transforms()
    
    print(f"\n=== Summary ===")
    print(f"Dataset creation: {'✓' if dataset is not None else '✗'}")
    print(f"Manual H5 access: {'✓' if manual_success else '✗'}")
    print(f"Dataset __getitem__: {'✓' if getitem_success else '✗'}")
    print(f"With transforms: {'✓' if transform_success else '✗'}")

if __name__ == "__main__":
    main() 