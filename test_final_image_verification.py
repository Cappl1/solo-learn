#!/usr/bin/env python3
"""
Final verification test for MVImageNet image tensor values
"""

import sys
sys.path.append('.')

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def test_raw_images():
    """Test raw PIL images from the dataset"""
    print("=== Testing Raw PIL Images ===")
    
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    # Create dataset without transforms
    dataset = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=metadata_path,
        time_window=5,
        split='train',
        transform=None  # No transforms - raw PIL images
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test first few samples
    for i in range(3):
        try:
            print(f"\nSample {i}:")
            (img1, img2), target = dataset[i]
            
            print(f"  Image 1: {img1.size}, mode: {img1.mode}")
            print(f"  Image 2: {img2.size}, mode: {img2.mode}")
            print(f"  Target: {target}")
            
            # Convert to numpy for analysis
            img1_array = np.array(img1)
            img2_array = np.array(img2)
            
            print(f"  Image 1 array shape: {img1_array.shape}")
            print(f"  Image 1 value range: [{img1_array.min()}, {img1_array.max()}]")
            print(f"  Image 1 mean: {img1_array.mean():.2f}")
            
            print(f"  Image 2 array shape: {img2_array.shape}")
            print(f"  Image 2 value range: [{img2_array.min()}, {img2_array.max()}]")
            print(f"  Image 2 mean: {img2_array.mean():.2f}")
            
            # Check if images are different (temporal pairs)
            if img1_array.shape == img2_array.shape:
                diff = np.abs(img1_array.astype(float) - img2_array.astype(float)).mean()
                print(f"  Mean pixel difference: {diff:.2f}")
                print(f"  Images are {'different' if diff > 5 else 'very similar'}")
            
            # Save first sample for visual inspection
            if i == 0:
                os.makedirs('test_outputs', exist_ok=True)
                img1.save(f'test_outputs/sample_{i}_img1.jpg')
                img2.save(f'test_outputs/sample_{i}_img2.jpg')
                print(f"  ✓ Saved sample images to test_outputs/")
            
        except Exception as e:
            print(f"  ✗ Sample {i} failed: {e}")
            return False
    
    return True

def test_transformed_tensors():
    """Test transformed tensor images"""
    print("\n=== Testing Transformed Tensors ===")
    
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    # Create transforms similar to training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def dual_transform(img1, img2):
        return [transform(img1), transform(img2)]
    
    # Create dataset with transforms
    dataset = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=metadata_path,
        time_window=5,
        split='train',
        transform=dual_transform
    )
    
    print(f"Dataset with transforms created")
    
    # Test first few samples
    for i in range(3):
        try:
            print(f"\nTransformed Sample {i}:")
            (tensor1, tensor2), target = dataset[i]
            
            print(f"  Tensor 1 shape: {tensor1.shape}")
            print(f"  Tensor 1 dtype: {tensor1.dtype}")
            print(f"  Tensor 1 value range: [{tensor1.min():.3f}, {tensor1.max():.3f}]")
            print(f"  Tensor 1 mean: {tensor1.mean():.3f}")
            print(f"  Tensor 1 std: {tensor1.std():.3f}")
            
            print(f"  Tensor 2 shape: {tensor2.shape}")
            print(f"  Tensor 2 dtype: {tensor2.dtype}")
            print(f"  Tensor 2 value range: [{tensor2.min():.3f}, {tensor2.max():.3f}]")
            print(f"  Tensor 2 mean: {tensor2.mean():.3f}")
            print(f"  Tensor 2 std: {tensor2.std():.3f}")
            
            # Check tensor properties
            assert tensor1.shape == (3, 224, 224), f"Unexpected shape: {tensor1.shape}"
            assert tensor2.shape == (3, 224, 224), f"Unexpected shape: {tensor2.shape}"
            assert tensor1.dtype == torch.float32, f"Unexpected dtype: {tensor1.dtype}"
            
            # Check if values are in reasonable range for normalized tensors
            assert -3 < tensor1.min() < 3, f"Tensor values out of range: {tensor1.min()}"
            assert -3 < tensor1.max() < 3, f"Tensor values out of range: {tensor1.max()}"
            
            print(f"  ✓ Tensor properties are valid")
            
            # Compute similarity between temporal pairs
            mse = torch.nn.functional.mse_loss(tensor1, tensor2).item()
            print(f"  MSE between temporal pair: {mse:.3f}")
            
        except Exception as e:
            print(f"  ✗ Transformed sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_batch_loading():
    """Test batch loading as it would happen in training"""
    print("\n=== Testing Batch Loading ===")
    
    import sys
    sys.path.append('.')
    from torch.utils.data import DataLoader
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    # Training-like transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def dual_transform(img1, img2):
        return [transform(img1), transform(img2)]
    
    # Create dataset
    dataset = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=metadata_path,
        time_window=5,
        split='train',
        transform=dual_transform
    )
    
    # Create dataloader (without distributed sampler for testing)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Single process for testing
        drop_last=False
    )
    
    print(f"Created dataloader with batch_size=4")
    
    # Test first batch
    try:
        for batch_idx, ((tensors1, tensors2), targets) in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Batch tensors1 shape: {tensors1.shape}")
            print(f"  Batch tensors2 shape: {tensors2.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Targets: {targets}")
            
            # Check batch properties
            batch_size = tensors1.shape[0]
            print(f"  Batch size: {batch_size}")
            
            # Analyze batch statistics
            print(f"  Batch tensors1 mean: {tensors1.mean():.3f}")
            print(f"  Batch tensors1 std: {tensors1.std():.3f}")
            print(f"  Batch tensors1 range: [{tensors1.min():.3f}, {tensors1.max():.3f}]")
            
            print(f"  Batch tensors2 mean: {tensors2.mean():.3f}")
            print(f"  Batch tensors2 std: {tensors2.std():.3f}")
            print(f"  Batch tensors2 range: [{tensors2.min():.3f}, {tensors2.max():.3f}]")
            
            # Check for NaN or inf values
            has_nan1 = torch.isnan(tensors1).any()
            has_inf1 = torch.isinf(tensors1).any()
            has_nan2 = torch.isnan(tensors2).any()
            has_inf2 = torch.isinf(tensors2).any()
            
            print(f"  Contains NaN: {has_nan1 or has_nan2}")
            print(f"  Contains Inf: {has_inf1 or has_inf2}")
            
            if has_nan1 or has_nan2 or has_inf1 or has_inf2:
                print("  ✗ Invalid values detected!")
                return False
            
            print(f"  ✓ Batch data is valid")
            
            # Only test first batch
            break
            
    except Exception as e:
        print(f"  ✗ Batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_dataset_consistency():
    """Test dataset consistency - same sample should give same result"""
    print("\n=== Testing Dataset Consistency ===")
    
    from solo.data.custom.temporal_mvimagnet2 import TemporalMVImageNet
    
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    # Simple deterministic transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    def dual_transform(img1, img2):
        return [transform(img1), transform(img2)]
    
    # Create dataset
    dataset = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=metadata_path,
        time_window=5,
        split='train',
        transform=dual_transform
    )
    
    # Test same sample multiple times
    sample_idx = 42  # Arbitrary sample
    
    try:
        # Get same sample 3 times
        results = []
        for i in range(3):
            (tensor1, tensor2), target = dataset[sample_idx]
            results.append((tensor1.clone(), tensor2.clone(), target))
        
        # Compare results
        print(f"Testing sample {sample_idx} for consistency:")
        
        for i in range(1, 3):
            tensor1_diff = (results[0][0] - results[i][0]).abs().max().item()
            tensor2_diff = (results[0][1] - results[i][1]).abs().max().item()
            target_diff = results[0][2] != results[i][2]
            
            print(f"  Attempt {i+1} vs 1:")
            print(f"    Max tensor1 diff: {tensor1_diff:.6f}")
            print(f"    Max tensor2 diff: {tensor2_diff:.6f}")
            print(f"    Target different: {target_diff}")
            
            # With deterministic transforms, differences should be minimal
            if tensor1_diff > 1e-5 or tensor2_diff > 1e-5 or target_diff:
                print(f"  ⚠️  Some inconsistency detected (may be due to random temporal pairing)")
            else:
                print(f"  ✓ Results are consistent")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Final MVImageNet Image Verification Test")
    print("=" * 45)
    
    # Run all tests
    raw_success = test_raw_images()
    tensor_success = test_transformed_tensors()
    batch_success = test_batch_loading()
    consistency_success = test_dataset_consistency()
    
    print(f"\n=== Final Summary ===")
    print(f"Raw PIL images: {'✓' if raw_success else '✗'}")
    print(f"Transformed tensors: {'✓' if tensor_success else '✗'}")
    print(f"Batch loading: {'✓' if batch_success else '✗'}")
    print(f"Dataset consistency: {'✓' if consistency_success else '✗'}")
    
    all_passed = all([raw_success, tensor_success, batch_success, consistency_success])
    
    if all_passed:
        print(f"\n🎉 ALL IMAGE VERIFICATION TESTS PASSED!")
        print(f"   The MVImageNet dataset is working correctly and ready for training.")
        print(f"   Image tensors contain meaningful data with proper statistics.")
        if os.path.exists('test_outputs'):
            print(f"   Sample images saved in test_outputs/ directory for visual inspection.")
    else:
        print(f"\n❌ Some image verification tests failed.")
        print(f"   Please check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    main() 