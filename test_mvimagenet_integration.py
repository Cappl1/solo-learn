#!/usr/bin/env python3
"""
Test script to verify temporal MVImageNet integration with solo-learn
"""

import sys
import os
sys.path.append('.')

import torch
from omegaconf import DictConfig
from solo.data.StatefulDistributeSampler import DataPrepIterCheck
from solo.args.pretrain import parse_cfg
from solo.methods import METHODS
import traceback

def test_config():
    """Test loading the configuration"""
    print("=== Testing Configuration Loading ===")
    
    # Create a minimal config for testing
    cfg = DictConfig({
        'method': 'predify_simclr',
        'seed': 42,
        'data': {
            'dataset': 'temporal_mvimagenet',
            'train_path': '/home/data/MVImageNet/data_all.h5',
            'val_path': '/home/data/MVImageNet/data_all.h5',
            'format': 'h5',
            'num_workers': 2,  # Reduced for testing
            'dataset_kwargs': {
                'metadata_path': '/home/data/MVImageNet/dataset_val_all3.parquet',
                'time_window': 5,  # Smaller for testing
                'split': 'train'
            },
            'no_labels': False,
            'fraction': -1.0
        },
        'augmentations': [{
            'num_crops': 2,
            'crop_size': 224,
            'rrc': {'enabled': False},
            'color_jitter': {'prob': 0.0},
            'grayscale': {'prob': 0.0},
            'gaussian_blur': {'prob': 0.0},
            'solarization': {'prob': 0.0},
            'equalization': {'prob': 0.0},
            'horizontal_flip': {'prob': 0.0}
        }],
        'backbone': {
            'name': 'vgg16'
        },
        'method_kwargs': {
            'proj_hidden_dim': 512,  # Smaller for testing
            'proj_output_dim': 128,
            'temperature': 0.1,
            'timesteps': 2,  # Smaller for testing
            'pred_loss_weight': 1.0,
            'use_local_updates': True,
            'use_true_predify': False,
            'enable_pcoder_grads': True,
            'pcoder_grad_scale': [0.1, 0.1, 0.1, 0.1, 0.1],
            'ffm': [0.4, 0.3, 0.2, 0.1, 0.1],
            'fbm': [0.05, 0.05, 0.05, 0.05, 0.0],
            'erm': [0.001, 0.001, 0.001, 0.001, 0.001],
            'use_temporal_pairs': True
        },
        'optimizer': {
            'batch_size': 4,  # Very small for testing
            'lr': 0.01,
            'name': 'sgd',
            'weight_decay': 1e-6,
            'classifier_lr': 3e-3  # Required for validation data path
        },
        'debug_augmentations': False,
        'performance': {
            'disable_channel_last': True  # For testing
        },
        'knn_clb': {
            'enabled': False,  # Disable for testing
            'dataset': 'temporal_mvimagenet',  # Required even if disabled
            'train_path': '/home/data/MVImageNet/data_all.h5',
            'val_path': '/home/data/MVImageNet/data_all.h5',
            'num_workers': 2,
            'batch_size': 4,
            'k': [10, 20],
            'temperature': 0.07,
            'metadata_path': '/home/data/MVImageNet/dataset_val_all3.parquet',
            'time_window': 5,
            'split': 'train'
        },
        'wandb': {
            'enabled': False  # Disable for testing
        },
        'checkpoint': {
            'enabled': False  # Disable for testing
        },
        'auto_resume': {
            'enabled': False
        },
        'scheduler': {
            'name': 'none'
        },
        'max_epochs': 1,
        'devices': [0],
        'accelerator': 'cpu',  # Use CPU for testing
        'strategy': 'auto',
        'precision': 32
    })
    
    try:
        cfg = parse_cfg(cfg)
        print("✓ Configuration loaded successfully")
        return cfg
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        traceback.print_exc()
        return None

def test_dataset_loading(cfg):
    """Test dataset loading"""
    print("\n=== Testing Dataset Loading ===")
    
    if cfg is None:
        print("✗ Cannot test dataset - config is None")
        return None
    
    try:
        # Create data module
        data_module = DataPrepIterCheck(cfg)
        data_module.setup(stage='fit')
        
        print(f"✓ Dataset created successfully")
        print(f"  Dataset size: {len(data_module.train_dataset)}")
        
        # Test getting a single sample
        try:
            sample_idx, sample_data, sample_target = data_module.train_dataset[0]
            print(f"✓ Sample loaded successfully")
            print(f"  Sample index: {sample_idx}")
            print(f"  Sample data type: {type(sample_data)}")
            print(f"  Sample target: {sample_target}")
            
            if isinstance(sample_data, list) and len(sample_data) >= 2:
                img1, img2 = sample_data[0], sample_data[1]
                print(f"  Image 1 shape: {img1.shape}")
                print(f"  Image 2 shape: {img2.shape}")
            
        except Exception as e:
            print(f"✗ Sample loading failed: {e}")
            return None
        
        return data_module
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return None

def test_model_creation(cfg):
    """Test model creation"""
    print("\n=== Testing Model Creation ===")
    
    if cfg is None:
        print("✗ Cannot test model - config is None")
        return None
    
    try:
        model = METHODS[cfg.method](cfg)
        print(f"✓ Model created successfully")
        print(f"  Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None

def test_forward_pass(model, data_module):
    """Test forward pass"""
    print("\n=== Testing Forward Pass ===")
    
    if model is None or data_module is None:
        print("✗ Cannot test forward pass - model or data_module is None")
        return
    
    try:
        # Get a small batch
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"✓ Batch loaded successfully")
        print(f"  Batch type: {type(batch)}")
        
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            indexes, data = batch[0], batch[1]
            print(f"  Batch indexes shape: {indexes.shape}")
            print(f"  Batch data type: {type(data)}")
            
            if isinstance(data, list) and len(data) >= 2:
                print(f"  Data[0] shape: {data[0].shape}")
                print(f"  Data[1] shape: {data[1].shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch[1])  # Pass the data part
            print(f"✓ Forward pass successful")
            print(f"  Output keys: {list(output.keys()) if isinstance(output, dict) else type(output)}")
            
            if isinstance(output, dict):
                for key, value in output.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key} shape: {value.shape}")
                    else:
                        print(f"  {key} type: {type(value)}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

def main():
    """Main test function"""
    print("Temporal MVImageNet Integration Test")
    print("=" * 40)
    
    # Check file availability first
    h5_path = "/home/data/MVImageNet/data_all.h5"
    metadata_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    
    print("Checking file availability:")
    print(f"  H5 file: {os.path.exists(h5_path)} - {h5_path}")
    print(f"  Metadata: {os.path.exists(metadata_path)} - {metadata_path}")
    
    if not os.path.exists(h5_path):
        print("\nERROR: H5 file not found. Please update the path.")
        return
        
    if not os.path.exists(metadata_path):
        print("\nERROR: Metadata file not found. Please update the path.")
        return
    
    # Run tests
    cfg = test_config()
    data_module = test_dataset_loading(cfg)
    model = test_model_creation(cfg)
    test_forward_pass(model, data_module)
    
    print(f"\n=== Summary ===")
    print(f"Config: {'✓' if cfg is not None else '✗'}")
    print(f"Dataset: {'✓' if data_module is not None else '✗'}")
    print(f"Model: {'✓' if model is not None else '✗'}")

if __name__ == "__main__":
    main() 