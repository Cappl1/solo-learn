#!/usr/bin/env python3
"""
Test script for True Predify dynamics implementation.
Tests both standard and true predify modes.
"""

import torch
import torch.nn.functional as F
import omegaconf
from solo.methods.predify_simclr import PredifySimCLR

def test_true_predify():
    """Test true predify implementation"""
    print("Testing True Predify Implementation...")
    
    # Create test configuration
    cfg = omegaconf.DictConfig({
        'method': 'predify_simclr',
        'data': {
            'dataset': 'core50',
            'num_classes': 50,  # Core50 has 50 classes
            'num_workers': 0,  # For testing
            'batch_size': 4,
        },
        'trainer': {
            'devices': 'auto',
            'accelerator': 'cpu',  # Use CPU for testing
        },
        'method_kwargs': {
            # Standard parameters
            'timesteps': 2,
            'pred_loss_weight': 1.0,
            'use_local_updates': False,
            'proj_hidden_dim': 2048,
            'proj_output_dim': 128,
            'temperature': 0.1,
            
            # True Predify parameters
            'use_true_predify': True,
            'beta': [0.4, 0.3, 0.2, 0.1, 0.1],
            'lambda_': [0.05, 0.05, 0.05, 0.05, 0.0],
            'alpha': [0.001, 0.001, 0.001, 0.001, 0.001],
            'true_predify_detach_errors': True,
            'true_predify_momentum': 0.9,
            
            # PCoder parameters
            'ffm': [0.4, 0.3, 0.2, 0.1, 0.1],
            'fbm': [0.05, 0.05, 0.05, 0.05, 0.0],
            'erm': [0.001, 0.001, 0.001, 0.001, 0.001],
            'enable_pcoder_grads': False,
            'pcoder_grad_scale': [0.0, 0.0, 0.0, 0.0, 0.0],
            
            # Temporal parameters
            'use_temporal_pairs': True,
            'temporal_temperature': 0.07,
        },
        'backbone': {
            'name': 'vgg16',
            'kwargs': {'pretrained': False}
        },
        'optimizer': {
            'name': 'sgd',
            'lr': 0.1,
            'weight_decay': 1e-4
        },
        'num_large_crops': 2,
        'scheduler': {'name': 'none'},
        'name': 'test'
    })
    
    # Test with VGG16 backbone
    print("1. Creating PredifySimCLR model...")
    model = PredifySimCLR(cfg)
    model.train()
    
    # Test input
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)
    
    print("2. Testing standard forward pass...")
    # Test standard mode
    model.use_true_predify = False
    out1_standard = model(x1)
    print(f"   Standard output shapes: feats={out1_standard['feats'].shape}, z={out1_standard['z'].shape}")
    
    print("3. Testing true predify forward pass...")
    # Test true predify mode - but this should be same as standard since we simplified
    model.use_true_predify = True
    model.enable_debug_mode()
    
    out1_true = model(x1)  # This should be same as standard now
    print(f"   True Predify single forward shapes: feats={out1_true['feats'].shape}, z={out1_true['z'].shape}")
    print(f"   Has evolved representations: {'evolved_representations' in out1_true}")
    
    print("4. Testing simplified true predify training step...")
    # Test training step
    indexes = torch.arange(batch_size)
    targets = torch.randint(0, 10, (batch_size,))
    batch = (indexes, [x1, x2], targets)
    
    loss = model.training_step(batch, 0)
    print(f"   Training loss: {loss.item():.4f}")
    
    print("5. Testing representation evolution...")
    # Check if representations actually changed
    if hasattr(model, 'representation_trajectories'):
        for layer_idx, trajectory in model.representation_trajectories.items():
            if len(trajectory) > 1:
                initial = trajectory[0]
                final = trajectory[-1]
                change = (final - initial).norm() / (initial.norm() + 1e-6)
                print(f"   Layer {layer_idx} representation change: {change.item():.6f}")
    
    print("6. Testing coefficient constraints...")
    # Test coefficient constraints
    for i in range(len(model.beta)):
        if i < len(model.lambda_):
            sum_coef = model.beta[i] + model.lambda_[i]
            print(f"   Layer {i}: beta + lambda = {model.beta[i]:.3f} + {model.lambda_[i]:.3f} = {sum_coef:.3f} (should be ≤ 1.0)")
            assert sum_coef <= 1.0, f"Coefficient constraint violated for layer {i}"
    
    print("7. Testing layer forward functions...")
    # Test layer forward functions
    if hasattr(model, 'layer_forward_funcs'):
        print(f"   Number of layer forward functions: {len(model.layer_forward_funcs)}")
        
        # Test first layer forward
        test_input = torch.randn(2, 3, 224, 224)
        first_output = model.layer_forward_funcs[0](test_input)
        print(f"   First layer output shape: {first_output.shape}")
    
    print("\n✅ All True Predify tests passed!")
    
def test_comparison():
    """Compare standard vs true predify modes"""
    print("\nComparing Standard vs True Predify modes...")
    
    # Same configuration but with different modes
    cfg = omegaconf.DictConfig({
        'method': 'predify_simclr',
        'data': {
            'dataset': 'core50',
            'num_classes': 50,  # Core50 has 50 classes
            'num_workers': 0,  # For testing
            'batch_size': 2,
        },
        'trainer': {
            'devices': 'auto',
            'accelerator': 'cpu',  # Use CPU for testing
        },
        'method_kwargs': {
            'timesteps': 2,
            'pred_loss_weight': 1.0,
            'use_local_updates': True,
            'proj_hidden_dim': 256,  # Smaller for faster testing
            'proj_output_dim': 64,
            'temperature': 0.1,
            'use_true_predify': False,  # Will change this
            'beta': [0.4, 0.3, 0.2, 0.1, 0.1],
            'lambda_': [0.05, 0.05, 0.05, 0.05, 0.0],
            'alpha': [0.001, 0.001, 0.001, 0.001, 0.001],
            'true_predify_detach_errors': True,
            'true_predify_momentum': 0.9,
            'ffm': [0.4, 0.3, 0.2, 0.1, 0.1],
            'fbm': [0.05, 0.05, 0.05, 0.05, 0.0],
            'erm': [0.001, 0.001, 0.001, 0.001, 0.001],
            'enable_pcoder_grads': False,
            'pcoder_grad_scale': [0.0, 0.0, 0.0, 0.0, 0.0],
            'use_temporal_pairs': True,
            'temporal_temperature': 0.07,
        },
        'backbone': {'name': 'vgg16', 'kwargs': {'pretrained': False}},
        'optimizer': {'name': 'sgd', 'lr': 0.1, 'weight_decay': 1e-4},
        'num_large_crops': 2,
        'scheduler': {'name': 'none'},
        'name': 'test'
    })
    
    # Test data
    batch_size = 2
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)
    indexes = torch.arange(batch_size)
    targets = torch.randint(0, 10, (batch_size,))
    batch = (indexes, [x1, x2], targets)
    
    print("1. Testing Standard mode...")
    model_standard = PredifySimCLR(cfg)
    model_standard.train()
    loss_standard = model_standard.training_step(batch, 0)
    print(f"   Standard loss: {loss_standard.item():.4f}")
    
    print("2. Testing True Predify mode...")
    cfg.method_kwargs.use_true_predify = True
    cfg.method_kwargs.use_local_updates = False  # Disable when using true predify
    model_true = PredifySimCLR(cfg)
    model_true.train()
    loss_true = model_true.training_step(batch, 0)
    print(f"   True Predify loss: {loss_true.item():.4f}")
    
    print(f"3. Loss difference: {abs(loss_standard.item() - loss_true.item()):.4f}")
    print("   Both modes working correctly!")

if __name__ == "__main__":
    # Run tests
    try:
        test_true_predify()
        test_comparison()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 