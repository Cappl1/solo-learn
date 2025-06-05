#!/usr/bin/env python3
"""
Test script to verify that local updates are working properly in PredifySimCLR.
This script creates a minimal test case to check if timestep losses differ when using local updates.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

# Assuming the solo package is in the path
from solo.methods.predify_simclr import PredifySimCLR
from solo.backbones import vgg16

def test_local_updates():
    """Test that local updates produce different losses across timesteps"""
    
    # Create minimal config
    cfg = DictConfig({
        'backbone': {
            'name': 'vgg16',
            'kwargs': {}
        },
        'method_kwargs': {
            'proj_hidden_dim': 512,
            'proj_output_dim': 128,
            'temperature': 0.1,
            'timesteps': 3,  # Use 3 timesteps to see progression
            'pred_loss_weight': 1.0,
            'ffm': [0.4, 0.3, 0.2, 0.1, 0.1],
            'fbm': [0.05, 0.05, 0.05, 0.0, 0.0],
            'erm': [0.001, 0.001, 0.001, 0.001, 0.001],
            'use_local_updates': True,  # Enable local updates
            'use_temporal_pairs': True,
            'temporal_temperature': 0.07,
            'pcoder_grad_scale': [0.1, 0.08, 0.06, 0.04, 0.02],
            'enable_pcoder_grads': False,  # Disable for testing
        }
    })
    
    # Create model
    model = PredifySimCLR(cfg)
    model.eval()  # Set to eval to avoid training complexities
    
    # Enable debug mode to see detailed logs
    model.enable_debug_mode()
    
    # Create dummy data
    batch_size = 2
    image_size = 224
    x1 = torch.randn(batch_size, 3, image_size, image_size)
    x2 = torch.randn(batch_size, 3, image_size, image_size)
    
    print("Testing PredifySimCLR local updates...")
    print(f"Using {cfg.method_kwargs.timesteps} timesteps")
    print(f"Local updates enabled: {cfg.method_kwargs.use_local_updates}")
    
    # Forward pass on both views
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    
    # Test predictive dynamics
    print("\n" + "="*50)
    print("TESTING PREDICTIVE DYNAMICS")
    print("="*50)
    
    # Use view 2 representations as targets for view 1
    target_reps = out2["representations"]
    target_input = x2
    
    # Set up representations for dynamics
    model.representations = out1["representations"].copy()
    
    # Run predictive dynamics with debug enabled
    pred_errors = model.run_predictive_dynamics(target_reps, target_input, debug=True)
    
    # Analyze results
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    print(f"Number of timesteps: {len(pred_errors)}")
    
    timestep_losses = []
    for t, timestep_errors in enumerate(pred_errors):
        if timestep_errors:
            avg_loss = sum(err.item() if hasattr(err, 'item') else err for err in timestep_errors) / len(timestep_errors)
            timestep_losses.append(avg_loss)
            print(f"Timestep {t}: {len(timestep_errors)} errors, avg loss: {avg_loss:.6f}")
        else:
            print(f"Timestep {t}: No errors")
    
    # Check if losses are different across timesteps
    if len(timestep_losses) > 1:
        if len(set(f"{loss:.10f}" for loss in timestep_losses)) == 1:
            print("\n❌ ISSUE: All timestep losses are identical!")
            print("This suggests local updates are not working properly.")
            return False
        else:
            print("\n✅ SUCCESS: Timestep losses are different!")
            print("Local updates appear to be working correctly.")
            
            # Show the progression
            print("\nLoss progression:")
            for i, loss in enumerate(timestep_losses):
                print(f"  Timestep {i}: {loss:.8f}")
            
            if len(timestep_losses) > 1:
                improvement = timestep_losses[0] - timestep_losses[-1]
                print(f"\nTotal improvement: {improvement:.8f}")
            
            return True
    else:
        print("\n⚠️  WARNING: Only one timestep or no timesteps found")
        return False

if __name__ == "__main__":
    try:
        success = test_local_updates()
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n💥 Test revealed issues that need fixing!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc() 