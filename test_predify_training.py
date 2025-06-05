#!/usr/bin/env python3

"""
Quick training test for PredifyMoCo implementation.
"""

import torch
import torch.nn.functional as F
import omegaconf
from solo.methods.predify_moco import PredifyMoCo
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/brothen/solo-learn')

from solo.utils.pcoder import PCoder

def test_training_step():
    print("Testing PredifyMoCo training step...")
    
    # Create a mock configuration
    cfg = omegaconf.DictConfig({
        "method": "predify_moco",
        "backbone": {
            "name": "vgg16",
            "kwargs": {}
        },
        "method_kwargs": {
            "proj_hidden_dim": 512,
            "proj_output_dim": 128,
            "pred_hidden_dim": 512,
            "temperature": 0.2,
            "timesteps": 2,  # Use fewer timesteps for testing
            "pred_loss_weight": 1.0,
            "ffm": [0.8, 0.8, 0.8, 0.8],
            "fbm": [0.1, 0.1, 0.1, 0.0],
            "erm": [0.01, 0.01, 0.01, 0.01]
        },
        "momentum": {
            "base_tau": 0.99,
            "final_tau": 1.0
        },
        "data": {
            "num_classes": 10,
            "dataset": "cifar10",
            "num_large_crops": 2,
            "num_small_crops": 0
        },
        "max_epochs": 100,
        "optimizer": {
            "batch_size": 32,
            "lr": 0.001,
            "classifier_lr": 0.001,
            "weight_decay": 1e-4,
            "name": "adam",
            "kwargs": {},
            "exclude_bias_n_norm_wd": False
        },
        "scheduler": {
            "name": "warmup_cosine",
            "warmup_epochs": 10,
            "min_lr": 0.0,
            "warmup_start_lr": 3e-5,
            "lr_decay_steps": None,
            "interval": "step"
        },
        "performance": {
            "disable_channel_last": False
        },
        "knn_eval": {
            "enabled": False,
            "k": 20,
            "distance_func": "euclidean"
        },
        "no_validation": False,
        "accumulate_grad_batches": 1
    })
    
    try:
        # Initialize the model
        print("Initializing PredifyMoCo model...")
        model = PredifyMoCo(cfg)
        model.train()  # Set to training mode
        
        # Create mock batch data
        batch_size = 4
        # Simulate two augmented views
        x1 = torch.randn(batch_size, 3, 32, 32)  # CIFAR size
        x2 = torch.randn(batch_size, 3, 32, 32)
        targets = torch.randint(0, 10, (batch_size,))
        
        # Create batch in expected format: [idx, [X], Y]
        batch_idx = torch.arange(batch_size)
        batch = [batch_idx, [x1, x2], targets]
        
        print(f"Created batch with shapes: {x1.shape}, {x2.shape}")
        print(f"Targets: {targets}")
        
        # Test training step
        print("\nTesting training step...")
        total_loss = model.training_step(batch, 0)
        
        print(f"Training step completed successfully!")
        print(f"Total loss: {total_loss.item():.6f}")
        
        # Check if loss is reasonable (not NaN or infinity)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("❌ Loss is NaN or infinite!")
            return False
        
        if total_loss.item() > 0:
            print("✅ Loss is positive and finite")
        else:
            print("⚠️  Loss is zero or negative, might indicate an issue")
        
        # Test backward pass
        print("\nTesting backward pass...")
        total_loss.backward()
        
        # Check gradients
        grad_norm = 0.0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        grad_norm = grad_norm ** 0.5
        
        print(f"Gradient norm: {grad_norm:.6f}")
        print(f"Parameters with gradients: {param_count}")
        
        if grad_norm > 0:
            print("✅ Gradients computed successfully")
        else:
            print("❌ No gradients computed")
            return False
        
        print("\n✅ PredifyMoCo training test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ PredifyMoCo training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predify_dynamics():
    """Test that simulates the PredifySimCLR dynamics behavior"""
    
    print("Testing PredifySimCLR Dynamics Behavior")
    print("="*60)
    
    # Create multiple PCoders to simulate the hierarchy
    pcoders = {}
    
    # PCoder 1: Layer 1 (64) predicts INPUT IMAGE (3)
    pcoders[1] = PCoder(
        nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ),
        has_feedback=True,
        random_init=False
    )
    
    # PCoder 2: Layer 2 (128) predicts Layer 1 (64)
    pcoders[2] = PCoder(
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=10, stride=2, padding=4),
            nn.ReLU(inplace=True)
        ),
        has_feedback=True,
        random_init=False
    )
    
    # Create dummy data simulating VGG representations
    batch_size = 2
    
    # Simulate representations from two different views
    view1_reps = {
        0: torch.randn(batch_size, 64, 112, 112),   # Layer 1
        1: torch.randn(batch_size, 128, 56, 56),    # Layer 2
    }
    
    view2_reps = {
        0: torch.randn(batch_size, 64, 112, 112),   # Layer 1 (target)
        1: torch.randn(batch_size, 128, 56, 56),    # Layer 2 (target)
    }
    
    input_image = torch.randn(batch_size, 3, 224, 224)  # Target input image
    
    print(f"View 1 representations:")
    for idx, rep in view1_reps.items():
        print(f"  Layer {idx}: {rep.shape}")
    
    print(f"View 2 representations (targets):")
    for idx, rep in view2_reps.items():
        print(f"  Layer {idx}: {rep.shape}")
    
    # Simulate the dynamics
    timesteps = 3
    ffm = [0.4, 0.3]  # feedforward multipliers
    fbm = [0.05, 0.05]  # feedback multipliers  
    erm = [0.001, 0.001]  # error multipliers
    
    print(f"\nRunning {timesteps} timesteps of predictive dynamics...")
    
    all_errors = []
    
    for t in range(timesteps):
        print(f"\n--- Timestep {t} ---")
        timestep_errors = []
        
        # PCoder 1: predicts input image
        pcoder1 = pcoders[1]
        ff_input = view1_reps[0].detach()
        target = input_image.detach()
        
        # Get feedback from PCoder 2 if available
        fb_input = None
        if hasattr(pcoders[2], 'prd') and pcoders[2].prd is not None:
            fb_input = pcoders[2].prd
        
        rep1, pred1 = pcoder1(
            ff=ff_input,
            fb=fb_input,
            target=target,
            build_graph=False,
            ffm=ffm[0],
            fbm=fbm[0] if fb_input is not None else 0.0,
            erm=erm[0]
        )
        
        error1 = pcoder1.prediction_error.item()
        timestep_errors.append(error1)
        print(f"  PCoder1 error: {error1:.8f}")
        print(f"  PCoder1 rep norm: {pcoder1.rep.norm().item():.6f}")
        
        # PCoder 2: predicts layer 1 from view 2
        pcoder2 = pcoders[2]
        ff_input = view1_reps[1].detach()
        target = view2_reps[0].detach()  # Target from view 2
        
        rep2, pred2 = pcoder2(
            ff=ff_input,
            fb=None,  # No higher PCoder for feedback
            target=target,
            build_graph=False,
            ffm=ffm[1],
            fbm=0.0,
            erm=erm[1]
        )
        
        error2 = pcoder2.prediction_error.item()
        timestep_errors.append(error2)
        print(f"  PCoder2 error: {error2:.8f}")
        print(f"  PCoder2 rep norm: {pcoder2.rep.norm().item():.6f}")
        
        # Calculate average error for this timestep
        avg_error = sum(timestep_errors) / len(timestep_errors)
        print(f"  Average timestep error: {avg_error:.8f}")
        
        all_errors.append(avg_error)
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print(f"Timestep errors: {[f'{e:.8f}' for e in all_errors]}")
    
    # Check if errors are different across timesteps
    unique_errors = len(set(f"{e:.10f}" for e in all_errors))
    if unique_errors == 1:
        print("\n❌ ISSUE: All timestep errors are identical!")
        print("This suggests local updates are not working properly.")
        return False
    else:
        print(f"\n✅ SUCCESS: Found {unique_errors} different error values!")
        print("Local updates appear to be working correctly.")
        
        # Show progression
        if len(all_errors) > 1:
            improvement = all_errors[0] - all_errors[-1]
            print(f"Error change from first to last: {improvement:.8f}")
            
            if improvement > 0:
                print("✅ Error decreased over timesteps (good!)")
            elif improvement < 0:
                print("⚠️ Error increased over timesteps")
            else:
                print("⚠️ No change in error")
        
        return True

if __name__ == "__main__":
    try:
        success = test_predify_dynamics()
        if success:
            print("\n🎉 Test completed successfully!")
            print("The PredifySimCLR local updates fix is working!")
        else:
            print("\n💥 Test revealed issues that need fixing!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc() 