#!/usr/bin/env python3
"""
Simple test script to verify that local updates are working properly in PredifySimCLR.
This bypasses the configuration framework to test the core functionality.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/brothen/solo-learn')

from solo.utils.pcoder import PCoder

def create_simple_pcoder():
    """Create a simple PCoder for testing"""
    return PCoder(
        nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ),
        has_feedback=False,  # Set to False since we're not providing feedback
        random_init=False
    )

def test_pcoder_local_updates():
    """Test that PCoder local updates produce different losses across timesteps"""
    
    print("Testing PCoder local updates...")
    
    # Create a simple PCoder
    pcoder = create_simple_pcoder()
    
    # Create dummy data
    batch_size = 2
    ff_input = torch.randn(batch_size, 64, 32, 32)  # Feedforward input
    target = torch.randn(batch_size, 32, 32, 32)    # Target to predict
    
    print(f"Input shape: {ff_input.shape}")
    print(f"Target shape: {target.shape}")
    
    # Test multiple timesteps
    timesteps = 3
    errors = []
    current_rep = ff_input.clone()
    
    print(f"\nRunning {timesteps} timesteps...")
    
    for t in range(timesteps):
        print(f"\n--- Timestep {t} ---")
        
        # Run PCoder - KEY FIX: Use the same ff_input for all timesteps
        # The PCoder internally maintains its own representation state
        rep, pred = pcoder(
            ff=ff_input.detach(),  # Always use original input
            fb=None,  # No feedback for simplicity
            target=target.detach(),
            build_graph=False,  # No gradients for testing
            ffm=0.3,   # feedforward multiplier
            fbm=0.0,   # no feedback
            erm=0.01   # error multiplier - increased for more visible changes
        )
        
        # Get prediction error
        error = pcoder.prediction_error.item()
        errors.append(error)
        
        print(f"Prediction error: {error:.8f}")
        print(f"PCoder internal rep norm: {pcoder.rep.norm().item():.6f}")
        print(f"Returned rep norm: {rep.norm().item():.6f}")
        
        # The PCoder maintains its own internal state, so we don't need to manually update
        # current_rep - the PCoder does this internally
    
    # Analyze results
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    print(f"Errors across timesteps: {[f'{e:.8f}' for e in errors]}")
    
    # Check if errors are different
    unique_errors = len(set(f"{e:.10f}" for e in errors))
    if unique_errors == 1:
        print("\n❌ ISSUE: All timestep errors are identical!")
        print("This suggests local updates are not working properly.")
        return False
    else:
        print(f"\n✅ SUCCESS: Found {unique_errors} different error values!")
        print("Local updates appear to be working correctly.")
        
        # Show progression
        if len(errors) > 1:
            improvement = errors[0] - errors[-1]
            print(f"Error change from first to last: {improvement:.8f}")
            
            if improvement > 0:
                print("✅ Error decreased over timesteps (good!)")
            elif improvement < 0:
                print("⚠️ Error increased over timesteps")
            else:
                print("⚠️ No change in error")
        
        return True

def test_representation_updates():
    """Test that representations actually change between timesteps"""
    
    print("\n" + "="*60)
    print("TESTING REPRESENTATION UPDATES")
    print("="*60)
    
    # Create test data
    batch_size = 1
    original_rep = torch.randn(batch_size, 64, 16, 16)
    target = torch.randn(batch_size, 32, 16, 16)
    
    pcoder = PCoder(
        nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ),
        has_feedback=False,
        random_init=False
    )
    
    # Test single update step
    print(f"Original representation norm: {original_rep.norm().item():.6f}")
    
    updated_rep, pred = pcoder(
        ff=original_rep,
        fb=None,
        target=target,
        build_graph=False,
        ffm=0.5,   # Stronger update
        fbm=0.0,
        erm=0.01   # Stronger error signal
    )
    
    print(f"Updated representation norm: {updated_rep.norm().item():.6f}")
    
    # Calculate change
    change = (updated_rep - original_rep).norm().item()
    change_pct = change / (original_rep.norm().item() + 1e-8) * 100
    
    print(f"Absolute change: {change:.6f}")
    print(f"Relative change: {change_pct:.3f}%")
    
    if change > 1e-6:
        print("✅ Representation successfully updated!")
        return True
    else:
        print("❌ Representation did not change!")
        return False

if __name__ == "__main__":
    try:
        print("Testing PredifySimCLR Local Updates")
        print("="*60)
        
        # Test 1: PCoder local updates
        success1 = test_pcoder_local_updates()
        
        # Test 2: Representation updates
        success2 = test_representation_updates()
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        if success1 and success2:
            print("🎉 All tests passed! Local updates are working correctly.")
        else:
            print("💥 Some tests failed:")
            if not success1:
                print("  - PCoder local updates failed")
            if not success2:
                print("  - Representation updates failed")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc() 