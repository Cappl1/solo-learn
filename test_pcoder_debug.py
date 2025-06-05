#!/usr/bin/env python3
"""
Detailed debugging test for PCoder behavior across timesteps.
"""

import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/home/brothen/solo-learn')
from solo.utils.pcoder import PCoder

def debug_pcoder_behavior():
    """Detailed analysis of PCoder behavior across timesteps"""
    
    print("=== DEBUGGING PCODER BEHAVIOR ===")
    
    # Create PCoder
    pcoder = PCoder(
        nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ),
        has_feedback=False,
        random_init=False
    )
    
    # Small test tensors for easier debugging
    ff_input = torch.randn(1, 4, 8, 8) * 0.1  # Small values for stability
    target = torch.randn(1, 2, 8, 8) * 0.1
    
    print(f"Input shape: {ff_input.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Initial PCoder state:")
    print(f"  rep: {pcoder.rep}")
    print(f"  grd: {pcoder.grd}")
    print(f"  prd: {getattr(pcoder, 'prd', 'None')}")
    
    # Parameters for updates
    ffm = 0.5   # feedforward multiplier  
    erm = 0.1   # error multiplier
    
    timesteps = 4
    
    for t in range(timesteps):
        print(f"\n--- TIMESTEP {t} ---")
        
        print(f"Before forward call:")
        print(f"  rep: {pcoder.rep is not None} (norm: {pcoder.rep.norm().item() if pcoder.rep is not None else 'None'})")
        print(f"  grd: {pcoder.grd is not None} (norm: {pcoder.grd.norm().item() if pcoder.grd is not None else 'None'})")
        
        # Call PCoder
        rep_out, pred_out = pcoder(
            ff=ff_input,
            fb=None,
            target=target,
            build_graph=False,
            ffm=ffm,
            fbm=0.0,
            erm=erm
        )
        
        print(f"After forward call:")
        print(f"  rep: norm {pcoder.rep.norm().item():.6f}")
        print(f"  grd: norm {pcoder.grd.norm().item():.6f}")
        print(f"  prd: norm {pcoder.prd.norm().item():.6f}")
        print(f"  prediction_error: {pcoder.prediction_error.item():.6f}")
        print(f"  returned rep norm: {rep_out.norm().item():.6f}")
        print(f"  returned pred norm: {pred_out.norm().item():.6f}")
        
        # Check if representation actually changed
        if t == 0:
            initial_rep = pcoder.rep.clone()
            print(f"  Stored initial rep for comparison")
        else:
            change = (pcoder.rep - initial_rep).norm().item()
            change_pct = change / (initial_rep.norm().item() + 1e-8) * 100
            print(f"  Total change from initial: {change:.6f} ({change_pct:.3f}%)")
        
        # Manually verify the update equation
        if t > 0:  # After first timestep, we should have grd
            print(f"  Manual verification of update equation:")
            print(f"    ffm * ff: {(ffm * ff_input).norm().item():.6f}")
            print(f"    (1-ffm) * old_rep: {((1-ffm) * rep_before_update).norm().item():.6f}")
            print(f"    erm * grd: {(erm * grd_before_update).norm().item():.6f}")
            expected_rep = ffm * ff_input + (1-ffm) * rep_before_update - erm * grd_before_update
            print(f"    Expected new rep norm: {expected_rep.norm().item():.6f}")
            print(f"    Actual new rep norm: {pcoder.rep.norm().item():.6f}")
            
        # Store state for next iteration comparison
        rep_before_update = pcoder.rep.clone()
        grd_before_update = pcoder.grd.clone()

if __name__ == "__main__":
    debug_pcoder_behavior() 