#!/usr/bin/env python3

"""
Test script for PredifyMoCo implementation.
"""

import torch
import omegaconf
from solo.methods.predify_moco import PredifyMoCo

def test_predify_moco():
    print("Testing PredifyMoCo implementation...")
    
    # Create a mock configuration with all required parameters
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
        model.eval()
        
        print(f"Model created successfully!")
        print(f"Features dimension: {model.features_dim}")
        print(f"Number of PCoders: 4")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            # Test query forward
            model.training = True  # Set to training to activate hooks
            out = model(x)
            
            print(f"Query output shapes:")
            print(f"  feats: {out['feats'].shape}")
            print(f"  z: {out['z'].shape}")
            print(f"  p: {out['p'].shape}")
            if out['logits'] is not None:
                print(f"  logits: {out['logits'].shape}")
            
            # Test momentum forward
            momentum_out = model.momentum_forward(x)
            print(f"Momentum output shapes:")
            print(f"  feats: {momentum_out['feats'].shape}")
            print(f"  z: {momentum_out['z'].shape}")
            
            print(f"Query representations collected: {len(model.query_reps)}")
            print(f"Momentum representations collected: {len(model.momentum_reps)}")
            
            # Test predictive dynamics if we have representations
            if len(model.query_reps) == 5 and len(model.momentum_reps) == 5:
                print("\nTesting predictive dynamics...")
                pred_errors = model.run_predictive_dynamics(model.timesteps)
                print(f"Predictive errors for {model.timesteps} timesteps:")
                for t, errors in enumerate(pred_errors):
                    print(f"  Timestep {t}: {len(errors)} PCoder errors")
                    for i, error in enumerate(errors):
                        print(f"    PCoder {i+1}: {error.item():.6f}")
            else:
                print("Warning: Not all representations collected")
                print(f"Query reps keys: {list(model.query_reps.keys())}")
                print(f"Momentum reps keys: {list(model.momentum_reps.keys())}")
        
        print("\n✅ PredifyMoCo test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ PredifyMoCo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_predify_moco() 