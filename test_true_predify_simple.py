#!/usr/bin/env python3
"""
Simple test script for True Predify implementation that uses existing config.
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from solo.methods.predify_simclr import PredifySimCLR

def test_true_predify_with_config():
    """Test true predify using the existing config file"""
    print("Testing True Predify with existing config...")
    
    # Load the existing config
    with hydra.initialize(config_path="scripts/pretrain/core50"):
        cfg = hydra.compose(config_name="predify_simclr")
    
    # Allow dynamic config modification
    OmegaConf.set_struct(cfg, False)
    
    # Modify for testing
    cfg.method_kwargs.use_true_predify = True  # Enable true predify
    cfg.method_kwargs.enable_pcoder_grads = False  # Disable for simpler testing
    cfg.method_kwargs.pred_loss_weight = 0.1  # Reduce for stability
    cfg.accelerator = "cpu"  # Use CPU for testing
    cfg.devices = 1
    cfg.data.num_workers = 0  # Avoid multiprocessing issues
    cfg.data.num_classes = 50  # Core50 has 50 classes
    cfg.optimizer.batch_size = 2  # Small batch for testing
    
    print("1. Creating model...")
    model = PredifySimCLR(cfg)
    model.train()
    
    # Test input
    batch_size = 2
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)
    
    print("2. Testing standard forward...")
    model.use_true_predify = False
    out_standard = model(x1)
    print(f"   Standard output: feats={out_standard['feats'].shape}, z={out_standard['z'].shape}")
    
    print("3. Testing true predify forward...")
    model.use_true_predify = True
    out_true = model(x1)
    print(f"   True predify output: feats={out_true['feats'].shape}, z={out_true['z'].shape}")
    print(f"   Has evolved representations: {'evolved_representations' in out_true}")
    
    print("4. Testing training step...")
    indexes = torch.arange(batch_size)
    targets = torch.randint(0, 50, (batch_size,))  # Core50 has 50 classes
    batch = (indexes, [x1, x2], targets)
    
    loss = model.training_step(batch, 0)
    print(f"   Training loss: {loss.item():.4f}")
    
    print("5. Testing coefficient constraints...")
    for i in range(len(model.beta)):
        if i < len(model.lambda_):
            sum_coef = model.beta[i] + model.lambda_[i]
            print(f"   Layer {i}: beta + lambda = {sum_coef:.3f} (≤ 1.0: {sum_coef <= 1.0})")
    
    print("\n✅ True Predify test completed successfully!")

if __name__ == "__main__":
    try:
        test_true_predify_with_config()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 