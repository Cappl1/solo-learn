#!/usr/bin/env python3
"""
Check if EfficientNet pretrained weights are actually being loaded.
Run this script to debug the pretrained weight loading issue.
"""

import torch
import torch.nn as nn
import timm
from timm.models import efficientnet_b0 as timm_efficientnet_b0
import numpy as np


def check_model_weights(model, name="Model"):
    """Analyze model weights to determine if they're pretrained or random."""
    print(f"\n{'='*50}")
    print(f"Analyzing {name}")
    print('='*50)
    
    # Get first conv layer
    if hasattr(model, 'conv_stem'):
        conv = model.conv_stem
        conv_name = "conv_stem"
    elif hasattr(model, 'conv1'):
        conv = model.conv1
        conv_name = "conv1"
    else:
        print("ERROR: Can't find first conv layer!")
        return
    
    # Analyze weights
    with torch.no_grad():
        weights = conv.weight.data
        
        print(f"\n{conv_name} weight statistics:")
        print(f"  Shape: {weights.shape}")
        print(f"  Mean: {weights.mean():.6f}")
        print(f"  Std: {weights.std():.6f}")
        print(f"  Min: {weights.min():.6f}")
        print(f"  Max: {weights.max():.6f}")
        print(f"  Abs mean: {weights.abs().mean():.6f}")
        
        # Check if weights look random (near zero mean, small std)
        is_random = abs(weights.mean()) < 0.001 and weights.std() < 0.1
        print(f"\n  Looks random? {is_random}")
        
        # Sample some weights
        print(f"\n  First 5 weights from channel 0:")
        print(f"  {weights[0, 0, 0, :5].tolist()}")
        
    # Check BatchNorm parameters
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and 'bn1' in name:
            print(f"\nFirst BatchNorm ({name}) statistics:")
            print(f"  Running mean: {module.running_mean[:5].tolist()}")
            print(f"  Running var: {module.running_var[:5].tolist()}")
            print(f"  Weight: {module.weight[:5].tolist()}")
            print(f"  Bias: {module.bias[:5].tolist()}")
            break
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Check classifier
    if hasattr(model, 'classifier'):
        print(f"\nClassifier: {model.classifier}")
        if hasattr(model.classifier, 'in_features'):
            print(f"  in_features: {model.classifier.in_features}")
    
    return weights.mean().item(), weights.std().item()


def test_different_loading_methods():
    """Test different ways of loading EfficientNet to see which loads pretrained weights."""
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT EFFICIENTNET LOADING METHODS")
    print("="*70)
    
    results = {}
    
    # Method 1: Basic pretrained=True
    print("\n\nMethod 1: timm_efficientnet_b0(pretrained=True)")
    model1 = timm_efficientnet_b0(pretrained=True)
    mean1, std1 = check_model_weights(model1, "Method 1")
    results['method1'] = (mean1, std1)
    
    # Method 2: pretrained=True with num_classes=0
    print("\n\nMethod 2: timm_efficientnet_b0(pretrained=True, num_classes=0)")
    model2 = timm_efficientnet_b0(pretrained=True, num_classes=0)
    mean2, std2 = check_model_weights(model2, "Method 2")
    results['method2'] = (mean2, std2)
    
    # Method 3: pretrained=False for comparison
    print("\n\nMethod 3: timm_efficientnet_b0(pretrained=False) [RANDOM INIT]")
    model3 = timm_efficientnet_b0(pretrained=False)
    mean3, std3 = check_model_weights(model3, "Method 3")
    results['method3'] = (mean3, std3)
    
    # Method 4: Using timm.create_model
    print("\n\nMethod 4: timm.create_model('efficientnet_b0', pretrained=True)")
    model4 = timm.create_model('efficientnet_b0', pretrained=True)
    mean4, std4 = check_model_weights(model4, "Method 4")
    results['method4'] = (mean4, std4)
    
    # Method 5: create_model with num_classes=0
    print("\n\nMethod 5: timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)")
    model5 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    mean5, std5 = check_model_weights(model5, "Method 5")
    results['method5'] = (mean5, std5)
    
    # Method 6: Load pretrained then reset classifier
    print("\n\nMethod 6: Load pretrained, then reset_classifier(0)")
    model6 = timm.create_model('efficientnet_b0', pretrained=True)
    model6.reset_classifier(num_classes=0)
    mean6, std6 = check_model_weights(model6, "Method 6")
    results['method6'] = (mean6, std6)
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY - Conv stem weight statistics:")
    print("="*70)
    print(f"{'Method':<40} {'Mean':<12} {'Std':<12} {'Status'}")
    print("-"*70)
    
    for method, (mean, std) in results.items():
        is_random = abs(mean) < 0.001 and std < 0.1
        status = "RANDOM!" if is_random else "PRETRAINED"
        print(f"{method:<40} {mean:< 12.6f} {std:< 12.6f} {status}")
    
    # Compare specific weights between models
    print("\n\nWeight comparison (first 3 values of conv_stem):")
    print("-"*70)
    w1 = model1.conv_stem.weight[0, 0, 0, :3]
    w2 = model2.conv_stem.weight[0, 0, 0, :3]
    w3 = model3.conv_stem.weight[0, 0, 0, :3]
    
    print(f"Pretrained (method 1): {w1.tolist()}")
    print(f"Pretrained + num_classes=0 (method 2): {w2.tolist()}")
    print(f"Random (method 3): {w3.tolist()}")
    print(f"\nAre method 1 & 2 identical? {torch.allclose(w1, w2)}")
    
    # Check your specific use case
    print("\n\n" + "="*70)
    print("YOUR SPECIFIC USE CASE TEST")
    print("="*70)
    
    print("\nYour code: timm_efficientnet_b0(pretrained=True, num_classes=0)")
    your_model = timm_efficientnet_b0(pretrained=True, num_classes=0)
    your_mean = your_model.conv_stem.weight.mean().item()
    print(f"Your conv_stem mean: {your_mean:.6f}")
    
    if abs(your_mean) < 0.001:
        print("\n⚠️  WARNING: Your model weights look RANDOM, not pretrained!")
        print("   The combination of pretrained=True + num_classes=0 might be broken.")
        print("\n✅ SOLUTION: Use Method 6 - Load pretrained first, then reset classifier")
    else:
        print("\n✅ GOOD: Your model weights appear to be pretrained!")


if __name__ == "__main__":
    # Check TIMM version
    print(f"TIMM version: {timm.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run the tests
    test_different_loading_methods()