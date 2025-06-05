#!/usr/bin/env python3
"""
Debug script to investigate EfficientNet issues with MoCo v3
Tests the most plausible hypotheses systematically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from solo.backbones import efficientnet_b0, resnet18

def test_feature_extraction_methods():
    """Test different ways of extracting features from EfficientNet"""
    print("="*60)
    print("1. TESTING FEATURE EXTRACTION METHODS")
    print("="*60)
    
    # Create dummy input
    x = torch.randn(4, 3, 224, 224)
    
    # Test 1: Current implementation
    print("\n--- Test 1: Current implementation ---")
    try:
        model1 = efficientnet_b0(method="mocov3")
        with torch.no_grad():
            out1 = model1(x)
        print(f"✓ Current implementation: shape={out1.shape}, mean={out1.mean():.3f}, std={out1.std():.3f}")
        print(f"  Feature range: [{out1.min():.3f}, {out1.max():.3f}]")
    except Exception as e:
        print(f"✗ Current implementation failed: {e}")
    
    # Test 2: Direct timm with num_classes=0
    print("\n--- Test 2: Direct timm with num_classes=0 ---")
    try:
        model2 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        with torch.no_grad():
            out2 = model2(x)
        print(f"✓ Direct timm: shape={out2.shape}, mean={out2.mean():.3f}, std={out2.std():.3f}")
        print(f"  Feature range: [{out2.min():.3f}, {out2.max():.3f}]")
    except Exception as e:
        print(f"✗ Direct timm failed: {e}")
    
    # Test 3: Using forward_features
    print("\n--- Test 3: Using forward_features ---")
    try:
        model3 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1000)
        with torch.no_grad():
            out3 = model3.forward_features(x)
            print(f"Raw forward_features shape: {out3.shape}")
            # Apply global pooling manually
            out3_pooled = F.adaptive_avg_pool2d(out3, 1).flatten(1)
        print(f"✓ forward_features + manual pooling: shape={out3_pooled.shape}, mean={out3_pooled.mean():.3f}, std={out3_pooled.std():.3f}")
        print(f"  Feature range: [{out3_pooled.min():.3f}, {out3_pooled.max():.3f}]")
    except Exception as e:
        print(f"✗ forward_features failed: {e}")
    
    # Test 4: Compare with ResNet
    print("\n--- Test 4: Compare with ResNet baseline ---")
    try:
        resnet = resnet18(method="mocov3")
        with torch.no_grad():
            out_resnet = resnet(x)
        print(f"✓ ResNet baseline: shape={out_resnet.shape}, mean={out_resnet.mean():.3f}, std={out_resnet.std():.3f}")
        print(f"  Feature range: [{out_resnet.min():.3f}, {out_resnet.max():.3f}]")
    except Exception as e:
        print(f"✗ ResNet baseline failed: {e}")

def test_feature_distributions():
    """Test if feature distributions are different between models"""
    print("\n" + "="*60)
    print("2. TESTING FEATURE DISTRIBUTIONS")
    print("="*60)
    
    x = torch.randn(32, 3, 224, 224)  # Larger batch for better statistics
    
    models = {}
    try:
        models['ResNet18'] = resnet18(method="mocov3")
        models['EfficientNet-B0'] = efficientnet_b0(method="mocov3")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return
    
    print("\nFeature distribution comparison:")
    print(f"{'Model':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Sparsity':<10}")
    print("-" * 70)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            feats = model(x)
            
            mean_val = feats.mean().item()
            std_val = feats.std().item()
            min_val = feats.min().item()
            max_val = feats.max().item()
            sparsity = (feats.abs() < 0.01).float().mean().item()  # % of near-zero values
            
            print(f"{name:<20} {mean_val:<8.3f} {std_val:<8.3f} {min_val:<8.3f} {max_val:<8.3f} {sparsity:<10.3f}")

def test_normalization_impact():
    """Test if adding normalization helps EfficientNet features"""
    print("\n" + "="*60)
    print("3. TESTING NORMALIZATION IMPACT")
    print("="*60)
    
    x = torch.randn(8, 3, 224, 224)
    
    try:
        # Original EfficientNet
        model = efficientnet_b0(method="mocov3")
        model.eval()
        
        with torch.no_grad():
            feats_orig = model(x)
            
            # Test different normalizations
            feats_l2 = F.normalize(feats_orig, p=2, dim=1)
            feats_bn = F.batch_norm(feats_orig, 
                                  running_mean=torch.zeros(feats_orig.shape[1]),
                                  running_var=torch.ones(feats_orig.shape[1]),
                                  training=False)
            feats_layer_norm = F.layer_norm(feats_orig, feats_orig.shape[1:])
            
        print("\nNormalization comparison:")
        print(f"{'Method':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 60)
        
        for name, feats in [('Original', feats_orig), 
                           ('L2 Normalized', feats_l2),
                           ('Batch Normalized', feats_bn),
                           ('Layer Normalized', feats_layer_norm)]:
            print(f"{name:<20} {feats.mean():<8.3f} {feats.std():<8.3f} {feats.min():<8.3f} {feats.max():<8.3f}")
            
    except Exception as e:
        print(f"Normalization test failed: {e}")

def test_contrastive_behavior():
    """Test how different features behave in contrastive setting"""
    print("\n" + "="*60)
    print("4. TESTING CONTRASTIVE BEHAVIOR")
    print("="*60)
    
    # Simulate contrastive learning scenario
    batch_size = 16
    x1 = torch.randn(batch_size, 3, 224, 224)  # Augmented view 1
    x2 = torch.randn(batch_size, 3, 224, 224)  # Augmented view 2
    
    models = {}
    try:
        models['ResNet18'] = resnet18(method="mocov3")
        models['EfficientNet-B0'] = efficientnet_b0(method="mocov3")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return
    
    print("\nContrastive similarity analysis:")
    print(f"{'Model':<20} {'Self-Sim':<10} {'Cross-Sim':<10} {'Collapse Risk':<15}")
    print("-" * 65)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            feats1 = F.normalize(model(x1), dim=1)
            feats2 = F.normalize(model(x2), dim=1)
            
            # Self similarity (should be high for same sample, different augs)
            self_sim = (feats1 * feats2).sum(dim=1).mean()
            
            # Cross similarity (should be low between different samples)
            cross_sim = (feats1 @ feats2.t()).fill_diagonal_(0).abs().mean()
            
            # Collapse risk: if all features are too similar
            all_feats = torch.cat([feats1, feats2])
            pairwise_sim = all_feats @ all_feats.t()
            collapse_risk = pairwise_sim[~torch.eye(pairwise_sim.size(0), dtype=bool)].abs().mean()
            
            print(f"{name:<20} {self_sim:<10.3f} {cross_sim:<10.3f} {collapse_risk:<15.3f}")

def test_architecture_internals():
    """Inspect the actual architecture differences"""
    print("\n" + "="*60)
    print("5. TESTING ARCHITECTURE INTERNALS")
    print("="*60)
    
    try:
        # Check what layers exist after feature extraction
        resnet = resnet18(method="mocov3")
        efficientnet = efficientnet_b0(method="mocov3")
        
        print("\n--- ResNet18 final layers ---")
        for name, module in resnet.named_modules():
            if any(key in name.lower() for key in ['avgpool', 'fc', 'bn', 'norm']):
                if name:  # Skip empty names
                    print(f"{name}: {module}")
        
        print("\n--- EfficientNet-B0 final layers ---")
        for name, module in efficientnet.named_modules():
            if any(key in name.lower() for key in ['pool', 'head', 'classifier', 'bn', 'norm', 'drop']):
                if name:  # Skip empty names
                    print(f"{name}: {module}")
                    
        # Check if models have expected attributes
        print(f"\nResNet has fc: {hasattr(resnet, 'fc')}")
        print(f"ResNet fc type: {type(resnet.fc) if hasattr(resnet, 'fc') else 'None'}")
        print(f"EfficientNet has classifier: {hasattr(efficientnet, 'classifier')}")
        print(f"EfficientNet classifier type: {type(efficientnet.classifier) if hasattr(efficientnet, 'classifier') else 'None'}")
        print(f"EfficientNet has num_features: {hasattr(efficientnet, 'num_features')}")
        if hasattr(efficientnet, 'num_features'):
            print(f"EfficientNet num_features: {efficientnet.num_features}")
            
    except Exception as e:
        print(f"Architecture inspection failed: {e}")

def main():
    """Run all debug tests"""
    print("Starting EfficientNet Debug Analysis...")
    print("This will test the most plausible causes of EfficientNet issues with MoCo v3")
    
    # Test each hypothesis
    test_feature_extraction_methods()
    test_feature_distributions()
    test_normalization_impact()
    test_contrastive_behavior()
    test_architecture_internals()
    
    print("\n" + "="*60)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey things to look for:")
    print("1. Feature extraction: Are shapes and ranges reasonable?")
    print("2. Distributions: Are EfficientNet features very different from ResNet?")
    print("3. Normalization: Does normalization help feature quality?")
    print("4. Contrastive: Are features collapsing or behaving poorly?")
    print("5. Architecture: Are there missing/wrong layers?")
    
if __name__ == "__main__":
    main() 