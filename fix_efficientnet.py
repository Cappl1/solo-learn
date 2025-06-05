#!/usr/bin/env python3
"""
Proposed fixes for EfficientNet issues with MoCo v3
This implements the most promising solutions based on our analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from solo.backbones import register_model

# Fix 1: Improved EfficientNet implementation with proper feature extraction
@register_model
def efficientnet_b0_fixed(method, **kwargs):
    """Fixed EfficientNet-B0 implementation for better contrastive learning"""
    pretrained = kwargs.pop("pretrained", True)
    
    # Create model with num_classes=0 to remove classification head
    model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0, global_pool='avg', **kwargs)
    
    # Ensure we have the correct feature dimension
    model.num_features = 1280  # EfficientNet-B0 output dimension
    
    # Add feature normalization layer to match ResNet behavior
    # This helps with feature distribution and scale
    class EfficientNetWithNorm(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.feature_norm = nn.BatchNorm1d(1280, affine=False)  # Non-learnable normalization
            self.num_features = 1280
            
        def forward(self, x):
            # Extract features using the backbone
            feats = self.backbone(x)
            
            # Apply normalization to stabilize features
            feats = self.feature_norm(feats)
            
            return feats
    
    return EfficientNetWithNorm(model)

# Fix 2: Alternative implementation using forward_features
@register_model 
def efficientnet_b0_forward_features(method, **kwargs):
    """EfficientNet-B0 using forward_features method"""
    pretrained = kwargs.pop("pretrained", True)
    
    # Create model with original head intact
    model = timm.create_model('efficientnet_b0', pretrained=pretrained, **kwargs)
    
    class EfficientNetFeatures(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.num_features = 1280
            
        def forward(self, x):
            # Use forward_features to get features before head
            feats = self.backbone.forward_features(x)
            
            # Apply global pooling manually
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            
            return feats
    
    return EfficientNetFeatures(model)

# Fix 3: Direct modification of the existing base.py 
def patch_base_method_efficientnet_handling():
    """
    This function contains the modifications to make to base.py
    to fix EfficientNet handling
    """
    
    modifications = """
    # Modify the BaseMethod.__init__ method around line 210-220
    
    # OLD CODE:
    elif "efficientnet" in self.backbone_name:
        self.features_dim: int = self.backbone.num_features
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
    
    # NEW CODE:
    elif "efficientnet" in self.backbone_name:
        # Ensure we have the correct feature dimension
        if hasattr(self.backbone, 'num_features'):
            self.features_dim: int = self.backbone.num_features
        else:
            # Fallback to known dimensions
            if 'b0' in self.backbone_name:
                self.features_dim = 1280
            else:
                # Auto-detect features
                with torch.no_grad():
                    dummy_input = torch.zeros(1, 3, 224, 224)
                    dummy_output = self.backbone(dummy_input)
                    self.features_dim = dummy_output.shape[1]
        
        # Remove classifier and add normalization
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
            
        # Add feature normalization for stability
        self.efficientnet_norm = nn.BatchNorm1d(self.features_dim, affine=False)
    
    # Then modify the forward method around line 470
    
    # OLD CODE:
    def forward(self, X) -> Dict:
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        out = {"feats": feats}
        if not self.cfg.no_validation:
            logits = self.classifier(feats.detach())
            out.update({"logits": logits})
        return out
    
    # NEW CODE:
    def forward(self, X) -> Dict:
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        
        # Apply normalization for EfficientNet
        if "efficientnet" in self.backbone_name and hasattr(self, 'efficientnet_norm'):
            feats = self.efficientnet_norm(feats)
        
        out = {"feats": feats}
        if not self.cfg.no_validation:
            logits = self.classifier(feats.detach())
            out.update({"logits": logits})
        return out
    
    # ALSO modify BaseMomentumMethod.__init__ around line 695
    
    # OLD CODE:
    elif "efficientnet" in self.backbone_name:
        if hasattr(self.momentum_backbone, 'classifier'):
            self.momentum_backbone.classifier = nn.Identity()
    
    # NEW CODE:
    elif "efficientnet" in self.backbone_name:
        if hasattr(self.momentum_backbone, 'classifier'):
            self.momentum_backbone.classifier = nn.Identity()
        # Add same normalization to momentum backbone
        if hasattr(self, 'efficientnet_norm'):
            self.momentum_efficientnet_norm = nn.BatchNorm1d(self.features_dim, affine=False)
            # Copy parameters from main model
            if hasattr(self.efficientnet_norm, 'running_mean'):
                self.momentum_efficientnet_norm.running_mean = self.efficientnet_norm.running_mean.clone()
                self.momentum_efficientnet_norm.running_var = self.efficientnet_norm.running_var.clone()
    
    # And modify momentum_forward around line 750
    
    # OLD CODE:
    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.momentum_backbone(X)
        return {"feats": feats}
    
    # NEW CODE:
    @torch.no_grad()  
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.momentum_backbone(X)
        
        # Apply normalization for EfficientNet momentum backbone
        if "efficientnet" in self.backbone_name and hasattr(self, 'momentum_efficientnet_norm'):
            feats = self.momentum_efficientnet_norm(feats)
        
        return {"feats": feats}
    """
    
    return modifications

# Fix 4: Configuration adjustments for EfficientNet
def get_efficientnet_config_fixes():
    """Returns recommended config changes for EfficientNet"""
    
    config_fixes = {
        "learning_rate_adjustment": {
            "description": "Lower learning rate for EfficientNet",
            "optimizer": {
                "lr": 0.1,  # Much lower than 1.6 used for ResNet
                "weight_decay": 0.0001
            }
        },
        
        "temperature_adjustment": {
            "description": "Higher temperature for EfficientNet features",
            "method_kwargs": {
                "temperature": 0.2  # Instead of 0.1
            }
        },
        
        "alternative_optimizer": {
            "description": "Try AdamW instead of LARS",
            "optimizer": {
                "name": "adamw",
                "lr": 0.001,
                "weight_decay": 0.05
            }
        },
        
        "longer_warmup": {
            "description": "Longer warmup for EfficientNet",
            "scheduler": {
                "warmup_epochs": 20  # Instead of default
            }
        }
    }
    
    return config_fixes

def test_fixes():
    """Test the proposed fixes"""
    print("Testing proposed EfficientNet fixes...")
    
    # Test Fix 1: Improved implementation
    try:
        model1 = efficientnet_b0_fixed(method="mocov3")
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out1 = model1(x)
        print(f"✓ Fix 1 (BatchNorm): shape={out1.shape}, mean={out1.mean():.3f}, std={out1.std():.3f}")
    except Exception as e:
        print(f"✗ Fix 1 failed: {e}")
    
    # Test Fix 2: forward_features
    try:
        model2 = efficientnet_b0_forward_features(method="mocov3")
        with torch.no_grad():
            out2 = model2(x)
        print(f"✓ Fix 2 (forward_features): shape={out2.shape}, mean={out2.mean():.3f}, std={out2.std():.3f}")
    except Exception as e:
        print(f"✗ Fix 2 failed: {e}")
    
    print("\nConfiguration recommendations:")
    configs = get_efficientnet_config_fixes()
    for name, config in configs.items():
        print(f"- {name}: {config['description']}")

if __name__ == "__main__":
    test_fixes()
    print("\nTo apply fixes:")
    print("1. Run debug_efficientnet.py first to identify the exact issue")
    print("2. Apply the appropriate fix based on the debug results")
    print("3. Use the configuration adjustments as needed") 