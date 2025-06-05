#!/usr/bin/env python3

"""
Script to inspect the VGG16 architecture from solo-learn and its output feature dimension.
"""

import torch
from solo.backbones.vgg import vgg16 as solo_vgg16 # Import our wrapped vgg16

def inspect_model():
    print("--- Initializing VGG16 from solo.backbones.vgg ---")
    # The wrapper expects a 'method' argument, even if not used by vgg16 directly.
    # It also expects pretrained to be passed, which our wrapper handles by default.
    model = solo_vgg16(method="inspection_test", pretrained=True)

    print("\n--- VGG16 Model Architecture (from solo-learn wrapper) ---")
    print(model)

    # The num_features should be set by our _fix_vgg_for_ssl wrapper
    print(f"\n--- model.num_features attribute set by wrapper: {model.num_features} ---")

    # Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    # Standard ImageNet size
    dummy_input = torch.randn(1, 3, 224, 224)

    print("\n--- Performing a forward pass with dummy input ---")
    try:
        model.eval() # Set to evaluation mode
        with torch.no_grad():
            output_features = model(dummy_input)
        
        print(f"\n--- Output features shape: {output_features.shape} ---")
        
        if output_features.shape == (1, 25088):
            print("\n✅ SUCCESS: Output feature dimension (1, 25088) matches expected 512 * 7 * 7.")
            print("This confirms the VGG16 backbone, after the features and avgpool layers, and flattening, outputs 25088 features.")
            print("The classifier was correctly replaced with an Identity layer.")
        elif output_features.shape == (1, 512, 7, 7):
            print("\n⚠️  WARNING: Output shape is (1, 512, 7, 7).")
            print("This means the output is from after the avgpool layer but *before* flattening.")
            print("This would be unexpected if the goal is to get a flat feature vector from the backbone directly.")
            print("However, our _fix_vgg_for_ssl sets model.num_features to 25088, implying flattening should occur.")
        else:
            print(f"\n❌ ERROR: Output feature dimension is {output_features.shape}, which is unexpected.")
            print(f"Expected (1, 25088) or potentially (1, 512, 7, 7) if flatten was missed before Identity.")

    except Exception as e:
        print(f"\n❌ ERROR during forward pass or inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model() 