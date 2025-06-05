# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Adapted from timm https://github.com/rwightman/pytorch-image-models/blob/master/timm/

import torch.nn as nn
import torch.nn.init as init
from timm.models import efficientnet_b0 
from timm.models.registry import register_model
from solo.utils.misc import omegaconf_select


def _fix_efficientnet_bn_init(model):
    """Fix BatchNorm initialization for better random init performance.
    
    EfficientNet is particularly sensitive to BN running statistics when randomly initialized.
    This function resets them to proper initial values.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # Reset running statistics to proper values
            module.running_mean.zero_()
            module.running_var.fill_(1)
            # Ensure proper weight/bias initialization
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


@register_model
def efficientnet_b0(method, **kwargs):
    """Creates an EfficientNet-B0 backbone.
    
    Note: EfficientNet typically requires ImageNet pretraining to perform well.
    This wrapper uses pretrained=True by default. When pretrained=False, it applies
    better initialization to improve random initialization performance.
    
    Args:
        method (str): The method name (e.g., 'mocov3', 'simclr').
        **kwargs: Additional keyword arguments passed to the TIMM model.
    
    Returns:
        torch.nn.Module: EfficientNet-B0 model.
    """
    # Use pretrained weights by default for EfficientNet since it performs 
    # much better with ImageNet pretraining
    pretrained = kwargs.pop("pretrained", True)
    
    # Create the model with no classifier (num_classes=0)
    model = efficientnet_b0(pretrained=pretrained, **kwargs)


    print(f"First conv weight mean: {model.conv_stem.weight.mean():.4f}")
    
    # The classifier should already be Identity() when num_classes=0,
    # but let's ensure it for consistency
    if hasattr(model, 'classifier') and not isinstance(model.classifier, nn.Identity):
        model.classifier = nn.Identity()
    
    # If using random initialization, apply better BN initialization
    if not pretrained:
        _fix_efficientnet_bn_init(model)
    
    return model 