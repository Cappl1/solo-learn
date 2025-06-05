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

import torch.nn as nn
from .vgg import vgg16 as default_vgg16


def _fix_vgg_for_ssl(model):
    """Fix VGG for self-supervised learning by removing classifier."""
    # For standard VGG16, the features after the avgpool layer, when flattened,
    # are 512 * 7 * 7 = 25088 dimensional. This is the input to the original classifier.
    model.num_features = 512 * 7 * 7  # Correct feature dimension

    if hasattr(model, 'classifier'):
        # Replace the original fully-connected classifier with an Identity layer
        # to get the 25088-dimensional features as output from the backbone.
        model.classifier = nn.Identity()
    
    return model

def vgg16(method, *args, **kwargs):
    """VGG16 backbone for self-supervised learning."""
    # Set default pretrained=True for better performance
    kwargs.setdefault('pretrained', True)
    model = default_vgg16(*args, **kwargs)
    return _fix_vgg_for_ssl(model)

__all__ = ["vgg16"] 