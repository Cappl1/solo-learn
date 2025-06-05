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

from typing import Dict, List, Sequence, Union, Tuple
import numpy as np
import cv2

import torch


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)


def compute_shannon_entropy(img: np.ndarray) -> float:
    """Compute Shannon entropy of an image.
    
    Args:
        img (np.ndarray): Input image in RGB format.
        
    Returns:
        float: Shannon entropy value.
    """
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img
    
    # Compute histogram
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    
    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return float(entropy)


def compute_edge_density(img: np.ndarray) -> float:
    """Compute edge density of an image using Sobel filter.
    
    Args:
        img (np.ndarray): Input image in RGB format.
        
    Returns:
        float: Edge density value.
    """
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img
    
    # Apply Sobel filter
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Compute mean gradient magnitude (edge density)
    edge_density = np.mean(magnitude)
    
    return float(edge_density)


def compute_motion_magnitude(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute motion magnitude between two frames.
    
    Args:
        img1 (np.ndarray): First frame in RGB format.
        img2 (np.ndarray): Second frame in RGB format.
        
    Returns:
        float: Motion magnitude value.
    """
    # Convert to grayscale if RGB
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1
        gray2 = img2
    
    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Compute magnitude of flow vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Compute mean magnitude
    motion_magnitude = np.mean(magnitude)
    
    return float(motion_magnitude)


def compute_structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity index (SSIM) between two frames.
    
    Args:
        img1 (np.ndarray): First frame in RGB format.
        img2 (np.ndarray): Second frame in RGB format.
        
    Returns:
        float: SSIM value.
    """
    # Convert to grayscale if RGB
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1
        gray2 = img2
    
    # Compute SSIM
    ssim = cv2.compareSSIM(gray1, gray2)
    
    return float(ssim)


def compute_color_histogram_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute color histogram distance between two frames.
    
    Args:
        img1 (np.ndarray): First frame in RGB format.
        img2 (np.ndarray): Second frame in RGB format.
        
    Returns:
        float: Histogram distance value.
    """
    # Compute histograms for each channel
    hist1 = []
    hist2 = []
    
    for i in range(3):  # RGB channels
        hist1.append(cv2.calcHist([img1], [i], None, [64], [0, 256]))
        hist2.append(cv2.calcHist([img2], [i], None, [64], [0, 256]))
    
    # Normalize histograms
    for i in range(3):
        cv2.normalize(hist1[i], hist1[i], 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2[i], hist2[i], 0, 1, cv2.NORM_MINMAX)
    
    # Compute histogram distance (Bhattacharyya distance)
    dist = 0
    for i in range(3):
        dist += cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_BHATTACHARYYA)
    
    # Average over channels
    dist /= 3.0
    
    return float(dist)