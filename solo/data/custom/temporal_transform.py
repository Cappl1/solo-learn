from typing import Callable, List, Tuple, Union, Optional

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class TemporalPairTransform:
    """Custom transform for handling temporal pairs of frames.
    
    This transform applies the same augmentation to both frames in a temporal pair,
    ensuring temporal consistency in the augmentations.
    """
    
    def __init__(self, transform: Callable):
        """Initialize with a base transform.
        
        Args:
            transform (Callable): Base transform to apply to both frames.
        """
        self.transform = transform
        # Create transform pipeline with fixed random parameters
        self.fixed_params_transform = transforms.RandomApply([], p=0.0)  # Placeholder
    
    def __call__(self, image1: Image.Image, image2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the same transformation to both images.
        
        Args:
            image1 (PIL.Image): First image (anchor frame).
            image2 (PIL.Image): Second image (future frame).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image pair.
        """
        # Apply the same transform to both images
        # This ensures temporal consistency in the augmentations
        return self.transform(image1), self.transform(image2)


class TemporalNCropTransform:
    """Apply N augmentations to a temporal pair of images.
    
    This applies a set of transforms N times to generate N different views
    of the same temporal pair, while ensuring each pair is augmented consistently.
    """
    
    def __init__(self, transform: Callable, num_crops: int):
        """Initialize with base transform and number of crops.
        
        Args:
            transform (Callable): Base transform to apply.
            num_crops (int): Number of augmented views to generate.
        """
        self.transform = transform
        self.num_crops = num_crops
    
    def __call__(self, image1: Image.Image, image2: Image.Image) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply N transformations to the image pair.
        
        Args:
            image1 (PIL.Image): First image (anchor frame).
            image2 (PIL.Image): Second image (future frame).
            
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of N transformed image pairs.
        """
        # Apply the transform N times to get N different augmented pairs
        return [self.transform(image1, image2) for _ in range(self.num_crops)]


class TemporalFullTransformPipeline:
    """Full pipeline for generating multiple augmented views of temporal pairs.
    
    This handles both large and small crops for complex augmentation strategies.
    """
    
    def __init__(self, transforms: List[Callable]):
        """Initialize with a list of transforms.
        
        Args:
            transforms (List[Callable]): List of transforms to apply.
        """
        self.transforms = transforms
    
    def __call__(self, image1: Image.Image, image2: Image.Image) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply all transforms to generate multiple views.
        
        Args:
            image1 (PIL.Image): First image (anchor frame).
            image2 (PIL.Image): Second image (future frame).
            
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of all augmented pairs.
        """
        out = []
        for transform in self.transforms:
            out.extend(transform(image1, image2))
        return out


def prepare_temporal_transforms(
    dataset: str, 
    cfg, 
    num_large_crops: int = 1, 
    num_small_crops: int = 0
) -> Callable:
    """Prepare transforms for temporal pairs.
    
    Args:
        dataset (str): Dataset name.
        cfg: Configuration object.
        num_large_crops (int): Number of large crops to create.
        num_small_crops (int): Number of small crops to create.
        
    Returns:
        Callable: Transform for temporal pairs.
    """
    from solo.data.pretrain_dataloader import (
        build_transform_pipeline,
        prepare_n_crop_transform,
        FullTransformPipeline,
    )
    
    # Create standard transform pipelines
    large_crop_transform = build_transform_pipeline(dataset, cfg)
    
    # Wrap in temporal pair transform
    temporal_large_crop_transform = TemporalPairTransform(large_crop_transform)
    transforms = [temporal_large_crop_transform]
    
    # Add small crops if requested
    if num_small_crops > 0:
        # Create small crop config
        small_crop_cfg = cfg.augmentations.copy()
        small_crop_cfg.crop_size = small_crop_cfg.crop_size // 2
        small_crop_cfg.rrc.crop_min_scale = small_crop_cfg.rrc.crop_min_scale / 2
        small_crop_cfg.rrc.crop_max_scale = small_crop_cfg.rrc.crop_max_scale / 2
        
        # Create small crop transform
        small_crop_transform = build_transform_pipeline(dataset, small_crop_cfg)
        temporal_small_crop_transform = TemporalPairTransform(small_crop_transform)
        transforms.append(temporal_small_crop_transform)
    
    # Create N-crop transforms list
    n_crop_transforms = [TemporalNCropTransform(transforms[0], num_large_crops)]

    # Conditionally add small N-crop transforms
    if num_small_crops > 0:
        # Create TemporalNCropTransform for each small crop pipeline found in transforms[1:]
        small_n_crop_transforms = [TemporalNCropTransform(t, num_small_crops) for t in transforms[1:]]
        n_crop_transforms.extend(small_n_crop_transforms)
    
    # Return full pipeline using the correctly constructed list
    return TemporalFullTransformPipeline(n_crop_transforms)