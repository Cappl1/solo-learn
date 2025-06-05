from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .temporal_core50 import TemporalCore50
from .temporal_transform import prepare_temporal_transforms
from solo.data.pretrain_dataloader import prepare_dataloader


class TemporalDataModule(pl.LightningDataModule):
    """Data module for temporal datasets.
    
    This handles loading and preparing datasets with temporal pairs for self-supervised learning.
    """
    
    def __init__(
        self,
        dataset: str,
        data_path: str,
        batch_size: int,
        num_workers: int = 4,
        data_fraction: float = -1.0,
        transforms_cfg = None,
        extra_dataset_kwargs: Dict = None,
        num_large_crops: int = 1,
        num_small_crops: int = 0,
        val_path: str = None,
    ):
        """Initialize the data module.
        
        Args:
            dataset (str): Dataset name ('temporal_core50').
            data_path (str): Path to the dataset.
            batch_size (int): Batch size.
            num_workers (int): Number of workers for data loading.
            data_fraction (float): Fraction of data to use (-1.0 = all).
            transforms_cfg: Configuration for transforms.
            extra_dataset_kwargs (Dict): Extra arguments for dataset.
            num_large_crops (int): Number of large crops.
            num_small_crops (int): Number of small crops.
            val_path (str): Path to validation dataset (if different from training).
        """
        super().__init__()
        
        self.dataset = dataset
        self.data_path = data_path
        self.val_path = val_path if val_path else data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_fraction = data_fraction
        self.transforms_cfg = transforms_cfg
        self.extra_dataset_kwargs = extra_dataset_kwargs or {}
        self.num_large_crops = num_large_crops
        self.num_small_crops = num_small_crops
        
        # Create the transforms
        self.transform = prepare_temporal_transforms(
            dataset,
            transforms_cfg,
            num_large_crops=num_large_crops,
            num_small_crops=num_small_crops,
        )
        
        # Create simpler validation transform
        import torchvision.transforms as T
        from .temporal_transform import TemporalPairTransform
        
        # Basic transform for validation without heavy augmentations
        val_transform_base = T.Compose([
            T.Resize((transforms_cfg.crop_size, transforms_cfg.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = TemporalPairTransform(val_transform_base)
        
    def setup(self, stage=None):
        """Set up the datasets.
        
        Args:
            stage (Optional[str]): Stage ('fit', 'validate', 'test').
        """
        if stage == 'fit' or stage is None:
            if self.dataset == 'temporal_core50':
                # Create Core50 dataset with temporal pairs
                self.train_dataset = TemporalCore50(
                    h5_path=self.data_path,
                    transform=self.transform,
                    time_window=self.extra_dataset_kwargs.get('time_window', 15),
                    backgrounds=self.extra_dataset_kwargs.get('backgrounds', None),
                )
                
                # Create validation dataset with validation backgrounds if provided
                if 'val_backgrounds' in self.extra_dataset_kwargs:
                    self.val_dataset = TemporalCore50(
                        h5_path=self.val_path,
                        transform=self.val_transform,
                        time_window=self.extra_dataset_kwargs.get('time_window', 15),
                        backgrounds=self.extra_dataset_kwargs.get('val_backgrounds', None),
                    )
                else:
                    # Use small subset of train data for validation if specific val backgrounds not provided
                    from torch.utils.data import Subset
                    import numpy as np
                    
                    # Use 10% of data for validation
                    dataset_size = len(self.train_dataset)
                    val_size = int(0.1 * dataset_size)
                    indices = list(range(dataset_size))
                    np.random.shuffle(indices)
                    self.val_dataset = Subset(
                        TemporalCore50(
                            h5_path=self.data_path,
                            transform=self.val_transform,
                            time_window=self.extra_dataset_kwargs.get('time_window', 15),
                            backgrounds=self.extra_dataset_kwargs.get('backgrounds', None),
                        ),
                        indices[:val_size]
                    )
    
    def train_dataloader(self):
        """Create training dataloader.
        
        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
    def val_dataloader(self):
        """Create validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader.
        """
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        return None