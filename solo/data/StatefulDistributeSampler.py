from typing import Union, IO, Optional, Any, Dict
from typing_extensions import Self

import numpy as np
import torch
from lightning.fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import lightning.pytorch as pl


from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)

class DataPrepIterCheck(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.global_step = 0
    
    def setup(self, stage: str) -> None:
        pipelines = []
        for aug_cfg in self.cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(self.cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                )
            )
        transform = FullTransformPipeline(pipelines)

        if self.cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        # Filter dataset_kwargs to only pass appropriate parameters to the dataset constructor
        dataset_kwargs = self.cfg.data.dataset_kwargs.copy() if hasattr(self.cfg.data, 'dataset_kwargs') else {}
        
        # For temporal_core50, remove val_backgrounds from train dataset parameters
        if self.cfg.data.dataset == "temporal_core50" and "val_backgrounds" in dataset_kwargs:
            # Create a copy without val_backgrounds
            train_kwargs = {k: v for k, v in dataset_kwargs.items() if k != "val_backgrounds"}
        else:
            train_kwargs = dataset_kwargs

        self.train_dataset = prepare_datasets(
            self.cfg.data.dataset,
            transform,
            train_data_path=self.cfg.data.train_path,
            data_format=self.cfg.data.format,
            no_labels=self.cfg.data.no_labels,
            data_fraction=self.cfg.data.fraction,
            **train_kwargs
        )

        # For temporal_mvimagenet, don't create val_dataset in setup - we'll create it directly in val_dataloader()
        # using the base MVImageNet class to avoid temporal complexity in validation



    def val_dataloader(self):
        if self.cfg.data.dataset == "custom" and (self.cfg.data.no_labels or self.cfg.data.val_path is None):
            val_loader = None
        elif self.cfg.data.dataset in ["imagenet100", "imagenet", "ego4d"] and self.cfg.data.val_path is None:
            val_loader = None
        elif self.cfg.data.dataset == "temporal_mvimagenet" and self.cfg.data.val_path is not None:
            # For temporal_mvimagenet, use base MVImageNet for validation (single images, no temporal pairing)
            from solo.data.custom.mvimagenet import MVImageNet
            from torchvision import transforms
            
            # Create a basic transform for validation
            val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Create base MVImageNet dataset for validation (no temporal complexity)
            val_dataset = MVImageNet(
                h5_data_dir=self.cfg.data.val_path,
                transform=val_transform,
                split='val',
                train_metadata_path=self.cfg.data.dataset_kwargs.get("train_metadata_path"),
                val_metadata_path=self.cfg.data.dataset_kwargs.get("val_metadata_path")
            )
            
            # Create dataloader
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler
            
            sampler = DistributedSampler(val_dataset, shuffle=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.optimizer.batch_size,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                sampler=sampler,
            )
        elif self.cfg.data.dataset == "temporal_core50" and self.cfg.data.val_path is not None:
            # For temporal_core50, create a validation dataset with the val_backgrounds
            from solo.data.custom.temporal_core50 import TemporalCore50
            from torchvision import transforms
            
            # Create a transform wrapper that can handle paired images
            class PairedImageTransform:
                def __init__(self, transform):
                    self.transform = transform
                
                def __call__(self, img, paired_img=None):
                    # Only transform the first image and ignore the paired image
                    return self.transform(img)
            
            # Create a basic transform for validation
            base_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Wrap it to handle paired images
            val_transform = PairedImageTransform(base_transform)
            
            # Create dataset with validation backgrounds
            # Don't use dataset_with_index for validation - we don't need indices
            val_dataset = TemporalCore50(
                h5_path=self.cfg.data.val_path,
                transform=val_transform,
                time_window=self.cfg.data.dataset_kwargs.get("time_window", 15),
                backgrounds=self.cfg.data.dataset_kwargs.get("val_backgrounds", None)
            )
            
            # Create dataloader
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler
            
            sampler = DistributedSampler(val_dataset, shuffle=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.optimizer.batch_size,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                sampler=sampler,
            )
        else:
            val_loader = prepare_dataloader(
                self.val_dataset,
                batch_size=self.cfg.optimizer.batch_size,
                num_workers=self.cfg.data.num_workers,
            )
        return val_loader

    def train_dataloader(self):
        # if self.cfg.max_epochs == 1:
        sampler = StatefulDistributedSampler(self.train_dataset, batch_size=self.cfg.optimizer.batch_size, seed=self.cfg.seed)
        train_loader = prepare_dataloader(
            self.train_dataset, batch_size=self.cfg.optimizer.batch_size, num_workers=self.cfg.data.num_workers,
            sampler=sampler, shuffle=False
        )
        # else:
        #     train_loader = prepare_dataloader(
        #         self.train_dataset, batch_size=self.cfg.optimizer.batch_size, num_workers=self.cfg.data.num_workers
        #     )
        return train_loader

    def state_dict(self) -> Dict[str, Any]:
        return {"steps": self.trainer.global_step}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        try:
            self.train_dataloader().sampler.set_start_iter(state_dict["steps"])
        except:
            print("cant load")

    # def load_from_checkpoint(
    #     cls,
    #     checkpoint_path: Union[_PATH, IO],
    #     map_location: _MAP_LOCATION_TYPE = None,
    #     hparams_file: Optional[_PATH] = None,
    #     **kwargs: Any,
    # ) -> Self:
    #     return super().load_from_checkpoint(cls, checkpoint_path, map_location, hparams_file, **kwargs)





# Taken there: https://github.com/facebookresearch/vissl/blob/main/vissl/data/data_helper.py#L93
class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size=None, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, shuffle=False, seed=seed)

        self.start_iter = 0
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        indices = np.array(
            list(
                range(
                    (self.rank * self.num_samples), (self.rank + 1) * self.num_samples
                )
            )
        )[shuffling].tolist()

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter

