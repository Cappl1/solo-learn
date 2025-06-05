from typing import Dict, Any, Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from solo.data.classification_dataloader import prepare_datasets, prepare_transforms
from solo.utils.knn import WeightedKNNClassifier


class KNNCallback(pl.Callback):
    def __init__(self, cfg: DictConfig, ):
        self.cfg = cfg
        self.train_loader, self.test_loader = None, None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        T_train, T_val = prepare_transforms(self.cfg.dataset)

        # For TemporalCore50, wrap transforms to handle paired images
        if self.cfg.dataset in ["temporal_core50", "temporal_mvimagenet"]:
            # Create wrappers that handle paired images
            original_t_train = T_train
            original_t_val = T_val
            
            # Create wrapper classes that only use the first image
            class SingleImageWrapper:
                def __init__(self, transform):
                    self.transform = transform
                
                def __call__(self, img, paired_img=None):
                    # Only transform the first image, ignore the paired image
                    return self.transform(img)
            
            T_train = SingleImageWrapper(original_t_train)
            T_val = SingleImageWrapper(original_t_val)

        # Extract and prepare dataset_kwargs for special datasets
        dataset_kwargs = {}
        if self.cfg.dataset in ["core50", "temporal_core50"]:
            # For Core50 datasets, pass train/val backgrounds
            train_backgrounds = getattr(self.cfg, "train_backgrounds", None)
            val_backgrounds = getattr(self.cfg, "val_backgrounds", None)
            if train_backgrounds is not None:
                dataset_kwargs["train_backgrounds"] = train_backgrounds
            if val_backgrounds is not None:
                dataset_kwargs["val_backgrounds"] = val_backgrounds
        elif self.cfg.dataset == "temporal_mvimagenet":
            # For temporal MVImageNet, pass all required parameters
            if hasattr(self.cfg, "metadata_path"):
                dataset_kwargs["metadata_path"] = self.cfg.metadata_path
            if hasattr(self.cfg, "time_window"):
                dataset_kwargs["time_window"] = self.cfg.time_window
            if hasattr(self.cfg, "val_split"):
                dataset_kwargs["val_split"] = self.cfg.val_split
            if hasattr(self.cfg, "stratify_by_category"):
                dataset_kwargs["stratify_by_category"] = self.cfg.stratify_by_category
            if hasattr(self.cfg, "random_seed"):
                dataset_kwargs["random_seed"] = self.cfg.random_seed

        train_dataset, val_dataset = prepare_datasets(
            self.cfg.dataset,
            T_train,
            T_val,
            train_data_path=self.cfg.train_path,
            val_data_path=self.cfg.val_path,
            data_format=self.cfg.format,
            **dataset_kwargs
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(train_dataset, shuffle=False)
        )
        self.test_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=DistributedSampler(val_dataset, shuffle=False)
        )

        if isinstance(self.cfg.perform_every_n_batches, float):
            print("Estimated stepping batches", trainer.estimated_stepping_batches)
            self.cfg.perform_every_n_batches = int(trainer.estimated_stepping_batches * self.cfg.perform_every_n_batches / trainer.max_epochs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.cfg.perform_on_validation and trainer.current_epoch >= self.cfg.delay_epochs:
            self._run(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.cfg.perform_on_test:
            self._run(trainer, pl_module)

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        if self.cfg.perform_every_n_batches is not None and batch_idx % self.cfg.perform_every_n_batches == 0 and batch_idx != 0:
            self._run(trainer, pl_module)

    def _run(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.sanity_checking and not trainer.fast_dev_run:
            torch.cuda.empty_cache()
            
            original_training_state = pl_module.training # Store original mode

            # Always use eval mode for feature extraction in KNN
            pl_module.eval()
            # Optional: Add a print statement here if you want to confirm it's being set to eval
            # print("[KNN] Setting model to EVAL mode for feature extraction.")

            # Extract features
            result = self.run(trainer, pl_module)

            torch.cuda.empty_cache()
            
            # Restore original training/eval state
            pl_module.train(original_training_state)
            # Optional: Add a print statement here to confirm restoration
            # if original_training_state:
            #     print("[KNN] Restored model to TRAIN mode.")
            # else:
            #     print("[KNN] Restored model to EVAL mode.")

            for k, value in result.items():
                if hasattr(trainer.logger, 'log_metrics'):
                    trainer.logger.log_metrics({
                        f'knn/{self.cfg.dataset}_{k}_top1': value[0],
                        f'knn/{self.cfg.dataset}_{k}_top5': value[1]
                    }, step=trainer.global_step)
                else:
                    raise ValueError("Please use a logger that supports `log_metrics`")

    @torch.no_grad()
    def extract_features(self, loader: DataLoader, model: pl.LightningModule, mode: str = "train") -> Tuple[
        torch.Tensor, torch.Tensor]:
        bar = tqdm(loader, desc=f'{mode} KNN',
                   total=len(loader)) if self.cfg.verbose and model.local_rank == 0 else loader

        res_X, res_y = [], []
        for batch in bar:
            # Handle both dataset_with_index format (index, X, y) and normal format (X, y)
            if len(batch) == 3:  # dataset_with_index format
                _, X, y = batch
            else:  # normal format
                X, y = batch
                
            X = X.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)

            outs = model(X)
            res_X.append(outs["feats"].detach())
            res_y.append(y.detach())
        res_X = torch.cat(res_X)
        res_y = torch.cat(res_y)
        return res_X, res_y

    def run(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict:
        # extract train and test features
        X_train, y_train = self.extract_features(self.train_loader, pl_module, mode="train")
        X_test, y_test = self.extract_features(self.test_loader, pl_module, mode="test")

        # barrier to make sure all features are extracted
        trainer.strategy.barrier()

        result = {}
        for k in self.cfg.k:
            knn = WeightedKNNClassifier(k=k, T=self.cfg.T, distance_fx=self.cfg.distance_fx)
            knn(X_train, y_train, X_test, y_test)
            val_knn_acc1, val_knn_acc5 = knn.compute()
            result[k] = (val_knn_acc1, val_knn_acc5)
            del knn

        return result
