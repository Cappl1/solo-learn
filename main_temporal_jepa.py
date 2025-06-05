#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import inspect
from pathlib import Path

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from solo.methods import METHODS
from solo.data.custom.temporal_datamodule import TemporalDataModule
from solo.data.StatefulDistributeSampler import DataPrepIterCheck
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.difficulty_metrics_callbacks import DifficultyMetricsCallback
from solo.utils.misc import make_contiguous

def setup_wandb_logger(cfg, wandb_run_id=None):
    """Set up WandB logger with proper error handling.
    
    Args:
        cfg (DictConfig): Hydra configuration
        wandb_run_id (str, optional): WandB run ID for resuming
        
    Returns:
        WandbLogger or None: The configured logger or None if initialization failed
    """
    if not cfg.wandb.enabled:
        return None
    
    try:
        # Import wandb here to handle import errors
        import wandb
        
        # Check if wandb is properly logged in
        if not wandb.api.api_key:
            print("WandB API key not found. Running without WandB logging.")
            return None
        
        # Initialize WandB logger with proper exception handling
        try:
            wandb_logger = WandbLogger(
                name=cfg.name,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                offline=cfg.wandb.offline,
                group=cfg.wandb.group,
                job_type=cfg.wandb.job_type,
                resume="allow" if wandb_run_id else None,
                id=wandb_run_id,
                save_dir=os.path.join(cfg.checkpoint.dir, "wandb"),
            )
            
            # Test the connection with a simple log
            try:
                wandb_logger.experiment.log({"test": 1.0})
                print(f"WandB initialized successfully: {wandb_logger.version}")
                return wandb_logger
            except Exception as e:
                print(f"Warning: Error while testing WandB connection: {e}")
                wandb_logger.experiment.finish()
                return None
                
        except Exception as e:
            print(f"Warning: Failed to initialize WandB: {e}")
            return None
            
    except ImportError:
        print("WandB package not found. Running without WandB logging.")
        return None
    except Exception as e:
        print(f"Unexpected error with WandB: {e}")
        return None


def watch_model_safe(logger, model):
    """Safely watch a model with a logger.
    
    Args:
        logger: The logger instance
        model: The model to watch
    """
    if logger is None:
        return
        
    if isinstance(logger, WandbLogger):
        try:
            logger.watch(model, log="gradients", log_freq=100)
        except Exception as e:
            print(f"Warning: Could not watch model with WandB: {e}")


@hydra.main(config_path="scripts/pretrain", config_name="core50/temporal_jepa")
def main(cfg: DictConfig):
    """Main entry point for Temporal I-JEPA training.
    
    Args:
        cfg (DictConfig): Hydra configuration.
    """
    # Disable OmegaConf struct mode to allow adding default keys
    OmegaConf.set_struct(cfg, False)
    
    # For reproducibility
    seed_everything(cfg.get("seed", 42))
    
    # Create method instance
    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"
    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    
    # Performance optimization: convert to channels last
    if not cfg.performance.get("disable_channel_last", False):
        model = model.to(memory_format=torch.channels_last)

    # Create data module
    data_module = TemporalDataModule(
        dataset=cfg.data.dataset,
        data_path=cfg.data.train_path,
        val_path=cfg.data.get("val_path", None),
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        data_fraction=cfg.data.fraction,
        transforms_cfg=cfg.augmentations,
        extra_dataset_kwargs=cfg.data.get("dataset_kwargs", {}),
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=cfg.data.num_small_crops,
    )
    
    # Checkpoint handling
    ckpt_path, wandb_run_id = None, None
    if cfg.get("auto_resume", {}).get("enabled", False):
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method),
            max_hours=cfg.auto_resume.get("max_hours", 1),
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(f"Resuming from previous checkpoint: '{resume_from_checkpoint}'")
            ckpt_path = resume_from_checkpoint
    elif cfg.get("resume_from_checkpoint", None) is not None:
        ckpt_path = cfg.resume_from_checkpoint
        if hasattr(cfg, "resume_from_checkpoint"):
            del cfg.resume_from_checkpoint

    # Create callbacks
    callbacks = []

    # Checkpointer
    if cfg.get("checkpoint", {}).get("enabled", False):
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, cfg.method),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
            save_last=cfg.checkpoint.save_last,
        )
        callbacks.append(ckpt)

    # Early stopping
    if cfg.get("early_stopping", {}).get("enabled", False):
        from lightning.pytorch.callbacks import EarlyStopping
        es = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
        )
        callbacks.append(es)
    
    # Add difficulty metrics callback
    metrics_callback = DifficultyMetricsCallback(
        dataset_path=cfg.data.train_path,
        eval_frequency=cfg.get("metrics_eval_frequency", 5),  # Every 5 epochs by default
        num_samples=cfg.get("metrics_num_samples", 1000),     # 1000 samples by default
        compute_rank_stability=True,
        save_metrics=True,
        output_dir=os.path.join(cfg.checkpoint.dir, cfg.method, "metrics"),
        dataset_kwargs=cfg.data.get("dataset_kwargs", {}),
    )
    callbacks.append(metrics_callback)

    # WandB logging
    wandb_logger = None
    if cfg.get("wandb", {}).get("enabled", False):
        d = os.environ["WANDB_DIR"] if "WANDB_DIR" in os.environ else "./"
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
            save_dir=d
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
        
        # Add LR monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    # Create PyTorch Lightning trainer
    trainer_kwargs = OmegaConf.to_container(cfg)
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=True)
            if cfg.strategy == "ddp"
            else cfg.strategy,
        }
    )
    
    # Print debug info
    print(f"Using data module: {data_module.__class__.__name__}")
    print(f"Training with batch size: {cfg.optimizer.batch_size}")
    print(f"Augmentations config: {cfg.augmentations}")
    
    trainer = Trainer(**trainer_kwargs)
    
    # Setup data module before passing to trainer
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader() if hasattr(data_module, 'val_dataloader') else None
    
    # Start training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # Workaround for https://github.com/Lightning-AI/lightning/issues/5524
    torch.set_float32_matmul_precision('medium')
    main()