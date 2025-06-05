#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute difficulty metrics for a specific epoch using a checkpoint.
"""

import os
import argparse
from pathlib import Path

import torch
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf

from solo.methods import METHODS
from solo.utils.difficulty_metrics_callbacks import DifficultyMetricsCallback

def main():
    parser = argparse.ArgumentParser(description="Compute difficulty metrics for a specific epoch")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to assign to metrics")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Output directory for metrics (default: from config)")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    cfg = OmegaConf.load(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        cfg.difficulty_metrics.output_dir = args.output_dir
    
    # Create model
    print(f"Creating model of type {cfg.method}")
    model = METHODS[cfg.method](cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    
    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Create difficulty metrics callback
    print("Creating difficulty metrics callback")
    callback_kwargs = {k: v for k, v in cfg.difficulty_metrics.items() 
                      if k not in ["enabled"]}
    
    metrics_callback = DifficultyMetricsCallback(**callback_kwargs)
    
    # Compute metrics for the specified epoch
    print(f"Computing metrics for epoch {args.epoch}")
    metrics_df = metrics_callback.compute_metrics_for_epoch(model, args.epoch)
    
    print(f"Done! Metrics saved to {cfg.difficulty_metrics.output_dir}/metrics_epoch_{args.epoch}.parquet")
    
if __name__ == "__main__":
    main() 