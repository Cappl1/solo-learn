#!/usr/bin/env python3
"""
Batch script to train linear classifiers for multiple checkpoints.
Usage: python batch_linear_training.py
"""

import os
import subprocess
import sys
from pathlib import Path
import yaml
from typing import List, Tuple

def get_first_n_checkpoints(checkpoint_dir: str, n: int = 20) -> List[Tuple[int, str]]:
    """
    Get the first n checkpoint files sorted by epoch number.
    Returns list of tuples (epoch_number, checkpoint_path)
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Find all checkpoint files with stp=0 (start of epoch)
    checkpoint_files = []
    for ckpt_file in checkpoint_path.glob("*.ckpt"):
        filename = ckpt_file.name
        if "stp=0.ckpt" in filename:
            # Extract epoch number from filename
            # Format: mocov3-selective-curriculum-jepa-core50-5j35ltq7-ep=X-stp=0.ckpt
            try:
                epoch_part = filename.split("-ep=")[1].split("-stp=")[0]
                epoch_num = int(epoch_part)
                checkpoint_files.append((epoch_num, str(ckpt_file)))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse epoch from filename {filename}: {e}")
                continue
    
    # Sort by epoch number and take first n
    checkpoint_files.sort(key=lambda x: x[0])
    return checkpoint_files[:n]

def create_config_for_checkpoint(base_config_path: str, checkpoint_path: str, epoch_num: int, output_dir: str) -> str:
    """
    Create a modified config file for a specific checkpoint.
    Returns the path to the new config file.
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for this specific checkpoint
    config['pretrained_feature_extractor'] = checkpoint_path
    config['name'] = f"selective_curriculum_mocov3_t60_ep{epoch_num:02d}"
    
    # Create output directory structure that includes epoch info
    checkpoint_output_dir = f"{output_dir}/selective_curriculum_mocov3_t60_ep{epoch_num:02d}"
    config['checkpoint']['dir'] = checkpoint_output_dir
    
    # Create the config file path
    config_dir = Path("configs/linear_batch")
    config_dir.mkdir(exist_ok=True)
    config_file_path = config_dir / f"linear_ep{epoch_num:02d}.yaml"
    
    # Save modified config
    with open(config_file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_file_path)

def train_linear_classifier(config_path: str, epoch_num: int) -> bool:
    """
    Train a linear classifier using the specified config.
    Returns True if successful, False otherwise.
    """
    cmd = [
        sys.executable, "main_linear.py",
        "--config-path", str(Path(config_path).parent),
        "--config-name", Path(config_path).stem
    ]
    
    print(f"\n{'='*60}")
    print(f"Training linear classifier for epoch {epoch_num}")
    print(f"Config: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Successfully trained classifier for epoch {epoch_num}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train classifier for epoch {epoch_num}: {e}")
        return False

def main():
    # Configuration
    checkpoint_dir = "/home/brothen/solo-learn/trained_models/predify_simclr/a8ylecmc"
    base_config_path = "scripts/linear_probe/core50/mocov3_batch_base.yaml"  # Use the existing config as base
    output_base_dir = "trained_models/linear/simclr_baseline"
    n_checkpoints = 20
    
    # Validate inputs
    if not Path(base_config_path).exists():
        print(f"Error: Base config file not found at {base_config_path}")
        return 1
    
    try:
        checkpoints = get_first_n_checkpoints(checkpoint_dir, n_checkpoints)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if len(checkpoints) == 0:
        print("Error: No valid checkpoints found")
        return 1
    
    print(f"Found {len(checkpoints)} checkpoints to process:")
    for epoch_num, ckpt_path in checkpoints:
        print(f"  Epoch {epoch_num:2d}: {Path(ckpt_path).name}")
    
    # Create output directory
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each checkpoint
    successful_trainings = 0
    failed_trainings = 0
    
    for epoch_num, checkpoint_path in checkpoints:
        print(f"\nProcessing checkpoint for epoch {epoch_num}...")
        
        # Create config for this checkpoint
        try:
            config_path = create_config_for_checkpoint(
                base_config_path, checkpoint_path, epoch_num, output_base_dir
            )
            print(f"Created config: {config_path}")
        except Exception as e:
            print(f"❌ Failed to create config for epoch {epoch_num}: {e}")
            failed_trainings += 1
            continue
        
        # Train the classifier
        if train_linear_classifier(config_path, epoch_num):
            successful_trainings += 1
        else:
            failed_trainings += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total checkpoints processed: {len(checkpoints)}")
    print(f"Successful trainings: {successful_trainings}")
    print(f"Failed trainings: {failed_trainings}")
    print(f"\nTrained classifiers saved in: {output_base_dir}")
    print(f"Each classifier is in a subdirectory named: selective_curriculum_mocov3_t60_ep{XX}")
    print(f"Where XX is the zero-padded epoch number (00-19)")
    
    return 0 if failed_trainings == 0 else 1

if __name__ == "__main__":
    exit(main()) 