#!/usr/bin/env python3
"""
Fixed configuration and utility functions for difficulty analysis.
"""

import yaml
from pathlib import Path
from omegaconf import OmegaConf
import torch

def create_complete_config(output_path: str = "configs/difficulty_eval_fixed.yaml") -> str:
    """Create a complete config file with all required parameters."""
    
    config_dir = Path(output_path).parent
    config_dir.mkdir(exist_ok=True, parents=True)
    
    # Complete configuration matching the training setup
    complete_config = {
        'defaults': ['_self_'],
        'name': 'difficulty_analysis_selective_curriculum',
        'method': 'selective_curriculum_mocov3',
        
        # Backbone configuration
        'backbone': {
            'name': 'resnet18',
            'kwargs': {}
        },
        
        # CRITICAL: Complete method_kwargs section
        'method_kwargs': {
            # MoCo v3 parameters
            'proj_hidden_dim': 4096,
            'proj_output_dim': 256,
            'pred_hidden_dim': 4096,
            'temperature': 0.1,
            
            # Curriculum-specific parameters
            'curriculum_type': 'jepa',
            'curriculum_reverse': False,
            'curriculum_warmup_epochs': 20,
            'curriculum_weight': 1.0,
            'reconstruction_masking_ratio': 0.75,
            'num_candidates': 8,
            'curriculum_only_for_epochs': 100,
            'selection_epochs': 100,
            
            # Additional parameters that might be needed
            'base_tau_momentum': 0.996,
            'final_tau_momentum': 0.996,
        },
        
        # CRITICAL: Momentum section
        'momentum': {
            'base_tau': 0.996,
            'final_tau': 0.996
        },
        
        # CRITICAL: Complete data section with ALL required fields
        'data': {
            'dataset': 'core50_categories',
            'train_path': '/home/brothen/core50_arr.h5',
            'val_path': '/home/brothen/core50_arr.h5',
            'format': 'h5',
            'num_classes': 10,
            'num_workers': 4,
            
            # CRITICAL: These were missing and causing the first error
            'num_large_crops': 2,
            'num_small_crops': 0,
            
            # Background settings
            'train_backgrounds': ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11'],
            'val_backgrounds': ['s3', 's7', 's10'],
            
            # Dataset-specific parameters
            'dataset_kwargs': {
                'use_categories': True
            },
            
            # Augmentation parameters (might be needed)
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.2,
            'hue': 0.1,
            'color_jitter_prob': 0.8,
            'gray_scale_prob': 0.2,
            'horizontal_flip_prob': 0.5,
            'gaussian_prob': 1.0,
            'solarization_prob': 0.0,
            'crop_size': 224,
            'min_scale': 0.08,
            'max_scale': 1.0,
        },
        
        # CRITICAL: Complete optimizer section
        'optimizer': {
            'name': 'lars',
            'batch_size': 64,
            'lr': 1.6,
            'weight_decay': 1e-6,
            'momentum': 0.9,
            'eta': 0.02,
            'grad_clip_lars': False,
            'exclude_bias_n_norm': False,
            # Add the missing classifier_lr field
            'classifier_lr': 0.3,
        },
        
        # CRITICAL: Complete scheduler section (this was causing the second error)
        'scheduler': {
            'name': 'warmup_cosine',
            'warmup_epochs': 0.01,
            'warmup_start_lr': 0.0,
            'eta_min': 0.0,
            'max_epochs': 100,
            'interval': 'step',
            'frequency': 1,
            'lr_decay_steps': None,
            'min_lr': 0.0,
            'step_size': 30,
            'gamma': 0.1,
        },
        
        # Loss configuration
        'loss': {
            'name': 'mocov3',
            'temperature': 0.1
        },
        
        # Training parameters
        'max_epochs': 100,
        'devices': [0],
        'accelerator': 'gpu',
        'precision': 32,
        'sync_batchnorm': True,
        'strategy': 'auto',
        'enable_checkpointing': False,
        'logger': False,
        'callbacks': None,
        'num_sanity_val_steps': 0,
        'log_every_n_steps': 50,
        'check_val_every_n_epoch': 1,
        'accumulate_grad_batches': 1,
        
        # Additional parameters that might be expected
        'no_validation': False,
        'auto_resume': False,
        'resume_from_checkpoint': None,
        
        # Augmentation settings (empty but might be expected)
        'augmentations': [],
        'transform_kwargs': {},
        
        # Disable features that might cause issues during evaluation
        'auto_lr_find': False,
        'fast_dev_run': False,
        'overfit_batches': 0.0,
        'track_grad_norm': -1,
        'val_check_interval': 1.0,
        'limit_train_batches': 1.0,
        'limit_val_batches': 1.0,
        'limit_test_batches': 1.0,
        'limit_predict_batches': 1.0,
    }
    
    # Save the config
    with open(output_path, 'w') as f:
        yaml.dump(complete_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created complete config: {output_path}")
    return output_path

def extract_config_from_checkpoint(checkpoint_path: str, output_path: str = None) -> str:
    """Extract and save config from a checkpoint file."""
    
    if output_path is None:
        output_path = f"configs/extracted_config_{Path(checkpoint_path).stem}.yaml"
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try to extract config from checkpoint
        config_keys = ['cfg', 'config', 'hyper_parameters', 'hparams']
        extracted_config = None
        
        for key in config_keys:
            if key in checkpoint:
                extracted_config = checkpoint[key]
                print(f"Found config in checkpoint under key: {key}")
                break
        
        if extracted_config is None:
            print(f"No config found in checkpoint {checkpoint_path}")
            print(f"Available keys: {list(checkpoint.keys())}")
            return None
        
        # Convert to dict if it's an OmegaConf object
        if hasattr(extracted_config, '_content'):
            config_dict = OmegaConf.to_yaml(extracted_config)
        else:
            config_dict = extracted_config
        
        # Save extracted config
        config_dir = Path(output_path).parent
        config_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            if isinstance(config_dict, str):
                f.write(config_dict)
            else:
                yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"✅ Extracted config saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error extracting config from {checkpoint_path}: {e}")
        return None

def fix_config_for_evaluation(input_config_path: str, output_config_path: str = None) -> str:
    """Fix a config file to ensure it has all required parameters for evaluation."""
    
    if output_config_path is None:
        output_config_path = input_config_path.replace('.yaml', '_fixed.yaml')
    
    # Load existing config
    with open(input_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure required sections exist
    required_sections = {
        'data': {},
        'scheduler': {},
        'optimizer': {},
        'method_kwargs': {},
        'momentum': {}
    }
    
    for section, defaults in required_sections.items():
        if section not in config:
            config[section] = defaults
    
    # Fix data section
    data_fixes = {
        'num_large_crops': 2,
        'num_small_crops': 0,
        'num_workers': 4,
        'crop_size': 224,
    }
    
    for key, value in data_fixes.items():
        if key not in config['data']:
            config['data'][key] = value
    
    # Fix scheduler section
    scheduler_fixes = {
        'name': 'warmup_cosine',
        'warmup_epochs': 0.01,
        'warmup_start_lr': 0.0,
        'eta_min': 0.0,
        'max_epochs': 100,
        'interval': 'step',
        'frequency': 1,
        'lr_decay_steps': None,
        'min_lr': 0.0,
    }
    
    for key, value in scheduler_fixes.items():
        if key not in config['scheduler']:
            config['scheduler'][key] = value
    
    # Fix momentum section
    momentum_fixes = {
        'base_tau': 0.996,
        'final_tau': 0.996,
    }
    
    for key, value in momentum_fixes.items():
        if key not in config['momentum']:
            config['momentum'][key] = value
    
    # Fix method_kwargs
    method_kwargs_fixes = {
        'proj_hidden_dim': 4096,
        'proj_output_dim': 256,
        'pred_hidden_dim': 4096,
        'temperature': 0.1,
        'curriculum_type': 'jepa',
        'num_candidates': 8,
    }
    
    for key, value in method_kwargs_fixes.items():
        if key not in config['method_kwargs']:
            config['method_kwargs'][key] = value
    
    # Save fixed config
    config_dir = Path(output_config_path).parent
    config_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Fixed config saved to: {output_config_path}")
    return output_config_path

def main():
    """Main function to create or fix configs."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix configuration for difficulty analysis')
    parser.add_argument('--action', choices=['create', 'extract', 'fix'], default='create',
                        help='Action to perform')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to checkpoint (for extract action)')
    parser.add_argument('--input_config', type=str,
                        help='Path to input config (for fix action)')
    parser.add_argument('--output_config', type=str,
                        help='Path to output config')
    
    args = parser.parse_args()
    
    if args.action == 'create':
        output_path = args.output_config or "configs/difficulty_eval_complete.yaml"
        create_complete_config(output_path)
        
    elif args.action == 'extract':
        if not args.checkpoint_path:
            print("Error: --checkpoint_path required for extract action")
            return 1
        extract_config_from_checkpoint(args.checkpoint_path, args.output_config)
        
    elif args.action == 'fix':
        if not args.input_config:
            print("Error: --input_config required for fix action")
            return 1
        fix_config_for_evaluation(args.input_config, args.output_config)
    
    return 0

if __name__ == '__main__':
    exit(main())