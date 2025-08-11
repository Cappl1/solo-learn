#!/usr/bin/env python3
"""
Helper script to analyze results from batch linear classifier training.
Usage: python analyze_linear_results.py
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import re

def find_trained_classifiers(base_dir: str) -> List[Dict]:
    """
    Find all trained classifiers in the output directory.
    Returns list of classifier info dictionaries.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Base directory does not exist: {base_dir}")
        return []
    
    classifiers = []
    
    # Look for subdirectories matching the pattern
    pattern = re.compile(r'selective_curriculum_mocov3_t60_ep(\d+)')
    
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            match = pattern.match(subdir.name)
            if match:
                epoch_num = int(match.group(1))
                
                classifier_info = {
                    'epoch': epoch_num,
                    'directory': str(subdir),
                    'name': subdir.name,
                    'checkpoint_files': [],
                    'wandb_logs': None,
                    'best_checkpoint': None
                }
                
                # Find checkpoint files
                for ckpt_file in subdir.rglob("*.ckpt"):
                    classifier_info['checkpoint_files'].append(str(ckpt_file))
                
                # Find the best checkpoint (usually the one with best validation score)
                best_ckpt = None
                for ckpt_file in subdir.rglob("*best*.ckpt"):
                    best_ckpt = str(ckpt_file)
                    break
                
                if not best_ckpt and classifier_info['checkpoint_files']:
                    # If no "best" checkpoint, use the last one (assuming it's the final epoch)
                    best_ckpt = classifier_info['checkpoint_files'][-1]
                
                classifier_info['best_checkpoint'] = best_ckpt
                
                # Look for wandb logs or other metrics
                for log_dir in subdir.rglob("wandb"):
                    if log_dir.is_dir():
                        classifier_info['wandb_logs'] = str(log_dir)
                        break
                
                classifiers.append(classifier_info)
    
    # Sort by epoch number
    classifiers.sort(key=lambda x: x['epoch'])
    return classifiers

def extract_metrics_from_wandb(wandb_dir: str) -> Optional[Dict]:
    """
    Extract metrics from wandb logs if available.
    This is a placeholder - actual implementation would depend on wandb log format.
    """
    # This would need to be implemented based on your specific wandb setup
    # For now, just return None
    return None

def create_mapping_file(classifiers: List[Dict], output_file: str):
    """
    Create a CSV/JSON file mapping epochs to their trained classifiers.
    """
    mapping_data = []
    
    for classifier in classifiers:
        mapping_entry = {
            'epoch': classifier['epoch'],
            'classifier_name': classifier['name'],
            'classifier_directory': classifier['directory'],
            'best_checkpoint': classifier['best_checkpoint'],
            'num_checkpoints': len(classifier['checkpoint_files']),
            'has_wandb_logs': classifier['wandb_logs'] is not None,
            'source_checkpoint': f"mocov3-selective-curriculum-jepa-core50-5j35ltq7-ep={classifier['epoch']}-stp=0.ckpt"
        }
        mapping_data.append(mapping_entry)
    
    # Save as CSV
    df = pd.DataFrame(mapping_data)
    csv_file = output_file.replace('.json', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved mapping to CSV: {csv_file}")
    
    # Save as JSON for programmatic access
    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Saved mapping to JSON: {output_file}")
    
    return mapping_data

def print_summary(classifiers: List[Dict]):
    """
    Print a summary of the trained classifiers.
    """
    print(f"\n{'='*80}")
    print("TRAINED LINEAR CLASSIFIERS SUMMARY")
    print(f"{'='*80}")
    
    if not classifiers:
        print("No trained classifiers found.")
        return
    
    print(f"Total classifiers found: {len(classifiers)}")
    print(f"Epoch range: {min(c['epoch'] for c in classifiers)} - {max(c['epoch'] for c in classifiers)}")
    
    print(f"\n{'Epoch':<6} {'Name':<40} {'Checkpoints':<12} {'WandB':<6}")
    print("-" * 80)
    
    for classifier in classifiers:
        wandb_status = "✓" if classifier['wandb_logs'] else "✗"
        print(f"{classifier['epoch']:<6} {classifier['name']:<40} {len(classifier['checkpoint_files']):<12} {wandb_status:<6}")
    
    # Check for missing epochs
    all_epochs = set(c['epoch'] for c in classifiers)
    expected_epochs = set(range(0, 20))  # Expecting epochs 0-19
    missing_epochs = expected_epochs - all_epochs
    
    if missing_epochs:
        print(f"\n⚠️  Missing classifiers for epochs: {sorted(missing_epochs)}")
    else:
        print(f"\n✅ All expected classifiers (epochs 0-19) are present!")

def main():
    # Configuration
    base_dir = "trained_models/linear/selective_curriculum_mocov3_t60"
    mapping_file = "linear_classifiers_mapping.json"
    
    print("Analyzing trained linear classifiers...")
    
    # Find all trained classifiers
    classifiers = find_trained_classifiers(base_dir)
    
    if not classifiers:
        print(f"No trained classifiers found in {base_dir}")
        print("Make sure you've run the batch training script first.")
        return 1
    
    # Print summary
    print_summary(classifiers)
    
    # Create mapping file
    mapping_data = create_mapping_file(classifiers, mapping_file)
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES")
    print(f"{'='*80}")
    print("To use a specific classifier:")
    print("1. Load the mapping file to find the checkpoint for a specific epoch")
    print("2. Use the 'best_checkpoint' path for evaluation")
    print("\nExample:")
    print("import json")
    print(f"with open('{mapping_file}', 'r') as f:")
    print("    mapping = json.load(f)")
    print("epoch_5_classifier = next(c for c in mapping if c['epoch'] == 5)")
    print("checkpoint_path = epoch_5_classifier['best_checkpoint']")
    
    return 0

if __name__ == "__main__":
    exit(main()) 