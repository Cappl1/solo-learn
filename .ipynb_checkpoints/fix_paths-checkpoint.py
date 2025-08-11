#!/usr/bin/env python3
"""
Fix the data path issue in the config.
"""

import yaml
from pathlib import Path

def fix_config_data_paths(config_path: str):
    """Fix data paths in the config file."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fix data paths
    config['data']['train_path'] = '/home/brothen/core50_arr.h5'
    config['data']['val_path'] = '/home/brothen/core50_arr.h5'
    
    # Ensure the data format is correct
    config['data']['format'] = 'h5'
    
    # Print current config for debugging
    print("Current data config:")
    print(f"  train_path: {config['data']['train_path']}")
    print(f"  val_path: {config['data']['val_path']}")
    print(f"  format: {config['data']['format']}")
    print(f"  dataset: {config['data']['dataset']}")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Fixed data paths in {config_path}")

def check_data_file_exists():
    """Check if the data file actually exists."""
    data_path = Path('/home/brothen/core50_arr.h5')
    
    if data_path.exists():
        print(f"✅ Data file exists: {data_path}")
        print(f"  Size: {data_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    else:
        print(f"❌ Data file missing: {data_path}")
        
        # Look for alternatives
        possible_paths = [
            '/home/brothen/core50.h5',
            '/home/brothen/data/core50_arr.h5',
            '/home/brothen/data/core50.h5',
            '/data/core50_arr.h5',
            './core50_arr.h5',
            './data/core50_arr.h5'
        ]
        
        print("Looking for alternative paths:")
        for path in possible_paths:
            if Path(path).exists():
                print(f"✅ Found alternative: {path}")
                return path
            else:
                print(f"❌ Not found: {path}")
        
        return False

def main():
    print("Fixing data path issue...")
    
    # Check if data file exists
    data_status = check_data_file_exists()
    
    if data_status is True:
        # Fix all generated config files
        config_files = [
            'configs/difficulty_eval_complete.yaml',
            'configs/difficulty_eval_fixed_ep00.yaml'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"\nFixing {config_file}:")
                fix_config_data_paths(config_file)
            else:
                print(f"❌ Config file not found: {config_file}")
    
    elif isinstance(data_status, str):
        # Alternative path found
        print(f"\nUsing alternative data path: {data_status}")
        
        config_files = [
            'configs/difficulty_eval_complete.yaml',
            'configs/difficulty_eval_fixed_ep00.yaml'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                config['data']['train_path'] = data_status
                config['data']['val_path'] = data_status
                
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print(f"✅ Updated {config_file} with correct data path")
    
    else:
        print("❌ No valid data file found! Please check your Core50 dataset location.")
        print("Expected location: /home/brothen/core50_arr.h5")

if __name__ == "__main__":
    main()