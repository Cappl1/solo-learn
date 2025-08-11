#!/bin/bash
# Quick manual fix for data path issue

echo "🔍 Quick fix for data path issue..."

# Step 1: Find Core50 file
echo "Step 1: Looking for Core50 file..."
CORE50_FILE=""

# Check common locations
if [ -f "/home/brothen/core50_arr.h5" ]; then
    CORE50_FILE="/home/brothen/core50_arr.h5"
    echo "✅ Found: $CORE50_FILE"
elif [ -f "/home/brothen/core50.h5" ]; then
    CORE50_FILE="/home/brothen/core50.h5"
    echo "✅ Found: $CORE50_FILE"
else
    echo "❌ Core50 file not found in common locations"
    echo "🔍 Searching for Core50 files..."
    find /home/brothen -name "*core50*.h5" -type f 2>/dev/null | head -5
    
    # Ask user to specify path
    echo "Please run: ls -la /home/brothen/*core50*"
    echo "And then manually set the correct path below"
fi

if [ ! -z "$CORE50_FILE" ]; then
    echo "✅ Using Core50 file: $CORE50_FILE"
    
    # Step 2: Update config files
    echo "Step 2: Updating config files..."
    
    for CONFIG_FILE in "configs/difficulty_eval_complete.yaml" "configs/difficulty_eval_fixed_ep00.yaml"; do
        if [ -f "$CONFIG_FILE" ]; then
            echo "Updating $CONFIG_FILE..."
            
            # Create a Python script to update the config
            python3 -c "
import yaml
config_file = '$CONFIG_FILE'
data_path = '$CORE50_FILE'

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['data']['train_path'] = data_path
config['data']['val_path'] = data_path
config['data']['format'] = 'h5'

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f'✅ Updated {config_file}')
"
        else
            echo "⚠️  Config file not found: $CONFIG_FILE"
        fi
    done
    
    # Step 3: Test the fix
    echo "Step 3: Testing the fix..."
    echo "Running quick test..."
    
    python3 -c "
import yaml
from pathlib import Path

config_file = 'configs/difficulty_eval_fixed_ep00.yaml'
if Path(config_file).exists():
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    val_path = config['data']['val_path']
    print(f'Config val_path: {val_path}')
    
    if Path(val_path).exists():
        size_mb = Path(val_path).stat().st_size / (1024 * 1024)
        print(f'✅ File exists, size: {size_mb:.1f} MB')
    else:
        print(f'❌ File does not exist: {val_path}')
else:
    print('❌ Config file not found')
"
    
    echo ""
    echo "✅ Quick fix completed!"
    echo "Now try running your test script again:"
    echo "python fixed_test_batch_difficulty_analysis.py"
    
else
    echo ""
    echo "❌ Could not find Core50 file automatically"
    echo "Manual steps:"
    echo "1. Find your Core50 file: find /home/brothen -name '*core50*' -type f"
    echo "2. Edit configs/difficulty_eval_fixed_ep00.yaml"
    echo "3. Set data.train_path and data.val_path to the correct file path"
fi