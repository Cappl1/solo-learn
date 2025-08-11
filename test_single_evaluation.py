#!/usr/bin/env python3
"""
Test script to run a single difficulty evaluation for debugging.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_single_evaluation():
    """Test a single evaluation to debug the subprocess issue."""
    
    print("🧪 Testing single evaluation...")
    
    # Use the first available checkpoint pair
    backbone_path = "/home/brothen/solo-learn/trained_models/selective_curriculum_mocov3/t60/mocov3-selective-curriculum-jepa-core50-5j35ltq7-ep=0-stp=0.ckpt"
    linear_path = "/home/brothen/solo-learn/trained_models/linear/selective_curriculum_mocov3_t60/selective_curriculum_mocov3_t60_ep00/linear/x3jfoym7/selective_curriculum_mocov3_t60_ep00-x3jfoym7-ep=last-stp=last.ckpt"
    config_path = "/home/brothen/solo-learn/scripts/linear_probe/core50/difficulty_cur.yaml"
    
    # Check if files exist
    print(f"Checking files:")
    print(f"  Backbone: {Path(backbone_path).exists()} - {backbone_path}")
    print(f"  Linear: {Path(linear_path).exists()} - {linear_path}")
    print(f"  Config: {Path(config_path).exists()} - {config_path}")
    
    if not all([Path(backbone_path).exists(), Path(linear_path).exists(), Path(config_path).exists()]):
        print("❌ Some files don't exist!")
        return
    
    # Test with entropy method
    difficulty_method = "entropy"
    
    cmd = [
        "python", "/home/brothen/solo-learn/evaluate_difficulty.py",
        "--model_path", backbone_path,
        "--linear_path", linear_path,
        "--cfg_path", config_path,
        "--difficulty_method", difficulty_method,
        "--batch_size", "32",  # Smaller batch for testing
        "--num_workers", "2",
        "--device", "cuda",
        "--max_batches", "2"  # Only process 2 batches for testing
    ]
    
    print(f"\nRunning command:")
    print(f"{' '.join(cmd)}")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/brothen/solo-learn'
    
    # Run the command
    try:
        print(f"\n🚀 Starting evaluation...")
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              env=env, cwd="/home/brothen/solo-learn", timeout=180)  # 3 min timeout
        
        print(f"\n📊 Results:")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print(f"\nSTDERR:")
            print(result.stderr)
        
        # Check if results file was created
        results_file = Path("/home/brothen/solo-learn/difficulty_analysis/analysis_results_entropy.json")
        print(f"\nResults file exists: {results_file.exists()}")
        
        if results_file.exists():
            print(f"Results file size: {results_file.stat().st_size} bytes")
            # Show first few lines
            with open(results_file, 'r') as f:
                content = f.read()
                print(f"Results file content (first 200 chars):\n{content[:200]}...")
        
        # List all files in the main directory that might be related
        main_dir = Path("/home/brothen/solo-learn")
        print(f"\nFiles in {main_dir} that might be related:")
        for item in main_dir.iterdir():
            if (item.name.startswith('difficulty') or 
                item.name.endswith('.json') or 
                item.name.endswith('.png') or
                'analysis' in item.name.lower()):
                print(f"  {item.name} ({'file' if item.is_file() else 'directory'})")
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out!")
    except Exception as e:
        print(f"❌ Exception occurred: {e}")

if __name__ == "__main__":
    test_single_evaluation() 