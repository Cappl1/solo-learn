#!/usr/bin/env python3
"""Test pixel entropy analysis on a small dataset subset."""

import subprocess
import os

def test_pixel_entropy_analysis():
    """Test pixel entropy analysis with the actual evaluation script."""
    
    # Use the same paths as in the working manual script
    model_path = "/home/brothen/solo-learn/trained_models/curriculum_mocov3/mae-exponential-bs3x64/mocov3-curriculum-mae-exponential-core50-bs3x64-np6j8ah8-ep=20-stp=13125.ckpt"
    linear_path = "/home/brothen/solo-learn/trained_models/linear/linear/2bocrcbv/mae_exponential_bs64x3_ep20_categories-2bocrcbv-ep=9-stp=18750.ckpt"
    config_path = "/home/brothen/solo-learn/scripts/linear_probe/core50/difficulty_cur.yaml"
    
    print("🧪 Testing pixel entropy analysis...")
    print(f"Model: {model_path}")
    print(f"Linear: {linear_path}")
    print(f"Config: {config_path}")
    
    # Check if files exist
    for path, name in [(model_path, "Model"), (linear_path, "Linear"), (config_path, "Config")]:
        if not os.path.exists(path):
            print(f"❌ {name} file not found: {path}")
            return False
    
    print("✅ All files found")
    
    # Run pixel entropy analysis with a small number of batches
    cmd = [
        "python", "evaluate_difficulty.py",
        "--model_path", model_path,
        "--linear_path", linear_path,
        "--cfg_path", config_path,
        "--difficulty_method", "pixel_entropy",
        "--batch_size", "32",
        "--num_workers", "2", 
        "--device", "cuda",
        "--max_batches", "3"  # Only process 3 batches for testing
    ]
    
    print("Running pixel entropy analysis...")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/brothen/solo-learn")
        
        if result.returncode == 0:
            print("✅ Pixel entropy analysis completed successfully!")
            
            # Check if results were created
            results_file = "/home/brothen/solo-learn/difficulty_analysis/analysis_results_pixel_entropy.json"
            if os.path.exists(results_file):
                print(f"✅ Results file created: {results_file}")
                
                # Load and check results
                import json
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                print(f"Overall accuracy: {results['overall_accuracy']:.3f}")
                print(f"Difficulty range: [{results['difficulty_stats']['min']:.3f}, {results['difficulty_stats']['max']:.3f}]")
                print(f"Method: {results['method']}")
                
                return True
            else:
                print(f"❌ Results file not found: {results_file}")
                print("stdout:", result.stdout[-500:])
                return False
        else:
            print(f"❌ Analysis failed with return code {result.returncode}")
            print("stderr:", result.stderr[-500:])
            print("stdout:", result.stdout[-500:])
            return False
            
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_pixel_entropy_analysis()
    if success:
        print("🎉 Pixel entropy integration test passed!")
    else:
        print("💥 Pixel entropy integration test failed!") 