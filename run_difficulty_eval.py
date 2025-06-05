#!/usr/bin/env python3
"""
Example script to run difficulty evaluation.
Usage: python run_difficulty_eval.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Configuration paths (adjust these to match your setup)
    model_path = "trained_models/predify_simclr/scratch_no_pcoder/predify-simclr-temporal-no-aug-no-pcoder-grads-o0a4gkg7-ep=20-stp=26250.ckpt"
    linear_path = "trained_models/linear/predify_simclr_no_pcoder_grads_20epochs/your_linear_model.ckpt"  # Update with actual linear model checkpoint
    cfg_path = "configs/difficulty_eval_sample.yaml"
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please update the model_path in this script to point to your trained CurriculumMoCoV3 model.")
        return 1
        
    if not Path(linear_path).exists():
        print(f"Error: Linear classifier checkpoint not found at {linear_path}")
        print("Please update the linear_path in this script to point to your trained linear classifier.")
        return 1
        
    if not Path(cfg_path).exists():
        print(f"Error: Config file not found at {cfg_path}")
        print("Please ensure the config file exists.")
        return 1
    
    # Run difficulty evaluation
    cmd = [
        sys.executable, "main_difficulty_eval.py",
        "--model_path", model_path,
        "--linear_path", linear_path,
        "--cfg_path", cfg_path,
        "--difficulty_method", "reconstruction",  # or "entropy"
        "--batch_size", "64",
        "--num_workers", "4",
        "--device", "cuda",
        # "--max_batches", "10",  # Uncomment for quick testing
    ]
    
    print("Running difficulty evaluation with command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\nDifficulty evaluation completed successfully!")
        print("Results saved to 'difficulty_analysis/' directory")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nError running difficulty evaluation: {e}")
        return e.returncode

if __name__ == "__main__":
    exit(main()) 