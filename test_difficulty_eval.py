#!/usr/bin/env python3
"""
Test script to verify difficulty evaluation compatibility.
This script tests basic imports and functionality without running the full evaluation.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
        
    try:
        from solo.methods.curriculum_mocov3 import CurriculumMoCoV3
        print("✓ CurriculumMoCoV3 imported successfully")
    except ImportError as e:
        print(f"✗ CurriculumMoCoV3 import failed: {e}")
        return False
        
    try:
        from solo.data.classification_dataloader import prepare_data
        print("✓ Data loader imported successfully")
    except ImportError as e:
        print(f"✗ Data loader import failed: {e}")
        return False
        
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from scipy.stats import spearmanr, pearsonr
        from sklearn.linear_model import LogisticRegression
        print("✓ Analysis libraries imported successfully")
    except ImportError as e:
        print(f"✗ Analysis libraries import failed: {e}")
        return False
        
    return True

def test_difficulty_evaluator_class():
    """Test that the DifficultyEvaluator class can be imported and instantiated."""
    print("\nTesting DifficultyEvaluator class...")
    
    try:
        # Import the main module
        sys.path.insert(0, str(Path(__file__).parent))
        from main_difficulty_eval import DifficultyEvaluator
        print("✓ DifficultyEvaluator class imported successfully")
        
        # Test that we can access the class methods (without calling them)
        methods = ['load_pretrained_model', 'load_linear_classifier', 
                  'compute_reconstruction_difficulty', 'compute_entropy_difficulty',
                  'evaluate_dataset', 'analyze_results']
        
        for method in methods:
            if hasattr(DifficultyEvaluator, method):
                print(f"✓ Method {method} found")
            else:
                print(f"✗ Method {method} not found")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ DifficultyEvaluator test failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test that the sample config can be loaded."""
    print("\nTesting config loading...")
    
    try:
        from omegaconf import OmegaConf
        
        config_path = Path("configs/difficulty_eval_sample.yaml")
        if not config_path.exists():
            print(f"✗ Config file not found at {config_path}")
            return False
            
        cfg = OmegaConf.load(config_path)
        print("✓ Config loaded successfully")
        
        # Check required fields
        required_fields = [
            "backbone.name",
            "data.dataset", 
            "data.num_classes",
            "method_kwargs.curriculum_type"
        ]
        
        for field in required_fields:
            try:
                OmegaConf.select(cfg, field)
                print(f"✓ Config field {field} found")
            except Exception:
                print(f"✗ Config field {field} missing")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running difficulty evaluation compatibility tests...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("DifficultyEvaluator Class Test", test_difficulty_evaluator_class),
        ("Config Loading Test", test_config_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print('='*50)
        
        try:
            if test_func():
                print(f"\n✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("\n🎉 All tests passed! The difficulty evaluation should work correctly.")
        print("\nNext steps:")
        print("1. Train or locate your CurriculumMoCoV3 model")
        print("2. Train a linear classifier on the pretrained features")
        print("3. Update paths in run_difficulty_eval.py")
        print("4. Run the evaluation!")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues before running the evaluation.")
        return 1

if __name__ == "__main__":
    exit(main()) 