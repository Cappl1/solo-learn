#!/usr/bin/env python3
"""
Test the simple batch analysis on just a few epochs
"""

import sys
from simple_batch_analysis import find_checkpoint_pairs, run_single_analysis
from pathlib import Path

def main():
    print("🧪 Testing Simple Batch Analysis (3 epochs only)")
    print("=" * 50)
    
    # Find checkpoint pairs
    base_dir = "/home/brothen/solo-learn/trained_models"
    checkpoint_pairs = find_checkpoint_pairs(base_dir, max_epochs=20)
    
    if not checkpoint_pairs:
        print("❌ No checkpoint pairs found!")
        return 1
    
    # Test with just first 3 epochs
    test_pairs = checkpoint_pairs[:3]
    print(f"Testing with {len(test_pairs)} checkpoint pairs:")
    
    for epoch, backbone, linear in test_pairs:
        print(f"  Epoch {epoch}: {Path(backbone).name}")
        print(f"            {Path(linear).name}")
    
    # Test directory
    output_dir = Path("test_simple_batch")
    output_dir.mkdir(exist_ok=True)
    
    # Test with just entropy method first
    test_methods = ['entropy']
    
    print(f"\n🚀 Running test analysis...")
    
    for epoch_num, backbone_path, linear_path in test_pairs:
        print(f"\n📈 Testing Epoch {epoch_num}")
        
        for method in test_methods:
            success = run_single_analysis(backbone_path, linear_path, epoch_num, 
                                        method, output_dir)
            if success:
                print(f"    ✅ {method} analysis succeeded")
            else:
                print(f"    ❌ {method} analysis failed")
                # Show the directory structure to debug
                epoch_dir = output_dir / f"epoch_{epoch_num:02d}_{method}"
                if epoch_dir.exists():
                    print(f"    Directory contents: {list(epoch_dir.iterdir())}")
                return 1
    
    print(f"\n🎉 Test successful! Ready to run full batch analysis.")
    print(f"Run: conda activate solo-learn && python simple_batch_analysis.py")
    
    return 0

if __name__ == "__main__":
    exit(main()) 