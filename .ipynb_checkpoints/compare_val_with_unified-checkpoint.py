#!/usr/bin/env python3
"""
Compare Validation Parquet with Unified CSV

This script compares dataset_val_all3.parquet with unified_mvimagenet_train.csv
to understand the relationship between validation and training data.
"""

import pandas as pd
import numpy as np
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load both validation parquet and unified CSV datasets"""
    print("=" * 80)
    print("LOADING DATASETS FOR COMPARISON")
    print("=" * 80)
    
    # Paths
    val_parquet_path = "/home/data/MVImageNet/dataset_val_all3.parquet"
    unified_csv_path = "unified_mvimagenet_train.csv"
    
    datasets = {}
    
    # Load validation parquet
    print(f"📂 Loading validation parquet...")
    if os.path.exists(val_parquet_path):
        val_df = pd.read_parquet(val_parquet_path)
        datasets['val_parquet'] = val_df
        print(f"   ✅ Loaded {len(val_df):,} rows, {len(val_df.columns)} columns")
        print(f"   Size: {os.path.getsize(val_parquet_path) / (1024**2):.1f}MB")
    else:
        print(f"   ❌ File not found: {val_parquet_path}")
        return None
    
    # Load unified CSV
    print(f"\n📂 Loading unified CSV...")
    if os.path.exists(unified_csv_path):
        unified_df = pd.read_csv(unified_csv_path, low_memory=False)
        datasets['unified_csv'] = unified_df
        print(f"   ✅ Loaded {len(unified_df):,} rows, {len(unified_df.columns)} columns")
        print(f"   Size: {os.path.getsize(unified_csv_path) / (1024**2):.1f}MB")
    else:
        print(f"   ❌ File not found: {unified_csv_path}")
        return None
    
    return datasets

def compare_schemas(datasets):
    """Compare the schemas/columns of both datasets"""
    print(f"\n📋 SCHEMA COMPARISON")
    print("-" * 50)
    
    val_df = datasets['val_parquet']
    unified_df = datasets['unified_csv']
    
    val_columns = set(val_df.columns)
    unified_columns = set(unified_df.columns)
    
    print(f"Validation parquet columns ({len(val_columns)}): {sorted(val_columns)}")
    print(f"Unified CSV columns ({len(unified_columns)}): {sorted(unified_columns)}")
    
    # Common columns
    common_columns = val_columns.intersection(unified_columns)
    print(f"\nCommon columns ({len(common_columns)}): {sorted(common_columns)}")
    
    # Unique to each
    val_only = val_columns - unified_columns
    unified_only = unified_columns - val_columns
    
    if val_only:
        print(f"Only in validation: {sorted(val_only)}")
    if unified_only:
        print(f"Only in unified: {sorted(unified_only)}")
    
    return common_columns

def compare_path_overlap(datasets, common_columns):
    """Compare path overlap between datasets"""
    print(f"\n🔍 PATH OVERLAP ANALYSIS")
    print("-" * 50)
    
    val_df = datasets['val_parquet']
    unified_df = datasets['unified_csv']
    
    if 'path' not in common_columns:
        print("❌ No 'path' column in common - cannot compare paths")
        return
    
    # Get unique paths from each dataset
    val_paths = set(val_df['path'].dropna())
    unified_paths = set(unified_df['path'].dropna())
    
    print(f"Validation unique paths: {len(val_paths):,}")
    print(f"Unified CSV unique paths: {len(unified_paths):,}")
    
    # Calculate overlap
    overlap_paths = val_paths.intersection(unified_paths)
    val_only_paths = val_paths - unified_paths
    unified_only_paths = unified_paths - val_paths
    
    print(f"\n📊 OVERLAP STATISTICS:")
    print(f"  Overlapping paths: {len(overlap_paths):,}")
    print(f"  Only in validation: {len(val_only_paths):,}")
    print(f"  Only in unified: {len(unified_only_paths):,}")
    
    # Calculate percentages
    if len(val_paths) > 0:
        val_overlap_pct = len(overlap_paths) / len(val_paths) * 100
        print(f"  Validation overlap: {val_overlap_pct:.1f}%")
    
    if len(unified_paths) > 0:
        unified_overlap_pct = len(overlap_paths) / len(unified_paths) * 100
        print(f"  Unified overlap: {unified_overlap_pct:.1f}%")
    
    # Show sample overlapping paths
    if overlap_paths:
        print(f"\n📝 Sample overlapping paths:")
        for i, path in enumerate(sorted(overlap_paths)[:10]):
            print(f"  {i+1}. {path}")
    
    # Show sample validation-only paths
    if val_only_paths:
        print(f"\n📝 Sample validation-only paths:")
        for i, path in enumerate(sorted(val_only_paths)[:10]):
            print(f"  {i+1}. {path}")
    
    return {
        'overlap_paths': overlap_paths,
        'val_only_paths': val_only_paths,
        'unified_only_paths': unified_only_paths
    }

def compare_categories_and_objects(datasets, common_columns):
    """Compare categories and objects between datasets"""
    print(f"\n🏷️  CATEGORY & OBJECT COMPARISON")
    print("-" * 50)
    
    val_df = datasets['val_parquet']
    unified_df = datasets['unified_csv']
    
    # Compare categories
    if 'category' in common_columns:
        val_categories = set(val_df['category'].dropna())
        unified_categories = set(unified_df['category'].dropna())
        
        print(f"Categories:")
        print(f"  Validation: {len(val_categories)} unique")
        print(f"  Unified: {len(unified_categories)} unique")
        
        common_categories = val_categories.intersection(unified_categories)
        print(f"  Common: {len(common_categories)} ({len(common_categories)/max(len(val_categories), len(unified_categories)):.1%})")
        
        # Show category distribution comparison
        val_cat_counts = val_df['category'].value_counts()
        unified_cat_counts = unified_df['category'].value_counts()
        
        print(f"\nTop 10 categories in validation:")
        for cat, count in val_cat_counts.head(10).items():
            unified_count = unified_cat_counts.get(cat, 0)
            print(f"  Category {cat}: {count:,} (unified: {unified_count:,})")
    
    # Compare objects
    if 'object' in common_columns:
        val_objects = set(val_df['object'].dropna())
        unified_objects = set(unified_df['object'].dropna())
        
        print(f"\nObjects:")
        print(f"  Validation: {len(val_objects):,} unique")
        print(f"  Unified: {len(unified_objects):,} unique")
        
        common_objects = val_objects.intersection(unified_objects)
        print(f"  Common: {len(common_objects):,} ({len(common_objects)/max(len(val_objects), len(unified_objects)):.1%})")

def analyze_frame_distribution(datasets, common_columns):
    """Analyze frame distribution between datasets"""
    print(f"\n🎬 FRAME DISTRIBUTION COMPARISON")
    print("-" * 50)
    
    val_df = datasets['val_parquet']
    unified_df = datasets['unified_csv']
    
    if 'frame' in common_columns:
        val_frames = val_df['frame'].value_counts().sort_index()
        unified_frames = unified_df['frame'].value_counts().sort_index()
        
        print(f"Frame distribution comparison:")
        all_frames = sorted(set(val_frames.index) | set(unified_frames.index))
        
        for frame_num in all_frames:
            val_count = val_frames.get(frame_num, 0)
            unified_count = unified_frames.get(frame_num, 0)
            print(f"  Frame {frame_num:2d}: Val={val_count:,}, Unified={unified_count:,}")

def check_data_leakage(datasets, path_overlap_info):
    """Check for potential data leakage between validation and training"""
    print(f"\n⚠️  DATA LEAKAGE ANALYSIS")
    print("-" * 50)
    
    overlap_paths = path_overlap_info['overlap_paths']
    
    if len(overlap_paths) > 0:
        leakage_pct = len(overlap_paths) / len(datasets['val_parquet']) * 100
        print(f"🚨 POTENTIAL DATA LEAKAGE DETECTED!")
        print(f"   {len(overlap_paths):,} validation paths found in unified training data")
        print(f"   This represents {leakage_pct:.1f}% of validation data")
        print(f"   ⚠️  This could lead to overly optimistic validation results!")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Remove overlapping paths from unified training dataset")
        print(f"   2. Or use only validation parquet for validation")
        print(f"   3. Ensure clean train/val split")
    else:
        print(f"✅ NO DATA LEAKAGE DETECTED")
        print(f"   Validation and unified training data are completely separate")

def generate_clean_training_dataset(datasets, path_overlap_info):
    """Generate a clean training dataset without validation leakage"""
    print(f"\n🧹 GENERATING CLEAN TRAINING DATASET")
    print("-" * 50)
    
    overlap_paths = path_overlap_info['overlap_paths']
    
    if len(overlap_paths) == 0:
        print(f"✅ No cleanup needed - datasets are already separate")
        return
    
    unified_df = datasets['unified_csv']
    
    print(f"Original unified dataset: {len(unified_df):,} samples")
    
    # Remove overlapping paths
    clean_df = unified_df[~unified_df['path'].isin(overlap_paths)]
    
    print(f"Clean training dataset: {len(clean_df):,} samples")
    print(f"Removed: {len(unified_df) - len(clean_df):,} overlapping samples")
    
    # Save clean dataset
    output_file = "clean_mvimagenet_train.csv"
    clean_df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"💾 Saved clean training dataset: {output_file} ({file_size_mb:.1f}MB)")
    
    return clean_df

def main():
    """Main comparison function"""
    print("Starting validation vs unified dataset comparison...")
    
    # Load datasets
    datasets = load_datasets()
    if datasets is None:
        print("❌ Failed to load datasets")
        return
    
    # Compare schemas
    common_columns = compare_schemas(datasets)
    
    # Compare path overlap
    path_overlap_info = compare_path_overlap(datasets, common_columns)
    
    # Compare categories and objects
    compare_categories_and_objects(datasets, common_columns)
    
    # Analyze frame distribution
    analyze_frame_distribution(datasets, common_columns)
    
    # Check for data leakage
    check_data_leakage(datasets, path_overlap_info)
    
    # Generate clean dataset if needed
    generate_clean_training_dataset(datasets, path_overlap_info)
    
    print(f"\n🎉 COMPARISON COMPLETE!")
    print(f"Summary:")
    print(f"  - Validation: {len(datasets['val_parquet']):,} samples")
    print(f"  - Unified: {len(datasets['unified_csv']):,} samples")
    if path_overlap_info:
        print(f"  - Overlap: {len(path_overlap_info['overlap_paths']):,} samples")

if __name__ == "__main__":
    main() 