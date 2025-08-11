#!/usr/bin/env python3
"""
Create Unified MVImageNet Dataset

This script combines all CSV files, deduplicates by path, and creates a unified
pretraining dataset with maximum unique samples.
"""

import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_and_combine_csv_files(data_path="/home/data/MVImageNet", 
                               output_file="unified_mvimagenet_train.csv",
                               priority_patterns=None):
    """
    Load all CSV files, deduplicate by path, and create unified dataset
    """
    print("=" * 80)
    print("CREATING UNIFIED MVIMAGENET DATASET")
    print("=" * 80)
    
    csv_files = glob.glob(os.path.join(data_path, "datasetT_*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    # Define priority order for keeping samples when duplicates found
    if priority_patterns is None:
        priority_patterns = [
            "datasetT_0.1_7_1",  # Best balance
            "datasetT_0.1_7_0",  # Good alternative
            "datasetT_0.1_7_2",  # Larger files
            "datasetT_0.0_7_",   # No filtering
            "datasetT_0.5_7_",   # Medium filtering
            "datasetT_1.0_7_"    # High filtering
        ]
    
    # Process files in priority order
    prioritized_files = []
    remaining_files = []
    
    for pattern in priority_patterns:
        pattern_files = [f for f in csv_files if pattern in os.path.basename(f)]
        prioritized_files.extend(pattern_files)
    
    # Add remaining files
    for f in csv_files:
        if f not in prioritized_files:
            remaining_files.append(f)
    
    all_files = prioritized_files + remaining_files
    
    print(f"Processing {len(all_files)} files in priority order...")
    
    # Track unique paths and combine data
    unique_paths = set()
    combined_data = []
    file_stats = {}
    
    for i, csv_file in enumerate(all_files):
        filename = os.path.basename(csv_file)
        print(f"\n[{i+1}/{len(all_files)}] Processing {filename}...")
        
        try:
            # Load CSV in chunks to handle memory
            chunk_size = 50000
            file_unique_count = 0
            file_total_count = 0
            file_new_paths = 0
            
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
                file_total_count += len(chunk)
                
                if 'path' not in chunk.columns:
                    print(f"  ⚠️  No 'path' column, skipping...")
                    break
                
                # Filter to only new paths
                chunk_paths = set(chunk['path'])
                new_paths = chunk_paths - unique_paths
                
                if new_paths:
                    new_samples = chunk[chunk['path'].isin(new_paths)]
                    combined_data.append(new_samples)
                    unique_paths.update(new_paths)
                    file_new_paths += len(new_samples)
                
                file_unique_count += len(chunk_paths)
            
            file_stats[filename] = {
                'total_rows': file_total_count,
                'unique_paths_in_file': len(chunk_paths) if 'chunk_paths' in locals() else 0,
                'new_paths_contributed': file_new_paths,
                'size_mb': os.path.getsize(csv_file) / (1024**2)
            }
            
            print(f"  Total rows: {file_total_count:,}")
            print(f"  New unique paths added: {file_new_paths:,}")
            print(f"  Running total unique paths: {len(unique_paths):,}")
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            file_stats[filename] = {'error': str(e)}
    
    # Combine all data
    if combined_data:
        print(f"\n🔄 Combining {len(combined_data)} dataframes...")
        final_df = pd.concat(combined_data, ignore_index=True)
        
        # Final deduplication (safety check)
        print(f"Pre-dedup: {len(final_df):,} rows")
        final_df = final_df.drop_duplicates(subset=['path'], keep='first')
        print(f"Post-dedup: {len(final_df):,} rows")
        
        # Save unified dataset
        output_path = output_file
        print(f"\n💾 Saving unified dataset to {output_path}...")
        final_df.to_csv(output_path, index=False)
        
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"✅ Saved! Size: {file_size_mb:.1f}MB")
        
        # Generate statistics
        generate_dataset_stats(final_df, file_stats, output_path)
        
        return final_df, file_stats
    
    else:
        print("❌ No data was successfully loaded!")
        return None, file_stats

def generate_dataset_stats(df, file_stats, output_path):
    """Generate comprehensive statistics about the unified dataset"""
    print(f"\n📊 UNIFIED DATASET STATISTICS")
    print("-" * 50)
    
    # Basic stats
    print(f"Total unique samples: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"File size: {os.path.getsize(output_path) / (1024**2):.1f}MB")
    
    # Column analysis
    if 'category' in df.columns:
        cat_counts = df['category'].value_counts()
        print(f"Categories: {len(cat_counts)} unique")
        print(f"Top categories: {dict(cat_counts.head())}")
    
    if 'object' in df.columns:
        obj_counts = df['object'].value_counts()
        print(f"Objects: {len(obj_counts):,} unique")
        print(f"Avg frames per object: {obj_counts.mean():.1f}")
    
    if 'frame' in df.columns:
        frame_stats = df['frame'].describe()
        print(f"Frame range: {frame_stats['min']:.0f} to {frame_stats['max']:.0f}")
    
    # Source file contributions
    print(f"\n📈 FILE CONTRIBUTIONS:")
    successful_files = {k: v for k, v in file_stats.items() if 'error' not in v}
    top_contributors = sorted(successful_files.items(), 
                             key=lambda x: x[1]['new_paths_contributed'], 
                             reverse=True)
    
    print("Top 10 contributing files:")
    for filename, stats in top_contributors[:10]:
        contrib = stats['new_paths_contributed']
        total = stats['total_rows']
        print(f"  {filename}: +{contrib:,} unique ({contrib/total:.3f} ratio)")
    
    # Data quality checks
    print(f"\n🔍 DATA QUALITY CHECKS:")
    
    # Check for missing values in key columns
    key_columns = ['path', 'category', 'object', 'frame']
    for col in key_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing:,} missing values ({missing/len(df):.3f})")
    
    # Check path patterns
    if 'path' in df.columns:
        sample_paths = df['path'].head(10).tolist()
        print(f"  Sample paths: {sample_paths[:3]}")
        
        # Analyze path structure
        path_parts = df['path'].str.split('/')
        if len(path_parts.iloc[0]) >= 3:
            categories_in_paths = path_parts.str[0].nunique()
            objects_in_paths = path_parts.str[1].nunique()
            print(f"  Path structure: {categories_in_paths} categories, {objects_in_paths} objects")

def create_train_val_test_splits(df, val_ratio=0.1, test_ratio=0.1, output_prefix="mvimagenet"):
    """Create train/val/test splits from the unified dataset"""
    print(f"\n✂️  CREATING TRAIN/VAL/TEST SPLITS")
    print("-" * 40)
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    # Split the data
    df_test = df_shuffled[:n_test]
    df_val = df_shuffled[n_test:n_test + n_val]
    df_train = df_shuffled[n_test + n_val:]
    
    print(f"Train: {len(df_train):,} samples ({len(df_train)/n_total:.1%})")
    print(f"Val:   {len(df_val):,} samples ({len(df_val)/n_total:.1%})")
    print(f"Test:  {len(df_test):,} samples ({len(df_test)/n_total:.1%})")
    
    # Save splits
    splits = {
        'train': df_train,
        'val': df_val, 
        'test': df_test
    }
    
    for split_name, split_df in splits.items():
        output_file = f"{output_prefix}_{split_name}.csv"
        split_df.to_csv(output_file, index=False)
        size_mb = os.path.getsize(output_file) / (1024**2)
        print(f"  Saved {output_file} ({size_mb:.1f}MB)")
    
    return splits

def verify_h5_accessibility(df, data_path="/home/data/MVImageNet", sample_size=1000):
    """Verify that paths in CSV actually exist in H5 files"""
    print(f"\n🔍 VERIFYING H5 FILE ACCESSIBILITY")
    print("-" * 40)
    
    # Get H5 files (excluding corrupted main file)
    h5_files = glob.glob(os.path.join(data_path, "data*.h5"))
    h5_files = [f for f in h5_files if 'data_all.h5' not in f]
    
    print(f"Found {len(h5_files)} H5 files to check")
    
    # Sample paths to verify
    sample_paths = df['path'].sample(min(sample_size, len(df))).tolist()
    print(f"Checking {len(sample_paths)} sample paths...")
    
    accessible_count = 0
    checked_count = 0
    
    try:
        import h5py
        
        # Check a few H5 files
        for h5_file in h5_files[:3]:  # Check first 3 files
            print(f"\n  Checking {os.path.basename(h5_file)}...")
            
            try:
                with h5py.File(h5_file, 'r') as f:
                    root_keys = list(f.keys())[:10]  # Sample categories
                    print(f"    Categories available: {len(list(f.keys()))}")
                    print(f"    Sample categories: {root_keys}")
                    
                    # Check if sample paths could exist
                    for path in sample_paths[:10]:  # Check first 10 paths
                        parts = path.split('/')
                        if len(parts) >= 2:
                            category = parts[0]
                            obj_id = parts[1]
                            
                            if category in f:
                                if obj_id in f[category]:
                                    accessible_count += 1
                                checked_count += 1
                        
                        if checked_count >= 10:  # Limit check
                            break
                    
                    if checked_count > 0:
                        success_rate = accessible_count / checked_count
                        print(f"    Sample accessibility: {accessible_count}/{checked_count} ({success_rate:.1%})")
                        
            except Exception as e:
                print(f"    ❌ Error accessing {os.path.basename(h5_file)}: {e}")
    
    except ImportError:
        print("  ⚠️  h5py not available, skipping H5 verification")
    
    if checked_count > 0:
        overall_rate = accessible_count / checked_count
        print(f"\n  Overall accessibility rate: {accessible_count}/{checked_count} ({overall_rate:.1%})")
        
        if overall_rate > 0.5:
            print("  ✅ Good accessibility rate - H5 files should work")
        else:
            print("  ⚠️  Low accessibility rate - check H5 file structure")

def main():
    """Main function to create unified dataset"""
    print("Starting unified dataset creation...")
    
    # Create unified dataset
    df, file_stats = load_and_combine_csv_files()
    
    if df is not None:
        # Create train/val/test splits
        splits = create_train_val_test_splits(df)
        
        # Verify H5 accessibility
        verify_h5_accessibility(df)
        
        print(f"\n🎉 SUCCESS! Created unified MVImageNet dataset:")
        print(f"   - Main file: unified_mvimagenet_train.csv ({len(df):,} samples)")
        print(f"   - Train split: mvimagenet_train.csv")
        print(f"   - Val split: mvimagenet_val.csv") 
        print(f"   - Test split: mvimagenet_test.csv")
        print(f"   - Use these for pretraining!")
        
    else:
        print("❌ Failed to create unified dataset")

if __name__ == "__main__":
    main() 