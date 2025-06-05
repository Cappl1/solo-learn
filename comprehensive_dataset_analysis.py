#!/usr/bin/env python3
"""
Comprehensive MVImageNet Dataset Analysis

Based on initial findings:
- 75 CSV files with different threshold/parameter combinations
- 44 H5 files (one main data_all.h5 + 43 data chunks)
- 3 parquet files (train, test, val) with train file corrupted

Data structure appears to be:
- Video sequences from different camera viewpoints
- Pose/rotation data (q0-q3 quaternions, t0-t2 translations)
- Object tracking across frames
"""

import os
import pandas as pd
import numpy as np
import h5py
import glob
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_csv_patterns():
    """Analyze CSV filename patterns to understand parameter space"""
    print("=== CSV FILES PATTERN ANALYSIS ===")
    
    data_path = "/home/data/MVImageNet"
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    patterns = defaultdict(list)
    thresholds = []
    param1_values = []
    param2_values = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename.startswith("datasetT_"):
            # Extract parameters: datasetT_{threshold}_{param1}_{param2}_{optional_id}.csv
            parts = filename.replace("datasetT_", "").replace(".csv", "").split("_")
            if len(parts) >= 3:
                threshold = parts[0]
                param1 = parts[1]
                param2 = parts[2]
                
                thresholds.append(threshold)
                param1_values.append(param1)
                param2_values.append(param2)
                
                pattern_key = f"{threshold}_{param1}_{param2}"
                patterns[pattern_key].append(filename)
    
    print(f"Total CSV files: {len(csv_files)}")
    print(f"Unique thresholds: {sorted(set(thresholds))}")
    print(f"Unique param1 values: {sorted(set(param1_values))}")
    print(f"Unique param2 values: {sorted(set(param2_values))}")
    
    print(f"\nPattern distribution:")
    for pattern, files in sorted(patterns.items()):
        print(f"  {pattern}: {len(files)} files")
    
    return patterns

def analyze_csv_content_detailed():
    """Detailed analysis of CSV content"""
    print("\n=== CSV CONTENT DETAILED ANALYSIS ===")
    
    data_path = "/home/data/MVImageNet"
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    # Sample a few different types
    sample_files = csv_files[:3]  # First 3 files
    
    all_columns = set()
    file_stats = {}
    
    for csv_file in sample_files:
        filename = os.path.basename(csv_file)
        print(f"\n--- {filename} ---")
        
        try:
            # Read first 1000 rows for analysis
            df = pd.read_csv(csv_file, nrows=1000)
            file_size_mb = os.path.getsize(csv_file) / (1024**2)
            
            all_columns.update(df.columns)
            
            file_stats[filename] = {
                'shape': df.shape,
                'size_mb': file_size_mb,
                'columns': list(df.columns),
                'dtypes': dict(df.dtypes)
            }
            
            print(f"Shape: {df.shape}, Size: {file_size_mb:.1f}MB")
            print(f"Columns: {list(df.columns)}")
            
            # Analyze key columns
            if 'category' in df.columns:
                cat_counts = df['category'].value_counts()
                print(f"Categories: {len(cat_counts)} unique, distribution: {dict(cat_counts.head())}")
            
            if 'path' in df.columns:
                # Analyze path patterns
                sample_paths = df['path'].head(10).tolist()
                print(f"Sample paths: {sample_paths[:3]}")
                
                # Extract directory structure
                path_parts = df['path'].str.split('/').str[0].value_counts()
                print(f"Top directories: {dict(path_parts.head())}")
            
            # Analyze quaternion and translation data
            pose_cols = [col for col in df.columns if col in ['q0', 'q1', 'q2', 'q3', 't0', 't1', 't2']]
            if pose_cols:
                print(f"Pose data columns: {pose_cols}")
                for col in pose_cols:
                    print(f"  {col}: range [{df[col].min():.3f}, {df[col].max():.3f}], mean {df[col].mean():.3f}")
            
            # Analyze frame sequences
            if 'frame' in df.columns:
                frame_stats = df['frame'].describe()
                print(f"Frame range: {df['frame'].min()} to {df['frame'].max()}")
                
                # Check for continuous sequences
                if 'object' in df.columns:
                    obj_frame_counts = df.groupby('object')['frame'].count()
                    print(f"Frames per object: mean {obj_frame_counts.mean():.1f}, std {obj_frame_counts.std():.1f}")
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"\nAll unique columns across files: {sorted(all_columns)}")
    return file_stats

def analyze_parquet_files_detailed():
    """Detailed analysis of parquet files"""
    print("\n=== PARQUET FILES DETAILED ANALYSIS ===")
    
    data_path = "/home/data/MVImageNet"
    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
    
    for pf in parquet_files:
        filename = os.path.basename(pf)
        print(f"\n--- {filename} ---")
        
        try:
            df = pd.read_parquet(pf)
            file_size_mb = os.path.getsize(pf) / (1024**2)
            
            print(f"Shape: {df.shape}, Size: {file_size_mb:.1f}MB")
            print(f"Columns: {list(df.columns)}")
            
            # Basic statistics
            print(f"Data types: {dict(df.dtypes)}")
            
            # Analyze categories
            if 'category' in df.columns:
                cat_counts = df['category'].value_counts()
                print(f"Categories: {len(cat_counts)} unique")
                print(f"Category distribution: {dict(cat_counts.head(10))}")
            
            # Analyze objects
            if 'object' in df.columns:
                obj_counts = df['object'].value_counts()
                print(f"Objects: {len(obj_counts)} unique")
                print(f"Frames per object: mean {obj_counts.mean():.1f}, std {obj_counts.std():.1f}")
            
            # Analyze pose data
            pose_cols = [col for col in df.columns if col in ['q0', 'q1', 'q2', 'q3', 't0', 't1', 't2']]
            if pose_cols:
                print(f"Pose statistics:")
                for col in pose_cols:
                    stats = df[col].describe()
                    print(f"  {col}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            
            # Analyze frame sequences
            if 'frame' in df.columns:
                frame_stats = df['frame'].describe()
                print(f"Frame statistics: min={frame_stats['min']}, max={frame_stats['max']}, mean={frame_stats['mean']:.1f}")
            
            # Sample data
            print(f"Sample data:")
            print(df.head(3))
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

def analyze_h5_structure():
    """Analyze H5 file structure"""
    print("\n=== H5 FILES STRUCTURE ANALYSIS ===")
    
    data_path = "/home/data/MVImageNet"
    h5_files = glob.glob(os.path.join(data_path, "*.h5"))
    
    # Check main file first
    main_file = [f for f in h5_files if "data_all.h5" in f]
    if main_file:
        print(f"--- Main file: {os.path.basename(main_file[0])} ---")
        try:
            with h5py.File(main_file[0], 'r') as f:
                print(f"Root keys: {list(f.keys())}")
                
                def explore_h5_group(group, name, max_depth=2, current_depth=0):
                    if current_depth >= max_depth:
                        return
                    
                    for key in list(group.keys())[:5]:  # Show first 5 items
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            print(f"  {'  ' * current_depth}{name}/{key}/ (group, {len(item.keys())} items)")
                            explore_h5_group(item, f"{name}/{key}", max_depth, current_depth + 1)
                        elif isinstance(item, h5py.Dataset):
                            print(f"  {'  ' * current_depth}{name}/{key} (dataset, shape={item.shape}, dtype={item.dtype})")
                
                explore_h5_group(f, "", max_depth=3)
                
        except Exception as e:
            print(f"Error reading main H5 file: {e}")
    
    # Sample a few other H5 files
    other_files = [f for f in h5_files if "data_all.h5" not in f][:3]
    for h5_file in other_files:
        filename = os.path.basename(h5_file)
        print(f"\n--- Sample H5: {filename} ---")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                file_size_mb = os.path.getsize(h5_file) / (1024**2)
                print(f"Size: {file_size_mb:.1f}MB")
                print(f"Root keys: {list(f.keys())}")
                
                # Explore structure
                def explore_sample_h5(group, max_items=3):
                    items_shown = 0
                    for key in group.keys():
                        if items_shown >= max_items:
                            print(f"  ... and {len(group.keys()) - max_items} more items")
                            break
                        
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            print(f"  {key}/ (group, {len(item.keys())} items)")
                        elif isinstance(item, h5py.Dataset):
                            print(f"  {key} (dataset, shape={item.shape}, dtype={item.dtype}, size={item.nbytes/(1024**2):.1f}MB)")
                        
                        items_shown += 1
                
                explore_sample_h5(f)
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

def generate_summary_report():
    """Generate a summary report of findings"""
    print("\n" + "="*80)
    print("DATASET SUMMARY REPORT")
    print("="*80)
    
    print("""
Key Findings:
1. This appears to be a multi-view video dataset with pose/camera tracking
2. Images are organized by objects/scenes with frame sequences
3. Each sample has 6DOF pose data (quaternion rotation + translation)
4. Multiple data splits and parameter configurations available

Data Structure:
- path: relative path to images (e.g., '0/0000f5bc/images/001.jpg')
- category: object category (0-based integer)
- object: unique object identifier (hex string)
- frame: frame number in sequence
- q0-q3: quaternion rotation components
- t0-t2: translation vector components
- partition: data split identifier
- length: sequence length

File Organization:
- Main parquet files: train/test/val splits (~189MB/26MB/26MB)
- CSV files: 75 files with different threshold parameters
- H5 files: 44 files containing image data (3GB each)
- Masked dataset: Additional preprocessed data

Recommended Usage:
1. Use parquet files for metadata and pose information
2. Use H5 files for loading actual image data
3. CSV files contain filtered/processed versions with different parameters
""")

def main():
    """Run comprehensive analysis"""
    try:
        patterns = analyze_csv_patterns()
        csv_stats = analyze_csv_content_detailed()
        analyze_parquet_files_detailed()
        analyze_h5_structure()
        generate_summary_report()
        
        print("\n✅ Comprehensive analysis completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 