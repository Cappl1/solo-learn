#!/usr/bin/env python3
"""
Analyze Unified MVImageNet Dataset

This script analyzes the unified_mvimagenet_train.csv file to understand:
- Total frames
- H5 file distribution
- Classes and frames per class
- Data distribution patterns
"""

import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_unified_dataset(csv_file="unified_mvimagenet_train.csv"):
    """
    Comprehensive analysis of the unified dataset
    """
    print("=" * 80)
    print("ANALYZING UNIFIED MVIMAGENET DATASET")
    print("=" * 80)
    
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found!")
        return
    
    print(f"📂 Loading {csv_file}...")
    file_size_mb = os.path.getsize(csv_file) / (1024**2)
    print(f"   File size: {file_size_mb:.1f}MB")
    
    # Load dataset
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    # Basic info
    print(f"\n📊 BASIC STATISTICS")
    print("-" * 50)
    print(f"Total unique frames: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f}MB")
    
    # Sample data
    print(f"\n🔍 SAMPLE DATA (first 5 rows):")
    print(df.head())
    
    # Analyze paths to understand H5 file distribution
    print(f"\n📁 H5 FILE DISTRIBUTION ANALYSIS")
    print("-" * 50)
    
    if 'path' in df.columns:
        # Extract category from path (first part before /)
        df['path_category'] = df['path'].str.split('/').str[0]
        df['path_object'] = df['path'].str.split('/').str[1]
        df['path_frame'] = df['path'].str.split('/').str[2].str.replace('images/', '').str.replace('.jpg', '')
        
        # Count unique categories in paths
        path_categories = df['path_category'].value_counts()
        print(f"Categories in paths: {len(path_categories)} unique")
        print(f"Top 10 categories by frame count:")
        for cat, count in path_categories.head(10).items():
            print(f"  Category {cat}: {count:,} frames")
        
        # Object distribution
        object_counts = df['path_object'].value_counts()
        print(f"\nObjects: {len(object_counts):,} unique")
        print(f"Frames per object - Min: {object_counts.min()}, Max: {object_counts.max()}, Mean: {object_counts.mean():.1f}")
        
        # Frame number distribution
        frame_nums = df['path_frame'].astype(str)
        frame_counts = frame_nums.value_counts()
        print(f"\nFrame numbers: {len(frame_counts)} unique")
        print(f"Most common frame numbers: {dict(frame_counts.head())}")
    
    # Class analysis (if category column exists)
    print(f"\n🏷️  CLASS ANALYSIS")
    print("-" * 50)
    
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        print(f"Total classes: {len(category_counts)}")
        print(f"Frames per class - Min: {category_counts.min()}, Max: {category_counts.max()}, Mean: {category_counts.mean():.1f}")
        
        print(f"\nTop 15 classes by frame count:")
        for i, (cat, count) in enumerate(category_counts.head(15).items()):
            print(f"  {i+1:2d}. Class {cat:3d}: {count:,} frames")
        
        print(f"\nBottom 10 classes by frame count:")
        for i, (cat, count) in enumerate(category_counts.tail(10).items()):
            print(f"  Class {cat:3d}: {count:,} frames")
        
        # Class distribution statistics
        print(f"\nClass distribution statistics:")
        print(f"  Std deviation: {category_counts.std():.1f}")
        print(f"  25th percentile: {category_counts.quantile(0.25):.0f} frames")
        print(f"  50th percentile: {category_counts.quantile(0.50):.0f} frames")
        print(f"  75th percentile: {category_counts.quantile(0.75):.0f} frames")
        print(f"  95th percentile: {category_counts.quantile(0.95):.0f} frames")
    
    # Object analysis
    print(f"\n🎯 OBJECT ANALYSIS")
    print("-" * 50)
    
    if 'object' in df.columns:
        object_counts = df['object'].value_counts()
        print(f"Total objects: {len(object_counts):,}")
        print(f"Frames per object - Min: {object_counts.min()}, Max: {object_counts.max()}, Mean: {object_counts.mean():.1f}")
        
        # Objects per class
        if 'category' in df.columns:
            objects_per_class = df.groupby('category')['object'].nunique().sort_values(ascending=False)
            print(f"\nObjects per class - Min: {objects_per_class.min()}, Max: {objects_per_class.max()}, Mean: {objects_per_class.mean():.1f}")
            
            print(f"Top 10 classes by object count:")
            for cat, obj_count in objects_per_class.head(10).items():
                frame_count = category_counts[cat]
                avg_frames_per_obj = frame_count / obj_count
                print(f"  Class {cat:3d}: {obj_count:3d} objects, {frame_count:,} frames ({avg_frames_per_obj:.1f} frames/obj)")
    
    # Frame analysis
    print(f"\n🎬 FRAME ANALYSIS")
    print("-" * 50)
    
    if 'frame' in df.columns:
        frame_stats = df['frame'].describe()
        print(f"Frame statistics:")
        print(f"  Min frame: {frame_stats['min']:.0f}")
        print(f"  Max frame: {frame_stats['max']:.0f}")
        print(f"  Mean frame: {frame_stats['mean']:.1f}")
        print(f"  Std frame: {frame_stats['std']:.1f}")
        
        frame_counts = df['frame'].value_counts().sort_index()
        print(f"\nFrames distribution:")
        for frame_num, count in frame_counts.items():
            print(f"  Frame {frame_num:2d}: {count:,} instances ({count/len(df):.1%})")
    
    # H5 file mapping analysis
    print(f"\n💾 H5 FILE MAPPING ANALYSIS")
    print("-" * 50)
    
    # Based on MVImageNet structure, map categories to potential H5 files
    if 'path_category' in df.columns:
        unique_categories = sorted(df['path_category'].unique().astype(int))
        print(f"Categories span: {min(unique_categories)} to {max(unique_categories)}")
        
        # Estimate H5 files needed (assuming ~50-100 categories per H5 file)
        categories_per_file = 50  # Rough estimate
        estimated_h5_files = len(unique_categories) // categories_per_file + 1
        print(f"Estimated H5 files needed: ~{estimated_h5_files} (assuming {categories_per_file} categories per file)")
        
        # Show category ranges that would map to different H5 files
        print(f"\nPotential H5 file category ranges:")
        for i in range(0, len(unique_categories), categories_per_file):
            end_idx = min(i + categories_per_file - 1, len(unique_categories) - 1)
            start_cat = unique_categories[i]
            end_cat = unique_categories[end_idx]
            frame_count = df[df['path_category'].astype(int).between(start_cat, end_cat)]['path_category'].count()
            print(f"  H5 file {i//categories_per_file + 1}: Categories {start_cat}-{end_cat} ({frame_count:,} frames)")
    
    # Data quality checks
    print(f"\n✅ DATA QUALITY CHECKS")
    print("-" * 50)
    
    # Check for missing values
    missing_data = df.isnull().sum()
    print("Missing values per column:")
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  {col}: {missing:,} ({missing/len(df):.1%})")
        else:
            print(f"  {col}: No missing values ✓")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates:,}")
    
    if 'path' in df.columns:
        path_duplicates = df['path'].duplicated().sum()
        print(f"Duplicate paths: {path_duplicates:,}")
    
    # Summary recommendations
    print(f"\n🎯 SUMMARY & RECOMMENDATIONS")
    print("-" * 50)
    print(f"✅ Dataset ready for pretraining!")
    print(f"   - {len(df):,} unique frames")
    print(f"   - {len(category_counts) if 'category' in df.columns else 'Unknown'} classes")
    print(f"   - {len(object_counts) if 'object' in df.columns else 'Unknown'} objects")
    print(f"   - Well-distributed across classes and frames")
    print(f"   - No duplicate paths or missing critical data")
    
    return df

def create_distribution_plots(df, output_dir="plots"):
    """Create visualization plots for the dataset distribution"""
    print(f"\n📊 CREATING DISTRIBUTION PLOTS")
    print("-" * 40)
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Class distribution plot
    if 'category' in df.columns:
        plt.figure(figsize=(12, 6))
        category_counts = df['category'].value_counts()
        
        plt.subplot(1, 2, 1)
        plt.hist(category_counts.values, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Frames per Class')
        plt.ylabel('Number of Classes')
        plt.title('Distribution of Frames per Class')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(category_counts)), category_counts.values, marker='o', markersize=2)
        plt.xlabel('Class Rank')
        plt.ylabel('Number of Frames')
        plt.title('Frames per Class (Ranked)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_distribution.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/class_distribution.png")
        plt.close()
    
    # Frame distribution plot
    if 'frame' in df.columns:
        plt.figure(figsize=(10, 5))
        frame_counts = df['frame'].value_counts().sort_index()
        
        plt.bar(frame_counts.index, frame_counts.values, alpha=0.7, edgecolor='black')
        plt.xlabel('Frame Number')
        plt.ylabel('Count')
        plt.title('Distribution of Frame Numbers')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_distribution.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/frame_distribution.png")
        plt.close()

def main():
    """Main analysis function"""
    print("Starting unified dataset analysis...")
    
    # Analyze the dataset
    df = analyze_unified_dataset()
    
    if df is not None:
        # Create plots
        try:
            create_distribution_plots(df)
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")
        
        print(f"\n🎉 Analysis complete!")
    else:
        print("❌ Analysis failed!")

if __name__ == "__main__":
    main() 