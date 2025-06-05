#!/usr/bin/env python3
"""
MVImageNet Dataset Analysis Script

This script analyzes the structure and contents of the MVImageNet dataset located at /home/data/MVImageNet.
It examines:
1. File structure and sizes
2. CSV file contents and statistics
3. Parquet file contents and statistics  
4. H5 file structure and contents
5. Masked dataset structure
"""

import os
import pandas as pd
import numpy as np
import h5py
import glob
from pathlib import Path
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def analyze_file_structure(data_path="/home/data/MVImageNet"):
    """Analyze the overall file structure and sizes"""
    print("=" * 80)
    print("MVImageNet Dataset Analysis")
    print("=" * 80)
    
    # Get file counts and sizes by type
    file_stats = defaultdict(lambda: {'count': 0, 'total_size': 0, 'files': []})
    
    for file_path in glob.glob(os.path.join(data_path, "*")):
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            size = os.path.getsize(file_path)
            
            if filename.endswith('.h5'):
                file_stats['h5']['count'] += 1
                file_stats['h5']['total_size'] += size
                file_stats['h5']['files'].append((filename, size))
            elif filename.endswith('.csv'):
                file_stats['csv']['count'] += 1
                file_stats['csv']['total_size'] += size
                file_stats['csv']['files'].append((filename, size))
            elif filename.endswith('.parquet'):
                file_stats['parquet']['count'] += 1
                file_stats['parquet']['total_size'] += size
                file_stats['parquet']['files'].append((filename, size))
    
    print("\n📁 FILE STRUCTURE OVERVIEW")
    print("-" * 50)
    for file_type, stats in file_stats.items():
        size_gb = stats['total_size'] / (1024**3)
        print(f"{file_type.upper()} Files: {stats['count']} files, {size_gb:.2f} GB total")
    
    return file_stats

def analyze_csv_files(data_path="/home/data/MVImageNet", sample_size=5):
    """Analyze CSV files structure and content"""
    print("\n📊 CSV FILES ANALYSIS")
    print("-" * 50)
    
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    # Analyze filename patterns
    patterns = defaultdict(list)
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename.startswith("datasetT_"):
            # Extract parameters from filename
            parts = filename.replace("datasetT_", "").replace(".csv", "").split("_")
            if len(parts) >= 3:
                threshold = parts[0]
                param1 = parts[1]
                param2 = parts[2]
                patterns[f"{threshold}_{param1}_{param2}"].append(filename)
    
    print(f"\n🔍 CSV Filename Patterns ({len(patterns)} unique patterns):")
    for pattern, files in list(patterns.items())[:10]:  # Show first 10 patterns
        print(f"  {pattern}: {len(files)} files")
    
    # Analyze a sample of CSV files
    print(f"\n📋 Analyzing sample of {min(sample_size, len(csv_files))} CSV files:")
    
    csv_analysis = {}
    for i, csv_file in enumerate(csv_files[:sample_size]):
        filename = os.path.basename(csv_file)
        print(f"\n  📄 {filename}")
        
        try:
            # Read CSV with different encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding, nrows=1000)  # Read first 1000 rows for analysis
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is not None:
                file_size_mb = os.path.getsize(csv_file) / (1024**2)
                csv_analysis[filename] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': file_size_mb,
                    'column_names': list(df.columns),
                    'dtypes': dict(df.dtypes.astype(str)),
                    'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
                }
                
                print(f"    Shape: {df.shape}")
                print(f"    Size: {file_size_mb:.2f} MB")
                print(f"    Columns: {list(df.columns)}")
                print(f"    Data types: {dict(df.dtypes.astype(str))}")
                
                # Show basic statistics for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    print(f"    Numeric column stats:")
                    for col in numeric_cols[:3]:  # Show first 3 numeric columns
                        print(f"      {col}: min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")
                
            else:
                print(f"    ❌ Could not read file with any encoding")
                
        except Exception as e:
            print(f"    ❌ Error reading file: {str(e)}")
    
    return csv_analysis

def analyze_parquet_files(data_path="/home/data/MVImageNet"):
    """Analyze parquet files structure and content"""
    print("\n📦 PARQUET FILES ANALYSIS")
    print("-" * 50)
    
    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
    parquet_analysis = {}
    
    for parquet_file in parquet_files:
        filename = os.path.basename(parquet_file)
        print(f"\n  📄 {filename}")
        
        try:
            df = pd.read_parquet(parquet_file)
            file_size_mb = os.path.getsize(parquet_file) / (1024**2)
            
            parquet_analysis[filename] = {
                'rows': len(df),
                'columns': len(df.columns),
                'size_mb': file_size_mb,
                'column_names': list(df.columns),
                'dtypes': dict(df.dtypes.astype(str)),
                'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
            }
            
            print(f"    Shape: {df.shape}")
            print(f"    Size: {file_size_mb:.2f} MB")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Data types: {dict(df.dtypes.astype(str))}")
            
            # Show value counts for categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols[:3]:  # Show first 3 categorical columns
                unique_count = df[col].nunique()
                print(f"    {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"      Values: {list(df[col].unique())}")
            
            # Show basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"    Numeric column stats:")
                for col in numeric_cols[:3]:  # Show first 3 numeric columns
                    print(f"      {col}: min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")
                    
        except Exception as e:
            print(f"    ❌ Error reading parquet file: {str(e)}")
    
    return parquet_analysis

def analyze_h5_files(data_path="/home/data/MVImageNet", sample_size=3):
    """Analyze H5 files structure and content"""
    print("\n🗃️  H5 FILES ANALYSIS")
    print("-" * 50)
    
    h5_files = glob.glob(os.path.join(data_path, "*.h5"))
    print(f"Found {len(h5_files)} H5 files")
    
    # Separate main file from others
    main_file = [f for f in h5_files if "data_all.h5" in f]
    other_files = [f for f in h5_files if "data_all.h5" not in f]
    
    h5_analysis = {}
    
    # Analyze main file first
    if main_file:
        print(f"\n  📄 MAIN FILE: {os.path.basename(main_file[0])}")
        try:
            with h5py.File(main_file[0], 'r') as f:
                file_size_mb = os.path.getsize(main_file[0]) / (1024**2)
                structure = analyze_h5_structure(f, max_depth=3)
                h5_analysis[os.path.basename(main_file[0])] = {
                    'size_mb': file_size_mb,
                    'structure': structure
                }
                print(f"    Size: {file_size_mb:.2f} MB")
                print_h5_structure(structure, indent="    ")
        except Exception as e:
            print(f"    ❌ Error reading H5 file: {str(e)}")
    
    # Analyze sample of other H5 files
    print(f"\n  📄 SAMPLE OF OTHER H5 FILES ({min(sample_size, len(other_files))} files):")
    
    for i, h5_file in enumerate(other_files[:sample_size]):
        filename = os.path.basename(h5_file)
        print(f"\n    📄 {filename}")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                file_size_mb = os.path.getsize(h5_file) / (1024**2)
                structure = analyze_h5_structure(f, max_depth=2)
                h5_analysis[filename] = {
                    'size_mb': file_size_mb,
                    'structure': structure
                }
                print(f"      Size: {file_size_mb:.2f} MB")
                print_h5_structure(structure, indent="      ")
        except Exception as e:
            print(f"      ❌ Error reading H5 file: {str(e)}")
    
    return h5_analysis

def analyze_h5_structure(h5_group, max_depth=3, current_depth=0):
    """Recursively analyze H5 file structure"""
    structure = {}
    
    if current_depth >= max_depth:
        return {"...": "max_depth_reached"}
    
    for key in h5_group.keys():
        item = h5_group[key]
        if isinstance(item, h5py.Group):
            structure[key] = {
                'type': 'group',
                'keys': list(item.keys())[:10],  # Show first 10 keys
                'total_keys': len(item.keys()),
                'children': analyze_h5_structure(item, max_depth, current_depth + 1) if current_depth < max_depth - 1 else {}
            }
        elif isinstance(item, h5py.Dataset):
            structure[key] = {
                'type': 'dataset',
                'shape': item.shape,
                'dtype': str(item.dtype),
                'size_mb': item.nbytes / (1024**2)
            }
    
    return structure

def print_h5_structure(structure, indent=""):
    """Pretty print H5 structure"""
    for key, value in structure.items():
        if value.get('type') == 'group':
            total_keys = value.get('total_keys', 0)
            print(f"{indent}{key}/ (group, {total_keys} items)")
            if value.get('children'):
                print_h5_structure(value['children'], indent + "  ")
        elif value.get('type') == 'dataset':
            shape = value.get('shape', 'unknown')
            dtype = value.get('dtype', 'unknown')
            size_mb = value.get('size_mb', 0)
            print(f"{indent}{key} (dataset, shape={shape}, dtype={dtype}, {size_mb:.2f}MB)")
        else:
            print(f"{indent}{key}: {value}")

def analyze_masked_dataset(data_path="/home/data/MVImageNet"):
    """Analyze the masked dataset directory"""
    print("\n🎭 MASKED DATASET ANALYSIS")
    print("-" * 50)
    
    masked_path = os.path.join(data_path, "masked_dataset")
    if not os.path.exists(masked_path):
        print("Masked dataset directory not found")
        return {}
    
    files = os.listdir(masked_path)
    print(f"Found {len(files)} files in masked_dataset directory")
    
    masked_analysis = {}
    
    # Analyze parquet files in masked dataset
    parquet_files = [f for f in files if f.endswith('.parquet')]
    h5_files = [f for f in files if f.endswith('.h5')]
    
    print(f"  Parquet files: {len(parquet_files)}")
    print(f"  H5 files: {len(h5_files)}")
    
    # Analyze parquet files
    for parquet_file in parquet_files:
        file_path = os.path.join(masked_path, parquet_file)
        print(f"\n  📄 {parquet_file}")
        
        try:
            df = pd.read_parquet(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024**2)
            
            print(f"    Shape: {df.shape}")
            print(f"    Size: {file_size_mb:.2f} MB")
            print(f"    Columns: {list(df.columns)}")
            
            masked_analysis[parquet_file] = {
                'type': 'parquet',
                'rows': len(df),
                'columns': len(df.columns),
                'size_mb': file_size_mb,
                'column_names': list(df.columns)
            }
            
        except Exception as e:
            print(f"    ❌ Error reading parquet file: {str(e)}")
    
    # Analyze a sample of H5 files
    for h5_file in h5_files[:2]:  # Sample first 2 H5 files
        file_path = os.path.join(masked_path, h5_file)
        print(f"\n  📄 {h5_file}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                file_size_mb = os.path.getsize(file_path) / (1024**2)
                structure = analyze_h5_structure(f, max_depth=2)
                
                print(f"    Size: {file_size_mb:.2f} MB")
                print_h5_structure(structure, indent="    ")
                
                masked_analysis[h5_file] = {
                    'type': 'h5',
                    'size_mb': file_size_mb,
                    'structure': structure
                }
                
        except Exception as e:
            print(f"    ❌ Error reading H5 file: {str(e)}")
    
    return masked_analysis

def save_analysis_summary(file_stats, csv_analysis, parquet_analysis, h5_analysis, masked_analysis):
    """Save analysis summary to JSON file"""
    summary = {
        'file_structure': file_stats,
        'csv_analysis': csv_analysis,
        'parquet_analysis': parquet_analysis,
        'h5_analysis': h5_analysis,
        'masked_analysis': masked_analysis,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open('mvimagenet_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Analysis summary saved to: mvimagenet_analysis_summary.json")

def main():
    """Main analysis function"""
    data_path = "/home/data/MVImageNet"
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return
    
    print(f"🔍 Analyzing MVImageNet dataset at: {data_path}")
    
    # Run all analyses
    file_stats = analyze_file_structure(data_path)
    csv_analysis = analyze_csv_files(data_path, sample_size=5)
    parquet_analysis = analyze_parquet_files(data_path)
    h5_analysis = analyze_h5_files(data_path, sample_size=3)
    masked_analysis = analyze_masked_dataset(data_path)
    
    # Save summary
    save_analysis_summary(file_stats, csv_analysis, parquet_analysis, h5_analysis, masked_analysis)
    
    print("\n" + "=" * 80)
    print("✅ Analysis completed!")
    print("=" * 80)

if __name__ == "__main__":
    main() 