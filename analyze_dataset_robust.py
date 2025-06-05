#!/usr/bin/env python3
"""
Robust MVImageNet Dataset Analysis Script
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def safe_read_parquet(file_path):
    """Safely read parquet files with different engines"""
    try:
        # Try pyarrow first
        df = pd.read_parquet(file_path, engine='pyarrow')
        return df, 'pyarrow'
    except:
        try:
            # Try fastparquet
            df = pd.read_parquet(file_path, engine='fastparquet')
            return df, 'fastparquet'
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None

def analyze_parquet_files():
    """Analyze the main parquet dataset files"""
    print('=== MVImageNet Parquet Files Analysis ===')
    print()
    
    data_dir = Path('/home/data/MVImageNet')
    parquet_files = [
        'dataset_train_all3.parquet',
        'dataset_val_all3.parquet', 
        'dataset_test_all3.parquet'
    ]
    
    dataset_info = {}
    
    for filename in parquet_files:
        file_path = data_dir / filename
        if file_path.exists():
            print(f'=== {filename} ===')
            print(f'File size: {file_path.stat().st_size / 1024**2:.2f} MB')
            
            df, engine = safe_read_parquet(file_path)
            if df is not None:
                print(f'Successfully read with {engine} engine')
                print(f'Shape: {df.shape}')
                print(f'Columns: {list(df.columns)}')
                print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
                print()
                
                # Show data types
                print('Data types:')
                for col, dtype in df.dtypes.items():
                    print(f'  {col}: {dtype}')
                print()
                
                # Show first few rows
                if len(df) > 0:
                    print('First 3 rows:')
                    print(df.head(3))
                    print()
                    
                    # Basic statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        print('Statistics for numeric columns:')
                        print(df[numeric_cols].describe())
                        print()
                    
                    # Info about categorical columns
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    for col in categorical_cols:
                        unique_count = df[col].nunique()
                        print(f'{col}: {unique_count} unique values')
                        if unique_count <= 10:
                            print(f'  Values: {df[col].unique()[:10]}')
                        print()
                
                dataset_info[filename] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'success': True
                }
            else:
                dataset_info[filename] = {'success': False}
            
            print('-' * 60)
            print()
    
    return dataset_info

def analyze_csv_files():
    """Analyze CSV files with dataset parameters"""
    print('=== CSV Files Analysis ===')
    
    data_dir = Path('/home/data/MVImageNet')
    csv_files = list(data_dir.glob('datasetT_*.csv'))
    
    print(f'Found {len(csv_files)} CSV files with naming pattern datasetT_*.csv')
    print()
    
    # Analyze file naming pattern
    print('CSV file naming patterns:')
    patterns = {}
    for csv_file in csv_files:
        name_parts = csv_file.stem.split('_')
        if len(name_parts) >= 3:
            threshold = name_parts[1]
            param2 = name_parts[2]
            param3 = name_parts[3] if len(name_parts) > 3 else 'none'
            pattern = f'{threshold}_{param2}_{param3}'
            patterns[pattern] = patterns.get(pattern, 0) + 1
    
    for pattern, count in sorted(patterns.items()):
        print(f'  Pattern T_{pattern}: {count} files')
    print()
    
    # Analyze a few sample CSV files
    sample_files = csv_files[:3]
    for csv_file in sample_files:
        print(f'=== {csv_file.name} ===')
        try:
            df = pd.read_csv(csv_file)
            print(f'Shape: {df.shape}')
            print(f'Columns: {list(df.columns)}')
            print(f'File size: {csv_file.stat().st_size / 1024:.2f} KB')
            
            if len(df) > 0:
                print('First 3 rows:')
                print(df.head(3))
                print()
                
                # Show data types and basic stats
                print('Column info:')
                for col in df.columns:
                    dtype = df[col].dtype
                    non_null = df[col].count()
                    unique = df[col].nunique()
                    print(f'  {col}: {dtype}, {non_null} non-null, {unique} unique')
                print()
                
        except Exception as e:
            print(f'Error reading {csv_file}: {e}')
        print('-' * 40)
        print()

def analyze_h5_files_basic():
    """Basic analysis of H5 files without requiring h5py"""
    print('=== H5 Files Basic Analysis ===')
    
    data_dir = Path('/home/data/MVImageNet')
    h5_files = list(data_dir.glob('*.h5'))
    
    print(f'Found {len(h5_files)} H5 files')
    print()
    
    # Group files by naming pattern
    data_files = [f for f in h5_files if f.name.startswith('data') and not f.name == 'data_all.h5']
    main_file = data_dir / 'data_all.h5'
    
    print(f'Main data file (data_all.h5): {"Found" if main_file.exists() else "Not found"}')
    if main_file.exists():
        print(f'  Size: {main_file.stat().st_size / 1024**2:.2f} MB')
    
    print(f'Individual data files: {len(data_files)}')
    print()
    
    # Show size distribution
    sizes = [f.stat().st_size / 1024**2 for f in h5_files]
    if sizes:
        print('H5 file size statistics (MB):')
        print(f'  Total: {sum(sizes):.2f} MB')
        print(f'  Average: {np.mean(sizes):.2f} MB')
        print(f'  Min: {min(sizes):.2f} MB')
        print(f'  Max: {max(sizes):.2f} MB')
        print()
        
        # Show largest files
        file_sizes = [(f.name, f.stat().st_size / 1024**2) for f in h5_files]
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        print('Largest H5 files:')
        for name, size in file_sizes[:5]:
            print(f'  {name}: {size:.2f} MB')
        print()

def analyze_masked_dataset():
    """Analyze the masked dataset directory"""
    print('=== Masked Dataset Analysis ===')
    
    data_dir = Path('/home/data/MVImageNet')
    masked_dir = data_dir / 'masked_dataset'
    
    if masked_dir.exists():
        masked_files = list(masked_dir.iterdir())
        print(f'Files in masked_dataset: {len(masked_files)}')
        print()
        
        # Analyze file types and patterns
        file_types = {}
        for f in masked_files:
            ext = f.suffix
            file_types[ext] = file_types.get(ext, 0) + 1
        
        print('File types in masked_dataset:')
        for ext, count in file_types.items():
            print(f'  {ext}: {count} files')
        print()
        
        # Show sample filenames to understand naming pattern
        print('Sample filenames:')
        for f in masked_files[:5]:
            size_mb = f.stat().st_size / 1024**2
            print(f'  {f.name} ({size_mb:.2f} MB)')
        print()
    else:
        print('Masked dataset directory not found')
        print()

def dataset_overview():
    """Provide general overview of the dataset directory"""
    print('=== MVImageNet Dataset Overview ===')
    print()
    
    data_dir = Path('/home/data/MVImageNet')
    
    # Count different file types
    h5_files = list(data_dir.glob('*.h5'))
    csv_files = list(data_dir.glob('*.csv'))
    parquet_files = list(data_dir.glob('*.parquet'))
    
    print(f'Dataset location: {data_dir}')
    print(f'Total H5 files: {len(h5_files)}')
    print(f'Total CSV files: {len(csv_files)}')
    print(f'Total Parquet files: {len(parquet_files)}')
    print()
    
    # Calculate total size
    try:
        total_size = sum(f.stat().st_size for f in data_dir.iterdir() if f.is_file())
        print(f'Total dataset size: {total_size / 1024**3:.2f} GB')
    except Exception as e:
        print(f'Could not calculate total size: {e}')
    
    print()

if __name__ == "__main__":
    dataset_overview()
    analyze_parquet_files()
    analyze_csv_files()
    analyze_h5_files_basic()
    analyze_masked_dataset()
    
    print('=== Analysis Complete ===')
    print()
    print('Dataset Summary:')
    print('- MVImageNet appears to be a multi-view image dataset')
    print('- Contains train/val/test splits in parquet format')
    print('- Multiple H5 files likely contain image data or features')
    print('- CSV files contain experimental parameters with different thresholds')
    print('- Masked dataset suggests some form of data masking/augmentation') 