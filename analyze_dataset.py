#!/usr/bin/env python3
"""
MVImageNet Dataset Analysis Script
"""

import h5py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_parquet_files():
    """Analyze the main parquet dataset files"""
    print('=== MVImageNet Dataset Analysis ===')
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
            df = pd.read_parquet(file_path)
            
            print(f'Shape: {df.shape}')
            print(f'Columns: {list(df.columns)}')
            print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
            print()
            
            print('Data types:')
            print(df.dtypes)
            print()
            
            if len(df) > 0:
                print('First few rows:')
                print(df.head())
                print()
                
                # Statistical summary for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    print('Statistical summary for numeric columns:')
                    print(df[numeric_cols].describe())
                    print()
                
                # Value counts for categorical columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    if col in df.columns:
                        unique_count = df[col].nunique()
                        print(f'{col}: {unique_count} unique values')
                        if unique_count <= 20:
                            print(df[col].value_counts().head(10))
                        print()
            
            dataset_info[filename] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': dict(df.dtypes)
            }
            
            print('-' * 60)
            print()
    
    return dataset_info

def analyze_h5_files():
    """Analyze H5 files in the dataset"""
    print('=== H5 Files Analysis ===')
    
    data_dir = Path('/home/data/MVImageNet')
    
    # Check data_all.h5 first
    main_h5 = data_dir / 'data_all.h5'
    if main_h5.exists():
        print(f'=== {main_h5.name} ===')
        try:
            with h5py.File(main_h5, 'r') as f:
                print(f'Keys in H5 file: {list(f.keys())}')
                print(f'File size: {main_h5.stat().st_size / 1024**2:.2f} MB')
                print()
                
                for key in list(f.keys())[:10]:  # Show first 10 keys
                    dataset = f[key]
                    print(f'{key}:')
                    print(f'  Shape: {dataset.shape}')
                    print(f'  Dtype: {dataset.dtype}')
                    if hasattr(dataset, 'attrs'):
                        attrs = dict(dataset.attrs)
                        if attrs:
                            print(f'  Attributes: {attrs}')
                    print()
        except Exception as e:
            print(f'Error reading {main_h5}: {e}')
        
        print('-' * 60)
        print()
    
    # Check a few other H5 files
    h5_files = list(data_dir.glob('data*.h5'))[:3]  # Check first 3 H5 files
    for h5_file in h5_files:
        if h5_file.name != 'data_all.h5':
            print(f'=== {h5_file.name} ===')
            try:
                print(f'File size: {h5_file.stat().st_size / 1024**2:.2f} MB')
                with h5py.File(h5_file, 'r') as f:
                    print(f'Keys: {list(f.keys())[:5]}...')  # Show first 5 keys
                    if len(f.keys()) > 0:
                        first_key = list(f.keys())[0]
                        print(f'Sample dataset shape: {f[first_key].shape}')
                        print(f'Sample dataset dtype: {f[first_key].dtype}')
                print()
            except Exception as e:
                print(f'Error reading {h5_file}: {e}')
                print()

def analyze_csv_files():
    """Analyze CSV files with dataset parameters"""
    print('=== CSV Files Analysis ===')
    
    data_dir = Path('/home/data/MVImageNet')
    csv_files = list(data_dir.glob('datasetT_*.csv'))[:5]  # Check first 5 CSV files
    
    for csv_file in csv_files:
        print(f'=== {csv_file.name} ===')
        try:
            df = pd.read_csv(csv_file)
            print(f'Shape: {df.shape}')
            print(f'Columns: {list(df.columns)}')
            if len(df) > 0:
                print('First few rows:')
                print(df.head(3))
            print()
        except Exception as e:
            print(f'Error reading {csv_file}: {e}')
            print()

def dataset_overview():
    """Provide general overview of the dataset directory"""
    print('=== Dataset Directory Overview ===')
    
    data_dir = Path('/home/data/MVImageNet')
    
    # Count different file types
    h5_files = list(data_dir.glob('*.h5'))
    csv_files = list(data_dir.glob('*.csv'))
    parquet_files = list(data_dir.glob('*.parquet'))
    
    print(f'Total H5 files: {len(h5_files)}')
    print(f'Total CSV files: {len(csv_files)}')
    print(f'Total Parquet files: {len(parquet_files)}')
    print()
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in data_dir.iterdir() if f.is_file())
    print(f'Total dataset size: {total_size / 1024**3:.2f} GB')
    print()
    
    # Check if masked_dataset directory exists
    masked_dir = data_dir / 'masked_dataset'
    if masked_dir.exists():
        print('Masked dataset directory found:')
        masked_files = list(masked_dir.iterdir())
        print(f'  Files in masked_dataset: {len(masked_files)}')
        if masked_files:
            print(f'  Sample files: {[f.name for f in masked_files[:3]]}')
        print()

if __name__ == "__main__":
    dataset_overview()
    dataset_info = analyze_parquet_files()
    analyze_h5_files()
    analyze_csv_files()
    
    print('=== Analysis Complete ===')
    print('The dataset appears to be MVImageNet with multiple data splits and formats.')
    print('Main files: parquet for structured data, H5 for image/array data, CSV for parameters.') 