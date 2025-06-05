#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    data_dir = Path('/home/data/MVImageNet')
    
    print("=== Investigating CSV Files ===")
    
    # Get all CSV files
    csv_files = list(data_dir.glob('datasetT_*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    # Analyze file naming patterns in detail
    print(f"\n=== CSV File Patterns ===")
    patterns = {}
    for csv_file in csv_files:
        name_parts = csv_file.stem.split('_')
        if len(name_parts) >= 3:
            threshold = name_parts[1]
            param2 = name_parts[2]
            param3 = name_parts[3] if len(name_parts) > 3 else 'none'
            pattern = f'{threshold}_{param2}_{param3}'
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(csv_file.name)
    
    for pattern, files in sorted(patterns.items()):
        print(f"  Pattern T_{pattern}: {len(files)} files")
        if len(files) <= 3:
            for f in files:
                print(f"    {f}")
        else:
            print(f"    {files[0]}, {files[1]}, ... (+{len(files)-2} more)")
    
    # Try to read a few CSV files with different patterns
    print(f"\n=== Sample CSV File Contents ===")
    
    # Pick some representative files
    test_files = []
    for pattern, files in list(patterns.items())[:5]:
        test_files.append(data_dir / files[0])
    
    for csv_file in test_files:
        print(f"\n--- {csv_file.name} ---")
        try:
            # Try different ways to read the CSV
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            if len(df) > 0:
                print(f"  Sample rows:")
                print(df.head(3).to_string(index=False))
                
                # Check if it has the key columns we need
                key_columns = ['partition', 'category', 'object', 'frame', 'path']
                has_columns = [col for col in key_columns if col in df.columns]
                print(f"  Has key columns: {has_columns}")
                
                # Check for any column that might indicate train/val/test split
                split_indicators = ['split', 'train', 'val', 'test', 'set']
                split_columns = [col for col in df.columns if any(ind in col.lower() for ind in split_indicators)]
                if split_columns:
                    print(f"  Split-related columns: {split_columns}")
                    for col in split_columns:
                        if df[col].dtype == 'object' or df[col].nunique() < 10:
                            print(f"    {col}: {df[col].unique()}")
                
        except Exception as e:
            print(f"  ❌ Error reading: {e}")
            
            # Try with different parameters
            try:
                df = pd.read_csv(csv_file, sep=';', on_bad_lines='skip')
                print(f"  ✓ Success with semicolon separator")
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
            except:
                try:
                    df = pd.read_csv(csv_file, sep='\t', on_bad_lines='skip')
                    print(f"  ✓ Success with tab separator")
                    print(f"    Shape: {df.shape}")
                    print(f"    Columns: {list(df.columns)}")
                except:
                    print(f"  ❌ Failed with multiple separators")
    
    # Look for the largest/most complete CSV files
    print(f"\n=== Largest CSV Files ===")
    csv_sizes = [(f, f.stat().st_size) for f in csv_files]
    csv_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 largest CSV files:")
    for csv_file, size in csv_sizes[:5]:
        size_kb = size / 1024
        print(f"  {csv_file.name}: {size_kb:.1f} KB")
        
        # Try to quickly check what's in the largest files
        try:
            df = pd.read_csv(csv_file, nrows=5, on_bad_lines='skip')
            print(f"    Columns: {list(df.columns)}")
        except:
            print(f"    Could not read preview")
    
    # Look for any CSV that might contain full dataset info
    print(f"\n=== Looking for Complete Dataset CSVs ===")
    for csv_file in csv_sizes[:10]:  # Check top 10 largest
        file_path, size = csv_file
        if size > 50 * 1024:  # Files larger than 50KB
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip')
                if len(df) > 100000:  # Large number of rows
                    print(f"✓ Large dataset found: {file_path.name}")
                    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
                    print(f"  Columns: {list(df.columns)}")
                    
                    # Check if this could be training data
                    if len(df) > 500000:  # Potentially training set size
                        print(f"  🎯 Potential training set candidate!")
                        
                        # Quick analysis
                        if 'partition' in df.columns:
                            partitions = df['partition'].nunique()
                            print(f"    Partitions: {partitions}")
                        if 'object' in df.columns:
                            objects = df['object'].nunique()
                            print(f"    Objects: {objects}")
                        if 'category' in df.columns:
                            categories = df['category'].nunique()
                            print(f"    Categories: {categories}")
                            
            except Exception as e:
                continue

if __name__ == "__main__":
    main() 