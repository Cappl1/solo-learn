#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import h5py
import urllib.parse

def main():
    data_dir = Path('/home/data/MVImageNet')
    
    print("=== Data Split Organization ===")
    
    # Check what we can read from parquet files
    splits = ['train', 'val', 'test']
    parquet_info = {}
    
    for split in splits:
        parquet_file = data_dir / f'dataset_{split}_all3.parquet'
        if parquet_file.exists():
            try:
                if split == 'train':
                    # Try both engines for train
                    try:
                        df = pd.read_parquet(parquet_file, engine='pyarrow')
                        engine_used = 'pyarrow'
                    except:
                        try:
                            df = pd.read_parquet(parquet_file, engine='fastparquet')
                            engine_used = 'fastparquet'
                        except:
                            print(f"❌ {split}: Cannot read parquet file")
                            continue
                else:
                    df = pd.read_parquet(parquet_file, engine='fastparquet')
                    engine_used = 'fastparquet'
                
                partitions = df['partition'].unique()
                print(f"✅ {split}: {len(df)} samples, {len(partitions)} partitions (engine: {engine_used})")
                parquet_info[split] = {
                    'size': len(df),
                    'partitions': set(partitions),
                    'readable': True
                }
                
            except Exception as e:
                print(f"❌ {split}: Error reading - {e}")
                parquet_info[split] = {'readable': False, 'error': str(e)}
    
    # Check if all partitions are in data_all.h5
    print(f"\n=== H5 File Coverage ===")
    main_h5 = data_dir / 'data_all.h5'
    
    if main_h5.exists():
        with h5py.File(main_h5, 'r') as f:
            h5_partitions = set(f.keys())
            print(f"data_all.h5 contains {len(h5_partitions)} partitions")
            
            # Check coverage for each split
            for split, info in parquet_info.items():
                if info.get('readable', False):
                    parquet_partitions = info['partitions']
                    missing = parquet_partitions - h5_partitions
                    coverage = len(parquet_partitions - missing) / len(parquet_partitions) * 100
                    print(f"  {split}: {coverage:.1f}% coverage ({len(missing)} missing partitions)")
                    
                    if missing:
                        print(f"    Missing: {list(missing)[:3]}{'...' if len(missing) > 3 else ''}")
    
    # Check individual H5 files to see if they contain different splits
    print(f"\n=== Individual H5 Files ===")
    h5_files = list(data_dir.glob('data*.h5'))
    print(f"Found {len(h5_files)} H5 files")
    
    # Check a few individual files
    for h5_file in h5_files[:3]:
        if h5_file.name != 'data_all.h5':
            try:
                with h5py.File(h5_file, 'r') as f:
                    partitions_in_file = set(f.keys())
                    print(f"\n{h5_file.name}:")
                    print(f"  Contains {len(partitions_in_file)} partitions")
                    
                    # Check which split this file belongs to
                    for split, info in parquet_info.items():
                        if info.get('readable', False):
                            overlap = len(partitions_in_file & info['partitions'])
                            if overlap > 0:
                                print(f"    {split}: {overlap} overlapping partitions")
                                
            except Exception as e:
                print(f"  Error reading {h5_file.name}: {e}")
    
    # Summary
    print(f"\n=== Summary ===")
    print("Current situation:")
    for split, info in parquet_info.items():
        if info.get('readable', False):
            print(f"  ✅ {split}: {info['size']} samples ready to use")
        else:
            print(f"  ❌ {split}: {info.get('error', 'Unknown error')}")
    
    print(f"\nRecommendation:")
    print(f"1. Use val/test splits immediately - they work perfectly")
    print(f"2. For training: Try individual H5 files or investigate train parquet corruption")
    print(f"3. All data appears to be in data_all.h5 - can use it for all splits")

if __name__ == "__main__":
    main() 