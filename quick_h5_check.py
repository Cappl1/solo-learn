#!/usr/bin/env python3
import h5py
import pandas as pd
from pathlib import Path
import urllib.parse

def main():
    data_dir = Path('/home/data/MVImageNet')
    
    print("=== Quick H5 Structure Check ===")
    
    # Check data_all.h5 structure quickly
    main_h5 = data_dir / 'data_all.h5'
    if main_h5.exists():
        with h5py.File(main_h5, 'r') as f:
            keys = list(f.keys())
            print(f"data_all.h5 has {len(keys)} top-level keys")
            print(f"Sample keys: {keys[:3]}")
            
            # Check first key structure
            first_key = keys[0]
            print(f"\nStructure of '{first_key}':")
            item = f[first_key]
            
            if isinstance(item, h5py.Group):
                print(f"  Group with subkeys: {list(item.keys())}")
                # Check substructure
                for subkey in list(item.keys())[:2]:
                    subitem = item[subkey]
                    if isinstance(subitem, h5py.Dataset):
                        print(f"    {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
            elif isinstance(item, h5py.Dataset):
                print(f"  Dataset: shape={item.shape}, dtype={item.dtype}")
    
    # Check parquet to H5 mapping
    print("\n=== Parquet-H5 Mapping ===")
    val_parquet = data_dir / 'dataset_val_all3.parquet'
    
    if val_parquet.exists():
        df = pd.read_parquet(val_parquet, engine='fastparquet')
        print(f"Validation set: {df.shape[0]} samples")
        
        # Analyze path structure
        sample_paths = df['path'].head(10).tolist()
        print(f"Sample paths:")
        for path in sample_paths:
            print(f"  {path}")
        
        # Check if partition column maps to H5 keys
        partitions = df['partition'].unique()[:5]
        print(f"\nUnique partitions: {partitions}")
        
        # URL decode and check if they match H5 keys
        with h5py.File(main_h5, 'r') as f:
            h5_keys = list(f.keys())[:5]
            print(f"\nH5 keys (decoded):")
            for key in h5_keys:
                decoded = urllib.parse.unquote(key)
                print(f"  {key} -> {decoded}")
            
            # Check if any partitions match decoded keys
            print(f"\nChecking partition matches:")
            for partition in partitions:
                for h5_key in h5_keys:
                    decoded = urllib.parse.unquote(h5_key)
                    if str(partition) == decoded or str(partition) in decoded:
                        print(f"  MATCH: partition '{partition}' matches H5 key '{decoded}'")

if __name__ == "__main__":
    main() 