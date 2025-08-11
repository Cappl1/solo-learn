#!/usr/bin/env python3
"""
Optimize MVImageNet Dataset

1. Build category → H5 file mapping
2. Convert to optimized parquet format
3. Create fast dataset loader
"""

import pandas as pd
import h5py
import numpy as np
import os
import glob
import json
import time
from collections import defaultdict

def build_h5_category_mapping(h5_path="/home/data/MVImageNet", save_mapping=True):
    """Build and save category → H5 file mapping"""
    print("🗺️ BUILDING H5 CATEGORY MAPPING")
    print("=" * 60)
    
    h5_files = glob.glob(os.path.join(h5_path, "data*.h5"))
    h5_files = [f for f in h5_files if 'data_all.h5' not in f]
    
    print(f"Scanning {len(h5_files)} H5 files...")
    
    category_to_h5 = {}
    h5_stats = {}
    
    for i, h5_file in enumerate(h5_files):
        filename = os.path.basename(h5_file)
        print(f"[{i+1}/{len(h5_files)}] Scanning {filename}...")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                categories = list(f.keys())
                objects_count = 0
                
                # Count total objects
                for cat in categories:
                    try:
                        objects_count += len(list(f[cat].keys()))
                    except:
                        pass
                
                # Add to mapping
                for cat in categories:
                    category_to_h5[cat] = h5_file
                
                h5_stats[filename] = {
                    'categories': len(categories),
                    'category_range': f"{min(categories)}-{max(categories)}" if categories else "empty",
                    'objects': objects_count,
                    'size_gb': os.path.getsize(h5_file) / (1024**3)
                }
                
                print(f"  Categories: {len(categories)}, Objects: {objects_count}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            h5_stats[filename] = {'error': str(e)}
    
    print(f"\n📊 MAPPING RESULTS:")
    print(f"Total categories mapped: {len(category_to_h5)}")
    print(f"Category range: {min(category_to_h5.keys())} to {max(category_to_h5.keys())}")
    
    # Show distribution
    h5_distribution = defaultdict(int)
    for h5_file in category_to_h5.values():
        h5_distribution[os.path.basename(h5_file)] += 1
    
    print(f"\nCategories per H5 file:")
    for h5_file, count in sorted(h5_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {h5_file}: {count} categories")
    
    if save_mapping:
        # Save mapping as JSON
        mapping_file = "h5_category_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(category_to_h5, f, indent=2)
        print(f"\n💾 Saved mapping to {mapping_file}")
        
        # Save stats
        stats_file = "h5_file_stats.json" 
        with open(stats_file, 'w') as f:
            json.dump(h5_stats, f, indent=2)
        print(f"💾 Saved stats to {stats_file}")
    
    return category_to_h5, h5_stats

def optimize_dataset_to_parquet(csv_file="unified_mvimagenet_train.csv", 
                               output_file="optimized_mvimagenet_train.parquet"):
    """Convert CSV to optimized parquet with only necessary columns"""
    print(f"\n⚡ OPTIMIZING DATASET TO PARQUET")
    print("=" * 60)
    
    print(f"Loading {csv_file}...")
    start_time = time.time()
    df = pd.read_csv(csv_file)
    load_time = time.time() - start_time
    print(f"CSV loaded in {load_time:.2f}s: {len(df):,} rows")
    
    # Parse paths for faster access
    print("Parsing paths...")
    path_parts = df['path'].str.split('/')
    df['category_str'] = path_parts.str[0]  # Keep as string to match H5 keys
    df['object_id'] = path_parts.str[1]
    df['frame_idx'] = path_parts.str[3].str.replace('.jpg', '').astype(int) - 1
    
    # Select only necessary columns
    essential_columns = [
        'path',           # Original path for reference
        'category_str',   # Parsed category (string)
        'object_id',      # Parsed object ID  
        'frame_idx',      # Parsed frame index (0-based)
        'category',       # Original category (int)
        'object',         # Original object 
        'frame',          # Original frame
        'q0', 'q1', 'q2', 'q3',  # Quaternion
        't0', 't1', 't2'  # Translation
    ]
    
    # Filter to only essential columns that exist
    available_columns = [col for col in essential_columns if col in df.columns]
    df_optimized = df[available_columns].copy()
    
    print(f"Optimized columns: {available_columns}")
    print(f"Reduced from {len(df.columns)} to {len(available_columns)} columns")
    
    # Save as parquet
    print(f"Saving to {output_file}...")
    start_time = time.time()
    df_optimized.to_parquet(output_file, compression='snappy', index=False)
    save_time = time.time() - start_time
    
    # Compare file sizes
    csv_size = os.path.getsize(csv_file) / (1024**2)
    parquet_size = os.path.getsize(output_file) / (1024**2)
    compression_ratio = csv_size / parquet_size
    
    print(f"✅ Saved in {save_time:.2f}s")
    print(f"📊 Size comparison:")
    print(f"  CSV: {csv_size:.1f}MB")
    print(f"  Parquet: {parquet_size:.1f}MB")
    print(f"  Compression: {compression_ratio:.1f}x smaller")
    
    # Test loading speed
    print(f"\n🚀 Speed test:")
    
    # CSV loading
    start_time = time.time()
    df_csv_test = pd.read_csv(csv_file, nrows=100000)
    csv_load_time = time.time() - start_time
    
    # Parquet loading  
    start_time = time.time()
    df_parquet_test = pd.read_parquet(output_file)
    if len(df_parquet_test) > 100000:
        df_parquet_test = df_parquet_test.head(100000)
    parquet_load_time = time.time() - start_time
    
    speedup = csv_load_time / parquet_load_time
    print(f"  CSV (100K rows): {csv_load_time:.2f}s")
    print(f"  Parquet (100K rows): {parquet_load_time:.2f}s") 
    print(f"  Speedup: {speedup:.1f}x faster")
    
    return df_optimized

def create_optimized_dataset_loader():
    """Create the optimized dataset loader"""
    print(f"\n💻 OPTIMIZED DATASET LOADER")
    print("=" * 60)
    
    loader_code = '''
import pandas as pd
import h5py
import numpy as np
import cv2
import json
import os
from torch.utils.data import Dataset

class OptimizedMVImageNetDataset(Dataset):
    def __init__(self, parquet_file="optimized_mvimagenet_train.parquet", 
                 mapping_file="h5_category_mapping.json", 
                 transform=None):
        """
        Optimized MVImageNet Dataset Loader
        
        Args:
            parquet_file: Path to optimized parquet file
            mapping_file: Path to H5 category mapping JSON
            transform: Optional image transforms
        """
        
        print(f"Loading optimized dataset...")
        start_time = time.time()
        
        # Load optimized parquet (much faster than CSV!)
        self.df = pd.read_parquet(parquet_file)
        load_time = time.time() - start_time
        print(f"✅ Loaded {len(self.df):,} samples in {load_time:.2f}s")
        
        # Load pre-built H5 mapping (no scanning needed!)
        with open(mapping_file, 'r') as f:
            self.category_to_h5 = json.load(f)
        print(f"✅ Loaded mapping for {len(self.category_to_h5)} categories")
        
        # Filter to only loadable samples
        loadable = self.df['category_str'].isin(self.category_to_h5.keys())
        original_count = len(self.df)
        self.df = self.df[loadable].reset_index(drop=True)
        
        print(f"✅ Filtered to {len(self.df):,} loadable samples ({len(self.df)/original_count:.1%})")
        
        self.transform = transform
    
    def _decode_image(self, data_bytes):
        """Fast JPEG decoding"""
        try:
            img_array = np.frombuffer(data_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Direct access to parsed values (no string processing!)
        category_str = row['category_str']
        object_id = row['object_id'] 
        frame_idx = row['frame_idx']
        
        # Fast H5 file lookup
        h5_file = self.category_to_h5[category_str]
        
        try:
            with h5py.File(h5_file, 'r') as f:
                data_bytes = f[category_str][object_id][frame_idx]
                image = self._decode_image(data_bytes)
        except Exception as e:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'pose': {
                'quaternion': np.array([row['q0'], row['q1'], row['q2'], row['q3']]),
                'translation': np.array([row['t0'], row['t1'], row['t2']])
            },
            'category': int(row['category']),
            'object': row['object'],
            'frame': int(row['frame']),
            'path': row['path']
        }

# Usage:
if __name__ == "__main__":
    import time
    
    # Create optimized dataset
    dataset = OptimizedMVImageNetDataset()
    
    # Speed test
    print(f"\n🚀 Performance test:")
    start_time = time.time()
    
    # Load 100 samples
    for i in range(100):
        sample = dataset[i]
    
    elapsed = time.time() - start_time
    print(f"Loaded 100 samples in {elapsed:.2f}s ({elapsed/100*1000:.1f}ms per sample)")
    
    print(f"\n✅ Dataset ready for high-performance training!")
'''
    
    with open("OptimizedMVImageNetDataset.py", "w") as f:
        f.write(loader_code)
    
    print("💾 Saved optimized dataset loader to OptimizedMVImageNetDataset.py")
    print(loader_code)

def main():
    print("🚀 OPTIMIZING MVImageNet DATASET")
    print("=" * 80)
    
    # Step 1: Build H5 mapping (one-time setup)
    mapping_file = "h5_category_mapping.json"
    if not os.path.exists(mapping_file):
        print("Building H5 category mapping (one-time setup)...")
        category_to_h5, h5_stats = build_h5_category_mapping()
    else:
        print(f"✅ Using existing mapping: {mapping_file}")
    
    # Step 2: Optimize dataset to parquet
    parquet_file = "optimized_mvimagenet_train.parquet"
    if not os.path.exists(parquet_file):
        df_optimized = optimize_dataset_to_parquet()
    else:
        print(f"✅ Using existing optimized dataset: {parquet_file}")
    
    # Step 3: Create optimized loader
    create_optimized_dataset_loader()
    
    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print(f"✅ H5 mapping: {mapping_file}")
    print(f"✅ Optimized dataset: {parquet_file}")
    print(f"✅ Fast loader: OptimizedMVImageNetDataset.py")
    print(f"\n🚀 Ready for high-performance training!")

if __name__ == "__main__":
    main() 