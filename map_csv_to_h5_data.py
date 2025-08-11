#!/usr/bin/env python3
"""
Map CSV Metadata to H5 Image Data

This script shows how to properly use the CSV metadata files to access
the actual image data stored in the H5 files.
"""

import pandas as pd
import h5py
import numpy as np
import os
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def analyze_h5_structure(data_path="/home/data/MVImageNet"):
    """Analyze the structure of H5 files"""
    print("=" * 80)
    print("ANALYZING H5 FILE STRUCTURE")
    print("=" * 80)
    
    # Find all H5 files
    h5_files = glob.glob(os.path.join(data_path, "data*.h5"))
    h5_files = [f for f in h5_files if 'data_all.h5' not in f]  # Skip corrupted main file
    
    print(f"Found {len(h5_files)} H5 files:")
    for h5_file in h5_files:
        size_gb = os.path.getsize(h5_file) / (1024**3)
        print(f"  {os.path.basename(h5_file)}: {size_gb:.1f}GB")
    
    # Analyze structure of first few H5 files
    h5_structure = {}
    
    for i, h5_file in enumerate(h5_files[:3]):  # Check first 3 files
        print(f"\n📁 ANALYZING {os.path.basename(h5_file)}")
        print("-" * 50)
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # Get top-level categories
                categories = list(f.keys())
                print(f"Categories in this H5: {len(categories)}")
                print(f"Category range: {min(categories)} to {max(categories)}")
                
                # Sample a few categories to understand structure
                sample_categories = categories[:3]
                for cat in sample_categories:
                    cat_group = f[cat]
                    objects = list(cat_group.keys())
                    print(f"\n  Category {cat}:")
                    print(f"    Objects: {len(objects)}")
                    
                    # Look at first object
                    if objects:
                        first_obj = objects[0]
                        obj_group = cat_group[first_obj]
                        
                        if 'images' in obj_group:
                            images = obj_group['images']
                            print(f"    Sample object {first_obj}: {images.shape} images")
                            print(f"    Image dtype: {images.dtype}")
                            if len(images.shape) == 4:  # (n_frames, height, width, channels)
                                print(f"    Frame count: {images.shape[0]}")
                                print(f"    Image size: {images.shape[1]}x{images.shape[2]}")
                                print(f"    Channels: {images.shape[3]}")
                        
                        # Check for other data
                        other_keys = [k for k in obj_group.keys() if k != 'images']
                        if other_keys:
                            print(f"    Other data: {other_keys}")
                
                h5_structure[h5_file] = {
                    'categories': categories,
                    'num_categories': len(categories),
                    'category_range': (min(categories), max(categories))
                }
                
        except Exception as e:
            print(f"❌ Error reading {h5_file}: {e}")
    
    return h5_structure

def map_csv_paths_to_h5(csv_path, h5_structure):
    """Map CSV paths to H5 file locations"""
    print(f"\n🗺️  MAPPING CSV PATHS TO H5 FILES")
    print("-" * 50)
    
    # Load a sample from the unified CSV
    df = pd.read_csv(csv_path, nrows=10000)  # Load sample for analysis
    print(f"Loaded {len(df)} sample rows from CSV")
    
    # Analyze path structure
    print(f"\nPath structure analysis:")
    sample_paths = df['path'].head(10).tolist()
    print(f"Sample paths:")
    for i, path in enumerate(sample_paths):
        print(f"  {i+1}. {path}")
    
    # Extract category and object from paths
    df['path_category'] = df['path'].str.split('/').str[0].astype(int)
    df['path_object'] = df['path'].str.split('/').str[1]
    df['path_frame'] = df['path'].str.split('/').str[2].str.replace('images/', '').str.replace('.jpg', '').astype(int) - 1  # Convert to 0-based
    
    # Map categories to H5 files
    path_categories = df['path_category'].unique()
    print(f"\nCategories in CSV sample: {len(path_categories)}")
    print(f"Category range: {min(path_categories)} to {max(path_categories)}")
    
    # Find which H5 files contain which categories
    h5_mapping = {}
    for h5_file, structure in h5_structure.items():
        h5_categories = set(map(int, structure['categories']))
        csv_categories = set(path_categories)
        overlap = h5_categories.intersection(csv_categories)
        
        if overlap:
            h5_mapping[h5_file] = {
                'categories': sorted(overlap),
                'num_matches': len(overlap)
            }
            print(f"\n📂 {os.path.basename(h5_file)}:")
            print(f"   Matching categories: {len(overlap)}")
            print(f"   Category range: {min(overlap)} to {max(overlap)}")
    
    return h5_mapping, df

def demonstrate_data_loading(csv_df, h5_mapping, num_samples=5):
    """Demonstrate how to load actual image data using CSV metadata"""
    print(f"\n🖼️  DEMONSTRATING DATA LOADING")
    print("-" * 50)
    
    successful_loads = 0
    
    for i, row in csv_df.head(num_samples).iterrows():
        path = row['path']
        category = int(row['path_category'])
        object_id = row['path_object']
        frame_idx = int(row['path_frame'])
        
        print(f"\nSample {i+1}: {path}")
        print(f"  Category: {category}, Object: {object_id}, Frame: {frame_idx}")
        
        # Find which H5 file contains this category
        target_h5 = None
        for h5_file, mapping in h5_mapping.items():
            if category in mapping['categories']:
                target_h5 = h5_file
                break
        
        if target_h5 is None:
            print(f"  ❌ No H5 file found for category {category}")
            continue
        
        print(f"  📂 H5 file: {os.path.basename(target_h5)}")
        
        # Try to load the actual image
        try:
            with h5py.File(target_h5, 'r') as f:
                if str(category) in f:
                    cat_group = f[str(category)]
                    if object_id in cat_group:
                        obj_group = cat_group[object_id]
                        if 'images' in obj_group:
                            images = obj_group['images']
                            if frame_idx < images.shape[0]:
                                image = images[frame_idx]
                                print(f"  ✅ Image loaded: shape {image.shape}, dtype {image.dtype}")
                                print(f"     Value range: {image.min():.2f} to {image.max():.2f}")
                                successful_loads += 1
                            else:
                                print(f"  ❌ Frame {frame_idx} out of range (max: {images.shape[0]-1})")
                        else:
                            print(f"  ❌ No 'images' data in object {object_id}")
                    else:
                        print(f"  ❌ Object {object_id} not found in category {category}")
                else:
                    print(f"  ❌ Category {category} not found in H5 file")
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    
    print(f"\n📊 Loading success rate: {successful_loads}/{num_samples} ({successful_loads/num_samples:.1%})")

def create_dataset_loader_example():
    """Create an example dataset loader class"""
    print(f"\n💻 EXAMPLE DATASET LOADER CODE")
    print("-" * 50)
    
    loader_code = '''
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset
import cv2

class MVImageNetDataset(Dataset):
    def __init__(self, csv_file, h5_path="/home/data/MVImageNet", transform=None):
        """
        MVImageNet Dataset Loader
        
        Args:
            csv_file: Path to unified CSV metadata file
            h5_path: Path to directory containing H5 files
            transform: Optional transforms
        """
        self.df = pd.read_csv(csv_file)
        self.h5_path = h5_path
        self.transform = transform
        
        # Preprocess paths
        self.df['path_category'] = self.df['path'].str.split('/').str[0].astype(int)
        self.df['path_object'] = self.df['path'].str.split('/').str[1]
        self.df['path_frame'] = self.df['path'].str.split('/').str[2].str.replace('images/', '').str.replace('.jpg', '').astype(int) - 1
        
        # Map categories to H5 files (you'd need to implement this mapping)
        self.category_to_h5 = self._build_category_mapping()
    
    def _build_category_mapping(self):
        """Build mapping from categories to H5 files"""
        # This would scan H5 files and build the mapping
        # For now, return a placeholder
        return {}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Extract metadata
        category = int(row['path_category'])
        object_id = row['path_object']
        frame_idx = int(row['path_frame'])
        
        # Get pose data (quaternion + translation)
        pose = {
            'q': np.array([row['q0'], row['q1'], row['q2'], row['q3']]),
            't': np.array([row['t0'], row['t1'], row['t2']])
        }
        
        # Load image from H5
        h5_file = self.category_to_h5.get(category)
        if h5_file is None:
            raise ValueError(f"No H5 file for category {category}")
        
        with h5py.File(h5_file, 'r') as f:
            image = f[str(category)][object_id]['images'][frame_idx]
        
        # Convert to RGB if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'pose': pose,
            'category': category,
            'object': object_id,
            'frame': frame_idx,
            'path': row['path']
        }

# Usage example:
# dataset = MVImageNetDataset('unified_mvimagenet_train.csv')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
'''
    
    print(loader_code)

def main():
    """Main analysis function"""
    print("Starting CSV to H5 mapping analysis...")
    
    # Analyze H5 structure
    h5_structure = analyze_h5_structure()
    
    if not h5_structure:
        print("❌ No H5 files found or readable")
        return
    
    # Map CSV paths to H5 files
    csv_path = "unified_mvimagenet_train.csv"
    if os.path.exists(csv_path):
        h5_mapping, csv_df = map_csv_paths_to_h5(csv_path, h5_structure)
        
        # Demonstrate data loading
        demonstrate_data_loading(csv_df, h5_mapping)
        
        # Show example loader code
        create_dataset_loader_example()
        
        print(f"\n🎯 SUMMARY:")
        print(f"  - CSV contains metadata + logical paths")
        print(f"  - H5 files contain actual image data")
        print(f"  - Need to map category → H5 file → object → frame")
        print(f"  - Use CSV for training metadata, H5 for images")
        
    else:
        print(f"❌ CSV file not found: {csv_path}")

if __name__ == "__main__":
    main() 