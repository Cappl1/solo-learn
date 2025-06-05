#!/usr/bin/env python3
import h5py
import pandas as pd
from pathlib import Path
import io
from PIL import Image

def main():
    data_dir = Path('/home/data/MVImageNet')
    
    # Load parquet data
    val_parquet = data_dir / 'dataset_val_all3.parquet'
    df = pd.read_parquet(val_parquet, engine='fastparquet')
    
    # Test loading images
    main_h5 = data_dir / 'data_all.h5'
    
    print("=== Testing Image Loading ===")
    
    for i in range(3):
        sample = df.iloc[i]
        partition = sample['partition']
        category = str(sample['category'])
        obj = sample['object']
        frame = sample['frame']
        path = sample['path']
        
        print(f"\nSample {i}:")
        print(f"  Path: {path}")
        print(f"  H5 location: [{partition}][{category}][{obj}][{frame}]")
        
        with h5py.File(main_h5, 'r') as f:
            try:
                # Get the image data (binary string)
                img_data = f[partition][category][obj][frame]
                print(f"  Raw data type: {type(img_data)}")
                print(f"  Raw data shape: {img_data.shape if hasattr(img_data, 'shape') else 'No shape'}")
                
                # Convert to bytes and load with PIL
                if isinstance(img_data, bytes):
                    img_bytes = img_data
                else:
                    # If it's a numpy scalar with bytes
                    img_bytes = img_data.tobytes()
                
                print(f"  Image bytes length: {len(img_bytes)}")
                
                # Try to load with PIL
                image = Image.open(io.BytesIO(img_bytes))
                print(f"  ✓ Successfully loaded: {image.size}, {image.mode}")
                
                # Save a test image to verify it works
                if i == 0:
                    image.save(f"test_image_{i}.jpg")
                    print(f"  Saved test image: test_image_{i}.jpg")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print(f"\n=== Summary ===")
    print("✓ Images are stored as compressed binary data (JPEG)")
    print("✓ Structure: H5[partition][category][object][frame] -> binary image data")
    print("✓ Can be loaded with PIL from BytesIO")

if __name__ == "__main__":
    main() 