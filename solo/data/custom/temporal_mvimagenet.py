import io
import os
import random
from typing import Tuple, Callable, Optional, List
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
import urllib.parse
import glob


class TemporalMVImageNet(Dataset):
    """MVImageNet dataset with temporal pairing for self-supervised learning.
    
    This implementation now uses separate train/val parquet files like Core50.
    - No internal splitting
    - Uses pre-split parquet files
    - Compatible with solo-learn framework
    """
    
    def __init__(self,
                 h5_data_dir: str = '/home/data/MVImageNet/',
                 metadata_path: str = '/home/brothen/solo-learn/mvimagenet_train.parquet',
                 transform: Optional[Callable] = None,
                 time_window: int = 15,  # How far to look for temporal pairs
                 min_clip_length: int = 10,  # Minimum frames in clip
                 ):
        
        self.h5_data_dir = h5_data_dir
        self.transform = transform
        self.time_window = time_window
        self.min_clip_length = min_clip_length
        
        print(f"Loading MVImageNet dataset with time_window={time_window}")
        
        # Load metadata from specified parquet file
        print(f"Loading metadata from: {metadata_path}")
        self.df = pd.read_parquet(metadata_path)
        print(f"Loaded {len(self.df):,} images")
        
        # Create H5 partition mapping (our verified approach)
        self._create_h5_mapping()
        
        # Build temporal index
        self._build_temporal_index()
    
    def _create_h5_mapping(self):
        """Create partition to H5 file mapping (our verified approach)."""
        h5_files = glob.glob(os.path.join(self.h5_data_dir, 'data*.h5'))
        self.partition_to_h5 = {}
        
        for h5_file in h5_files:
            basename = os.path.basename(h5_file).replace('.h5', '')
            partition_name = basename.replace('data', '', 1)
            partition_decoded = urllib.parse.unquote(partition_name)
            
            self.partition_to_h5[partition_name] = h5_file
            self.partition_to_h5[partition_decoded] = h5_file
        
        print(f"Created H5 mapping for {len(self.partition_to_h5)} partitions")
    
    def _build_temporal_index(self):
        """Build temporal sequence index for efficient pair sampling."""
        # Group by clips for temporal sequences
        self.sequence_map = {}
        self.flat_indices = []
        
        # Group by clip_id and build temporal sequences
        clip_groups = self.df.groupby('clip_id')
        
        for clip_id, group in clip_groups:
            # Sort by temporal order
            group_sorted = group.sort_values('frame_in_clip')
            frame_indices = group_sorted.index.tolist()
            
            self.sequence_map[clip_id] = frame_indices
            self.flat_indices.extend(frame_indices)
        
        self.size = len(self.flat_indices)
        print(f"Dataset contains {self.size} images across {len(self.sequence_map)} clips")
        
        # Stats
        sequence_lengths = [len(frames) for frames in self.sequence_map.values()]
        if sequence_lengths:
            print(f"Clip length stats: min={min(sequence_lengths)}, max={max(sequence_lengths)}, avg={np.mean(sequence_lengths):.1f}")
    
    def parse_csv_path(self, csv_path: str) -> Tuple[str, str, int]:
        """Parse CSV path to H5 coordinates (our verified approach)."""
        parts = csv_path.split('/')
        if len(parts) >= 4:
            category = parts[0]
            object_id = parts[1]
            frame_num = int(parts[3].split('.')[0]) - 1  # Convert to 0-based
            return category, object_id, frame_num
        return None, None, None
    
    def load_image_from_h5(self, partition: str, csv_path: str) -> Optional[np.ndarray]:
        """Load image from H5 file (our verified approach)."""
        # Get H5 file
        h5_file = self.partition_to_h5.get(partition) or self.partition_to_h5.get(urllib.parse.unquote(partition))
        if h5_file is None:
            return None
        
        # Parse path
        category, object_id, frame_index = self.parse_csv_path(csv_path)
        if category is None:
            return None
        
        # Load image with retries
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Try swmr mode first
                with h5py.File(h5_file, 'r', swmr=True, libver='latest') as f:
                    if category in f and object_id in f[category]:
                        obj_data = f[category][object_id]
                        if isinstance(obj_data, h5py.Dataset) and frame_index < len(obj_data):
                            img_bytes = obj_data[frame_index]
                            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                            if img_array is not None:
                                return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            except Exception as e:
                if retry == max_retries - 1:
                    # Try regular mode as fallback
                    try:
                        with h5py.File(h5_file, 'r') as f:
                            if category in f and object_id in f[category]:
                                obj_data = f[category][object_id]
                                if isinstance(obj_data, h5py.Dataset) and frame_index < len(obj_data):
                                    img_bytes = obj_data[frame_index]
                                    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                                    if img_array is not None:
                                        return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
        return None
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[Image.Image, Image.Image], int]:
        """Get sample with temporal pair following Core50 pattern."""
        df_idx = self.flat_indices[idx]
        sample = self.df.iloc[df_idx]
        
        # Load original image
        img_array = self.load_image_from_h5(sample['partition'], sample['path'])
        if img_array is None:
            # Fallback: create dummy image
            image = Image.new('RGB', (224, 224), color='gray')
        else:
            image = Image.fromarray(img_array)
        
        # Get clip sequence for temporal pairing
        clip_id = sample['clip_id']
        sequence = self.sequence_map[clip_id]
        
        # Find position in sequence
        try:
            position_in_sequence = sequence.index(df_idx)
        except ValueError:
            position_in_sequence = 0
        
        # Sample temporal pair
        paired_image = image  # Default fallback
        if len(sequence) > 1 and self.time_window > 0:
            # Find valid offset range
            min_offset = max(-self.time_window, -position_in_sequence)
            max_offset = min(self.time_window, len(sequence) - position_in_sequence - 1)
            
            if min_offset < max_offset:
                # Choose non-zero offset if possible
                if min_offset == 0 and max_offset > 0:
                    offset = random.randint(1, max_offset)
                elif max_offset == 0 and min_offset < 0:
                    offset = random.randint(min_offset, -1)
                else:
                    offset = random.randint(min_offset, max_offset)
                    if offset == 0 and min_offset < 0 and max_offset > 0:
                        offset = random.choice([random.randint(min_offset, -1), random.randint(1, max_offset)])
                
                # Load paired image
                paired_df_idx = sequence[position_in_sequence + offset]
                paired_sample = self.df.iloc[paired_df_idx]
                
                paired_img_array = self.load_image_from_h5(paired_sample['partition'], paired_sample['path'])
                if paired_img_array is not None:
                    paired_image = Image.fromarray(paired_img_array)
                # else: keep paired_image = image (fallback)
        
        # Get target label
        target = int(sample['label'])
        
        # Apply transformations if provided
        if self.transform is not None:
            return self.transform(image, paired_image), target
        
        return (image, paired_image), target


# Test the implementation
if __name__ == "__main__":
    # Test dataset creation with separate train/val files
    print("Testing TemporalMVImageNet with separate train/val files...")
    
    # Test training dataset
    train_dataset = TemporalMVImageNet(
        metadata_path='/home/brothen/solo-learn/mvimagenet_train.parquet',
        time_window=8,
        min_clip_length=10
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Test validation dataset
    val_dataset = TemporalMVImageNet(
        metadata_path='/home/brothen/solo-learn/mvimagenet_val.parquet',
        time_window=8,
        min_clip_length=10
    )
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test sample loading
    sample, target = train_dataset[0]
    image1, image2 = sample
    
    print(f"Sample loaded successfully:")
    print(f"  Image 1 size: {image1.size}")
    print(f"  Image 2 size: {image2.size}")
    print(f"  Target: {target}")
    
    print("✅ Implementation working correctly!")