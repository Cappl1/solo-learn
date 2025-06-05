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
from sklearn.model_selection import train_test_split


class TemporalMVImageNet(Dataset):
    """MVImageNet dataset with temporal pairing for self-supervised learning.
    
    This implementation expects an H5 file with partition/category/object/frame structure
    and metadata from CSV/parquet files. It pairs each image with another from the same 
    object sequence within a time window.
    
    This version is designed to handle concurrent access to the H5 file from multiple processes.
    """
    def __init__(self,
                 h5_path: str,
                 metadata_path: str,
                 transform: Optional[Callable] = None,
                 time_window: int = 15,  # How far to look for temporal pairs
                 split: str = 'train',  # 'train', 'val', 'test'
                 val_split: float = 0.05,  # Fraction for validation set
                 stratify_by_category: bool = True,  # Whether to balance classes in val split
                 random_seed: int = 42,  # For reproducible splits
                 ):
        self.transform = transform
        self.time_window = time_window
        self.h5_path = h5_path
        self.split = split
        self.val_split = val_split
        self.stratify_by_category = stratify_by_category
        self.random_seed = random_seed
        
        print(f"Loading MVImageNet dataset from: {h5_path} with time_window={time_window}, split={split}")
        
        # Validate required parameters
        if metadata_path is None:
            raise ValueError("metadata_path is required but was None. Please provide a valid path to the metadata file (CSV or parquet).")
        
        # Load metadata
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.suffix == '.csv':
            self.df = pd.read_csv(metadata_path, on_bad_lines='skip')
        elif self.metadata_path.suffix == '.parquet':
            self.df = pd.read_parquet(metadata_path, engine='fastparquet')
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path}")
        
        print(f"Loaded {len(self.df)} samples from {metadata_path}")
        
        # Apply train/val split if requested
        if self.val_split > 0 and self.split in ['train', 'val']:
            self._apply_train_val_split()
        
        # Build temporal sequence index
        self._build_temporal_index()
    
    def _apply_train_val_split(self):
        """Split the dataset into train/val with balanced class distribution."""
        print(f"Applying {self.val_split*100:.1f}% validation split...")
        
        # Get class distribution
        category_counts = self.df['category'].value_counts()
        print(f"Original dataset has {len(category_counts)} categories")
        print(f"Category distribution (top 10): {category_counts.head(10).to_dict()}")
        
        if self.stratify_by_category and len(category_counts) > 1:
            # Stratified split by category to maintain class balance
            try:
                # For large datasets with many classes, we might need to handle small categories
                min_samples_per_class = int(1 / self.val_split) + 1  # At least 1 sample in val
                
                # Filter out categories with too few samples for stratification
                valid_categories = category_counts[category_counts >= min_samples_per_class].index
                
                if len(valid_categories) < len(category_counts):
                    print(f"Warning: {len(category_counts) - len(valid_categories)} categories have too few samples for stratification")
                    
                # Filter dataframe to only include categories with enough samples
                df_stratify = self.df[self.df['category'].isin(valid_categories)]
                df_remaining = self.df[~self.df['category'].isin(valid_categories)]
                
                if len(df_stratify) > 0:
                    # Perform stratified split
                    train_idx, val_idx = train_test_split(
                        df_stratify.index,
                        test_size=self.val_split,
                        stratify=df_stratify['category'],
                        random_state=self.random_seed
                    )
                    
                    # Add remaining samples to training set
                    if len(df_remaining) > 0:
                        train_idx = train_idx.tolist() + df_remaining.index.tolist()
                else:
                    # Fallback to random split if stratification fails
                    train_idx, val_idx = train_test_split(
                        self.df.index,
                        test_size=self.val_split,
                        random_state=self.random_seed
                    )
                    
            except Exception as e:
                print(f"Stratified split failed ({e}), falling back to random split")
                train_idx, val_idx = train_test_split(
                    self.df.index,
                    test_size=self.val_split,
                    random_state=self.random_seed
                )
        else:
            # Simple random split
            train_idx, val_idx = train_test_split(
                self.df.index,
                test_size=self.val_split,
                random_state=self.random_seed
            )
        
        # Apply the split
        if self.split == 'train':
            self.df = self.df.loc[train_idx].reset_index(drop=True)
            print(f"Training set: {len(self.df)} samples")
        elif self.split == 'val':
            self.df = self.df.loc[val_idx].reset_index(drop=True)
            print(f"Validation set: {len(self.df)} samples")
            
            # Show validation set class distribution
            val_category_counts = self.df['category'].value_counts()
            print(f"Validation set has {len(val_category_counts)} categories")
            print(f"Val category distribution (top 10): {val_category_counts.head(10).to_dict()}")
    
    def _build_temporal_index(self):
        """Build an index of sequences and frames for efficient temporal retrieval."""
        
        # Build an index that maps (partition, category, object) to lists of frame indices
        # This allows us to easily find frames from the same object sequence
        self.sequence_map = {}
        self.flat_indices = []
        
        # Group by object sequences
        object_groups = self.df.groupby(['partition', 'category', 'object'])
        
        for (partition, category, obj), group in object_groups:
            # Sort by frame number to ensure temporal order
            group_sorted = group.sort_values('frame')
            sequence_key = (partition, category, obj)
            
            # Store the dataframe indices for this sequence
            frame_indices = group_sorted.index.tolist()
            self.sequence_map[sequence_key] = frame_indices
            
            # Add all frames to flat index
            self.flat_indices.extend(frame_indices)
                
        self.size = len(self.flat_indices)
        print(f"Dataset contains {self.size} images across {len(self.sequence_map)} object sequences")
        
        # Store some stats
        sequence_lengths = [len(frames) for frames in self.sequence_map.values()]
        if sequence_lengths:
            print(f"Sequence length stats: min={min(sequence_lengths)}, max={max(sequence_lengths)}, avg={np.mean(sequence_lengths):.1f}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tuple[Image.Image, Image.Image], int]:
        # Maximum number of retries for file access
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                # Try with swmr mode first
                with h5py.File(self.h5_path, 'r', swmr=True, libver='latest') as h5_file:
                    return self._get_sample(idx, h5_file)
            except Exception as e:
                if retry == max_retries - 1:
                    # Fall back to regular mode on last retry
                    try:
                        with h5py.File(self.h5_path, 'r') as h5_file:
                            return self._get_sample(idx, h5_file)
                    except Exception as last_e:
                        print(f"Error accessing H5 file after {max_retries} retries: {last_e}")
                        # Return a dummy sample as last resort
                        dummy_img = Image.new('RGB', (224, 224), color='gray')
                        if self.transform:
                            return self.transform(dummy_img, dummy_img), -1
                        return (dummy_img, dummy_img), -1
        
        # This should never happen due to the fallback above, but just in case
        raise RuntimeError(f"Failed to access H5 file after {max_retries} retries")
    
    def _get_sample(self, idx, h5_file):
        """Get a sample with its temporal pair from the H5 file."""
        df_idx = self.flat_indices[idx]
        sample = self.df.iloc[df_idx]
        
        # Extract H5 path components
        partition = sample['partition']
        category = str(sample['category'])
        obj = sample['object']
        frame = sample['frame']
        
        # Check if the partition exists in H5 file
        if partition not in h5_file:
            print(f"Warning: Partition {partition} not found in H5 file, using dummy image")
            dummy_img = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                return self.transform(dummy_img, dummy_img), int(sample['category'])
            return (dummy_img, dummy_img), int(sample['category'])
        
        # Check if the category exists
        if category not in h5_file[partition]:
            print(f"Warning: Category {category} not found in partition {partition}, using dummy image")
            dummy_img = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                return self.transform(dummy_img, dummy_img), int(sample['category'])
            return (dummy_img, dummy_img), int(sample['category'])
        
        # Check if the object exists
        if obj not in h5_file[partition][category]:
            print(f"Warning: Object {obj} not found in {partition}/{category}, using dummy image")
            dummy_img = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                return self.transform(dummy_img, dummy_img), int(sample['category'])
            return (dummy_img, dummy_img), int(sample['category'])
        
        # Check if the frame exists
        object_dataset = h5_file[partition][category][obj]
        if frame >= len(object_dataset):
            print(f"Warning: Frame {frame} out of range for {partition}/{category}/{obj} (length: {len(object_dataset)}), using dummy image")
            dummy_img = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                return self.transform(dummy_img, dummy_img), int(sample['category'])
            return (dummy_img, dummy_img), int(sample['category'])
        
        # Get the original image
        try:
            img_data = h5_file[partition][category][obj][frame]
            if isinstance(img_data, bytes):
                img_bytes = img_data
            else:
                img_bytes = img_data.tobytes()
            
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"Warning: Error decoding image for {partition}/{category}/{obj}/{frame}: {e}, using dummy image")
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Get the sequence key and find temporal pair
        sequence_key = (partition, category, obj)
        sequence = self.sequence_map[sequence_key]
        
        # Find the position of current frame in the sequence
        try:
            position_in_sequence = sequence.index(df_idx)
        except ValueError:
            # Fallback if not found (shouldn't happen)
            position_in_sequence = 0
        
        # Choose another frame from the same sequence within the time window
        paired_image = image  # Default fallback
        if len(sequence) > 1 and self.time_window > 0:
            # Find valid range for temporal offset
            min_offset = max(-self.time_window, -position_in_sequence)
            max_offset = min(self.time_window, len(sequence) - position_in_sequence - 1)
            
            # Handle special cases
            if min_offset >= max_offset:  
                # If there's no valid range (can happen at sequence boundaries)
                paired_image = image  # Use the same image as fallback
            else:
                # Choose a non-zero offset if possible
                if min_offset == 0 and max_offset > 0:
                    offset = random.randint(1, max_offset)
                elif max_offset == 0 and min_offset < 0:
                    offset = random.randint(min_offset, -1)
                else:
                    offset = random.randint(min_offset, max_offset)
                    if offset == 0 and min_offset < 0 and max_offset > 0:
                        # Try again to avoid zero offset
                        offset = random.choice([random.randint(min_offset, -1), random.randint(1, max_offset)])
                
                # Get the paired frame
                paired_df_idx = sequence[position_in_sequence + offset]
                paired_sample = self.df.iloc[paired_df_idx]
                
                # Load the paired image with same error checking
                paired_partition = paired_sample['partition']
                paired_category = str(paired_sample['category'])
                paired_obj = paired_sample['object']
                paired_frame = paired_sample['frame']
                
                try:
                    # Check if paired frame exists in H5
                    if (paired_partition in h5_file and 
                        paired_category in h5_file[paired_partition] and 
                        paired_obj in h5_file[paired_partition][paired_category] and 
                        paired_frame < len(h5_file[paired_partition][paired_category][paired_obj])):
                        
                        paired_img_data = h5_file[paired_partition][paired_category][paired_obj][paired_frame]
                        if isinstance(paired_img_data, bytes):
                            paired_img_bytes = paired_img_data
                        else:
                            paired_img_bytes = paired_img_data.tobytes()
                        
                        paired_image = Image.open(io.BytesIO(paired_img_bytes)).convert("RGB")
                    else:
                        # Fall back to original image if paired doesn't exist
                        paired_image = image
                except Exception as e:
                    print(f"Warning: Error loading paired image: {e}, using original image")
                    paired_image = image
            
        # Apply transformations
        if self.transform is not None:
            # Pass both images to the transform pipeline
            return self.transform(image, paired_image), int(sample['category'])
        
        return (image, paired_image), int(sample['category'])


def create_temporal_mvimagenet_splits(
    h5_path: str = '/home/data/MVImageNet/data_all.h5',
    train_csv: str = '/home/data/MVImageNet/datasetT_0.1_7_2.csv',
    val_parquet: str = '/home/data/MVImageNet/dataset_val_all3.parquet',
    test_parquet: str = '/home/data/MVImageNet/dataset_test_all3.parquet',
    val_split: float = 0.05,
    **kwargs
):
    """
    Helper function to create train/val/test datasets with balanced validation split.
    
    Args:
        h5_path: Path to the main H5 file containing images
        train_csv: Path to CSV file for training (one of the large CSV files we found)
        val_parquet: Path to validation parquet file
        test_parquet: Path to test parquet file
        val_split: Fraction of training data to use for validation monitoring
        **kwargs: Additional arguments passed to TemporalMVImageNet
    
    Returns:
        Dict with 'train', 'val', 'test' datasets
    """
    
    datasets = {}
    
    # Training dataset (from CSV) - this will include both train and val for actual training
    # but val split will be used for monitoring
    datasets['train'] = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=train_csv,
        split='train',
        val_split=val_split,
        **kwargs
    )
    
    # Validation dataset (5% split from training data for monitoring)
    datasets['val'] = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=train_csv,  # Same source as train
        split='val',
        val_split=val_split,
        **kwargs
    )
    
    # Test dataset (from separate parquet file)
    datasets['test'] = TemporalMVImageNet(
        h5_path=h5_path,
        metadata_path=test_parquet,
        split='test',
        val_split=0.0,  # No splitting for test set
        **kwargs
    )
    
    return datasets 