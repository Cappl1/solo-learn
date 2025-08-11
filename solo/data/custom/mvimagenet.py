import io
import os
from typing import Tuple, Callable, Optional
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
import urllib.parse
import glob


class MVImageNet(Dataset):
    """Simple MVImageNet dataset for classification, similar to Core50.
    
    This loads images from H5 files with a simple interface, without temporal pairing.
    """
    
    def __init__(self,
                 h5_data_dir: str = '/home/data/MVImageNet/',
                 metadata_path: str = '/home/brothen/solo-learn/mvimagenet_usable.parquet',
                 transform: Optional[Callable] = None,
                 split: str = 'train',  # 'train', 'val', 'test'
                 use_categories: bool = False,  # For compatibility with Core50 interface
                 # Support for separate train/val metadata files (like TemporalMVImageNet)
                 train_metadata_path: Optional[str] = None,
                 val_metadata_path: Optional[str] = None,
                 ):
        
        self.h5_data_dir = h5_data_dir
        self.transform = transform
        self.split = split
        self.use_categories = use_categories
        
        print(f"Loading MVImageNet dataset from: {h5_data_dir}, split: {split}")
        
        # Load metadata - support both single file and separate train/val files
        if train_metadata_path is not None and val_metadata_path is not None:
            # Use separate train/val metadata files (like TemporalMVImageNet)
            if split == 'train':
                actual_metadata_path = train_metadata_path
            elif split == 'val':
                actual_metadata_path = val_metadata_path
            else:
                # For test or other splits, use train metadata as fallback
                actual_metadata_path = train_metadata_path
            
            print(f"Using separate metadata files - loading {split} from: {actual_metadata_path}")
            if not os.path.exists(actual_metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {actual_metadata_path}")
                
            self.df = pd.read_parquet(actual_metadata_path)
            print(f"Loaded {len(self.df):,} samples for {split}")
        else:
            # Use single metadata file with internal splitting (original behavior)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
                
            print(f"Loading metadata from: {metadata_path}")
            self.df = pd.read_parquet(metadata_path)
            print(f"Loaded {len(self.df):,} samples")
            
            # Simple split (we can make this more sophisticated later)
            if split == 'train':
                # Use first 90% for training
                split_idx = int(0.9 * len(self.df))
                self.df = self.df[:split_idx].reset_index(drop=True)
            elif split == 'val':
                # Use last 10% for validation
                split_idx = int(0.9 * len(self.df))
                self.df = self.df[split_idx:].reset_index(drop=True)
            # For 'test' or any other split, use all data
            
            print(f"Dataset {split} contains {len(self.df):,} samples")
        
        # Create H5 partition mapping
        self._create_h5_mapping()
    
    def _create_h5_mapping(self):
        """Create partition to H5 file mapping."""
        h5_files = glob.glob(os.path.join(self.h5_data_dir, 'data*.h5'))
        self.partition_to_h5 = {}
        
        for h5_file in h5_files:
            basename = os.path.basename(h5_file).replace('.h5', '')
            partition_name = basename.replace('data', '', 1)
            partition_decoded = urllib.parse.unquote(partition_name)
            
            self.partition_to_h5[partition_name] = h5_file
            self.partition_to_h5[partition_decoded] = h5_file
        
        print(f"Created H5 mapping for {len(self.partition_to_h5)} partitions")
    
    def parse_csv_path(self, csv_path: str) -> Tuple[str, str, int]:
        """Parse CSV path to H5 coordinates."""
        parts = csv_path.split('/')
        if len(parts) >= 4:
            category = parts[0]
            object_id = parts[1]
            frame_num = int(parts[3].split('.')[0]) - 1  # Convert to 0-based
            return category, object_id, frame_num
        return None, None, None
    
    def load_image_from_h5(self, partition: str, csv_path: str) -> Optional[np.ndarray]:
        """Load image from H5 file."""
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
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """Get single image and label (Core50-style interface)."""
        sample = self.df.iloc[idx]
        
        # Load image
        img_array = self.load_image_from_h5(sample['partition'], sample['path'])
        if img_array is None:
            # Fallback: create dummy image
            image = Image.new('RGB', (224, 224), color='gray')
        else:
            image = Image.fromarray(img_array)
        
        # Get target label
        target = int(sample['label'])
        
        # Apply category mapping if requested (similar to Core50)
        if self.use_categories:
            # MVImageNet has 1000 categories, we could group them somehow
            # For now, just use label // 4 to get ~250 categories  
            target = target // 4
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target


class MVImageNetForCameraPoseRegression(Dataset):
    """MVImageNet dataset for camera pose regression tasks."""
    
    def __init__(self,
                 h5_data_dir: str = '/home/data/MVImageNet/',
                 metadata_path: str = '/home/brothen/solo-learn/mvimagenet_usable.parquet',
                 transform: Optional[Callable] = None,
                 split: str = 'train'
                 ):
        
        self.h5_data_dir = h5_data_dir
        self.transform = transform
        self.split = split
        
        print(f"Loading MVImageNet for pose regression from: {h5_data_dir}")
        
        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        self.df = pd.read_parquet(metadata_path)
        print(f"Loaded {len(self.df):,} samples for pose regression")
        
        # Create H5 mapping
        self._create_h5_mapping()
        
        # Simple split
        if split == 'train':
            split_idx = int(0.9 * len(self.df))
            self.df = self.df[:split_idx].reset_index(drop=True)
        elif split == 'val':
            split_idx = int(0.9 * len(self.df))
            self.df = self.df[split_idx:].reset_index(drop=True)
        
        print(f"Pose regression {split} contains {len(self.df):,} samples")
    
    def _create_h5_mapping(self):
        """Create partition to H5 file mapping."""
        h5_files = glob.glob(os.path.join(self.h5_data_dir, 'data*.h5'))
        self.partition_to_h5 = {}
        
        for h5_file in h5_files:
            basename = os.path.basename(h5_file).replace('.h5', '')
            partition_name = basename.replace('data', '', 1)
            partition_decoded = urllib.parse.unquote(partition_name)
            
            self.partition_to_h5[partition_name] = h5_file
            self.partition_to_h5[partition_decoded] = h5_file
    
    def parse_csv_path(self, csv_path: str) -> Tuple[str, str, int]:
        """Parse CSV path to H5 coordinates."""
        parts = csv_path.split('/')
        if len(parts) >= 4:
            category = parts[0]
            object_id = parts[1]
            frame_num = int(parts[3].split('.')[0]) - 1
            return category, object_id, frame_num
        return None, None, None
    
    def load_image_from_h5(self, partition: str, csv_path: str) -> Optional[np.ndarray]:
        """Load image from H5 file."""
        h5_file = self.partition_to_h5.get(partition) or self.partition_to_h5.get(urllib.parse.unquote(partition))
        if h5_file is None:
            return None
        
        category, object_id, frame_index = self.parse_csv_path(csv_path)
        if category is None:
            return None
        
        try:
            with h5py.File(h5_file, 'r', swmr=True, libver='latest') as f:
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
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, np.ndarray]:
        """Get image and camera pose."""
        sample = self.df.iloc[idx]
        
        # Load image
        img_array = self.load_image_from_h5(sample['partition'], sample['path'])
        if img_array is None:
            image = Image.new('RGB', (224, 224), color='gray')
        else:
            image = Image.fromarray(img_array)
        
        # Get camera pose (azimuth, elevation as a simple 2D pose)
        pose = np.array([sample.get('azimuth', 0.0), sample.get('elevation', 0.0)], dtype=np.float32)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, pose


def create_mvimagenet_splits(
    h5_data_dir: str = '/home/data/MVImageNet/',
    metadata_path: str = '/home/brothen/solo-learn/mvimagenet_usable.parquet',
    dataset_class: type = MVImageNet,
    **kwargs
):
    """
    Helper function to create train/val/test datasets.
    
    Args:
        h5_data_dir: Directory containing H5 files
        metadata_path: Path to metadata parquet file
        dataset_class: Which dataset class to use (MVImageNet, MVImageNetForCameraPoseRegression)
        **kwargs: Additional arguments passed to the dataset class
    
    Returns:
        Dict with 'train', 'val', 'test' datasets
    """
    
    datasets = {}
    
    # Training dataset
    datasets['train'] = dataset_class(
        h5_data_dir=h5_data_dir,
        metadata_path=metadata_path,
        split='train',
        **kwargs
    )
    
    # Validation dataset
    datasets['val'] = dataset_class(
        h5_data_dir=h5_data_dir,
        metadata_path=metadata_path,
        split='val',
        **kwargs
    )
    
    # Test dataset (same as val for now)
    datasets['test'] = dataset_class(
        h5_data_dir=h5_data_dir,
        metadata_path=metadata_path,
        split='test',
        **kwargs
    )
    
    return datasets 