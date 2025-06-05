import io
from typing import Tuple, Callable, Optional
from pathlib import Path
import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MVImageNet(Dataset):
    """MVImageNet dataset for classification and evaluation.
    
    This implementation loads individual samples from H5 files using metadata
    from CSV/parquet files. Suitable for standard classification, evaluation,
    and linear evaluation protocols.
    """
    def __init__(self,
                 h5_path: str,
                 metadata_path: str,
                 transform: Optional[Callable] = None,
                 use_categories: bool = True,  # Use category labels (268 classes)
                 use_objects: bool = False,   # Use object instance labels (~31K classes)
                 split: str = 'train',
                 ):
        self.transform = transform
        self.use_categories = use_categories
        self.use_objects = use_objects
        self.h5_path = h5_path
        self.split = split
        
        print(f"Loading MVImageNet dataset from: {h5_path}, split: {split}")
        print(f"Label mode: {'categories' if use_categories else 'objects' if use_objects else 'none'}")

        # Load metadata
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.suffix == '.csv':
            self.df = pd.read_csv(metadata_path, on_bad_lines='skip')
        elif self.metadata_path.suffix == '.parquet':
            self.df = pd.read_parquet(metadata_path, engine='fastparquet')
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path}")
        
        print(f"Loaded {len(self.df)} samples from {metadata_path}")
        
        # Analyze the data
        self._analyze_data()

    def _analyze_data(self):
        """Analyze the loaded data and print statistics."""
        print(f"Data analysis:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Unique partitions: {self.df['partition'].nunique()}")
        print(f"  Unique categories: {self.df['category'].nunique()}")
        print(f"  Unique objects: {self.df['object'].nunique()}")
        
        # Check label ranges
        if self.use_categories:
            min_cat = self.df['category'].min()
            max_cat = self.df['category'].max()
            print(f"  Category range: {min_cat} to {max_cat}")
            
        if self.use_objects:
            unique_objects = self.df['object'].unique()
            print(f"  Object instances: {len(unique_objects)}")
            # Create object to integer mapping
            self.object_to_idx = {obj: idx for idx, obj in enumerate(sorted(unique_objects))}
            print(f"  Object label range: 0 to {len(unique_objects)-1}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        sample = self.df.iloc[idx]
        
        # Extract H5 path components
        partition = sample['partition']
        category = str(sample['category'])
        obj = sample['object']
        frame = sample['frame']
        
        # Load image with retries for robustness
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Try with swmr mode for better concurrent access
                with h5py.File(self.h5_path, 'r', swmr=True, libver='latest') as h5_file:
                    img_data = h5_file[partition][category][obj][frame]
                    break
            except:
                if retry == max_retries - 1:
                    # Fall back to regular mode
                    try:
                        with h5py.File(self.h5_path, 'r') as h5_file:
                            img_data = h5_file[partition][category][obj][frame]
                        break
                    except Exception as e:
                        print(f"Error loading image at idx {idx}: {e}")
                        # Return dummy image as last resort
                        dummy_img = Image.new('RGB', (224, 224), color='gray')
                        return (dummy_img, -1) if self.transform is None else (self.transform(dummy_img), -1)
        
        # Convert binary data to image
        if isinstance(img_data, bytes):
            img_bytes = img_data
        else:
            img_bytes = img_data.tobytes()
        
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Determine label based on configuration
        if self.use_categories:
            label = int(sample['category'])
            # Validate category range
            assert 0 <= label <= 267, f"Invalid category label {label} at index {idx}"
        elif self.use_objects:
            obj_str = sample['object']
            label = self.object_to_idx[obj_str]
        else:
            label = 0  # Dummy label if neither categories nor objects are used
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MVImageNetForCameraPoseRegression(Dataset):
    """MVImageNet dataset for camera pose regression tasks.
    
    Returns images with their corresponding camera poses (quaternion + translation).
    Useful for camera pose estimation and 3D understanding tasks.
    """
    def __init__(self,
                 h5_path: str,
                 metadata_path: str,
                 transform: Optional[Callable] = None,
                 split: str = 'train',
                 ):
        self.transform = transform
        self.h5_path = h5_path
        self.split = split
        
        print(f"Loading MVImageNet (pose regression) from: {h5_path}, split: {split}")

        # Load metadata
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.suffix == '.csv':
            self.df = pd.read_csv(metadata_path, on_bad_lines='skip')
        elif self.metadata_path.suffix == '.parquet':
            self.df = pd.read_parquet(metadata_path, engine='fastparquet')
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path}")
        
        print(f"Loaded {len(self.df)} samples with camera poses")
        
        # Verify pose columns exist
        pose_columns = ['q0', 'q1', 'q2', 'q3', 't0', 't1', 't2']
        missing_cols = [col for col in pose_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing pose columns: {missing_cols}")
        
        print(f"Camera pose statistics:")
        print(f"  Quaternion (q0-q3) ranges:")
        for i in range(4):
            col = f'q{i}'
            print(f"    {col}: [{self.df[col].min():.3f}, {self.df[col].max():.3f}]")
        print(f"  Translation (t0-t2) ranges:")
        for i in range(3):
            col = f't{i}'
            print(f"    {col}: [{self.df[col].min():.3f}, {self.df[col].max():.3f}]")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Tuple[float, ...]]:
        sample = self.df.iloc[idx]
        
        # Extract H5 path components
        partition = sample['partition']
        category = str(sample['category'])
        obj = sample['object']
        frame = sample['frame']
        
        # Load image
        max_retries = 3
        for retry in range(max_retries):
            try:
                with h5py.File(self.h5_path, 'r', swmr=True, libver='latest') as h5_file:
                    img_data = h5_file[partition][category][obj][frame]
                    break
            except:
                if retry == max_retries - 1:
                    with h5py.File(self.h5_path, 'r') as h5_file:
                        img_data = h5_file[partition][category][obj][frame]
                    break
        
        # Convert to image
        if isinstance(img_data, bytes):
            img_bytes = img_data
        else:
            img_bytes = img_data.tobytes()
        
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Extract camera pose (quaternion + translation)
        pose = (
            float(sample['q0']), float(sample['q1']), float(sample['q2']), float(sample['q3']),  # quaternion
            float(sample['t0']), float(sample['t1']), float(sample['t2'])   # translation
        )
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, pose


def create_mvimagenet_splits(
    h5_path: str = '/home/data/MVImageNet/data_all.h5',
    train_csv: str = '/home/data/MVImageNet/datasetT_0.1_7_2.csv',
    val_parquet: str = '/home/data/MVImageNet/dataset_val_all3.parquet',
    test_parquet: str = '/home/data/MVImageNet/dataset_test_all3.parquet',
    dataset_class: type = MVImageNet,
    **kwargs
):
    """
    Helper function to create train/val/test datasets.
    
    Args:
        h5_path: Path to the main H5 file containing images
        train_csv: Path to CSV file for training
        val_parquet: Path to validation parquet file
        test_parquet: Path to test parquet file
        dataset_class: Which dataset class to use (MVImageNet, MVImageNetForCameraPoseRegression)
        **kwargs: Additional arguments passed to the dataset class
    
    Returns:
        Dict with 'train', 'val', 'test' datasets
    """
    
    datasets = {}
    
    # Training dataset (from CSV)
    datasets['train'] = dataset_class(
        h5_path=h5_path,
        metadata_path=train_csv,
        split='train',
        **kwargs
    )
    
    # Validation dataset (from parquet)
    datasets['val'] = dataset_class(
        h5_path=h5_path,
        metadata_path=val_parquet,
        split='val',
        **kwargs
    )
    
    # Test dataset (from parquet)
    datasets['test'] = dataset_class(
        h5_path=h5_path,
        metadata_path=test_parquet,
        split='test',
        **kwargs
    )
    
    return datasets 