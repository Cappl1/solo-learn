import io
import os
import random
from typing import Tuple, Callable, Optional, List
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TemporalCore50(Dataset):
    """Core50 dataset with temporal pairing for self-supervised learning.
    
    This implementation expects an H5 file with session groups containing images and targets.
    It pairs each image with another from the same object sequence within a time window.
    
    This version is designed to handle concurrent access to the H5 file from multiple processes.
    """
    def __init__(self,
                 h5_path: str,
                 transform: Optional[Callable] = None,
                 time_window: int = 15,  # How far to look for temporal pairs
                 backgrounds: Optional[List[str]] = None,  # Session IDs to use
                 ):
        self.transform = transform
        self.time_window = time_window
        self.h5_path = h5_path
        print(f"Loading Core50 dataset from: {h5_path} with time_window={time_window}")
        
        # Use swmr mode (single-writer, multiple-reader) for better concurrent access
        # Mode 'r' is read-only, which is what we need for the dataset
        try:
            # Try with swmr=True for better concurrent access
            with h5py.File(h5_path, 'r', swmr=True, libver='latest') as h5_file:
                self._build_index(h5_file, backgrounds)
        except:
            # Fall back to regular mode if swmr fails
            print("Warning: Falling back to standard H5 read mode. Concurrent access may cause issues.")
            with h5py.File(h5_path, 'r') as h5_file:
                self._build_index(h5_file, backgrounds)
    
    def _build_index(self, h5_file, backgrounds):
        """Build an index of sequences and frames for efficient retrieval."""
        # Get available sessions
        all_sessions = list(h5_file.keys())
        self.sessions = backgrounds if backgrounds is not None else all_sessions
        print(f"Using sessions: {self.sessions}")
        
        # Store metadata about the structure to avoid opening the file again
        self.metadata = {}
        
        # Build an index that maps (session, object_id) to lists of image indices
        # This allows us to easily find frames from the same object sequence
        self.sequence_map = {}
        self.flat_indices = []
        
        total_counter = 0
        for session in self.sessions:
            if session not in h5_file:
                print(f"Warning: Session {session} not found in H5 file")
                continue
                
            session_group = h5_file[session]
            images_shape = session_group['images'].shape
            targets_shape = session_group['targets'].shape
            
            # Store metadata
            self.metadata[session] = {
                'images_shape': images_shape,
                'targets_shape': targets_shape
            }
            
            targets = session_group['targets'][:]  # Load all targets into memory
            
            # Group images by object_id (target)
            for i in range(len(targets)):
                object_id = targets[i]
                key = (session, int(object_id))
                
                if key not in self.sequence_map:
                    self.sequence_map[key] = []
                
                # Store the global index for this image
                self.sequence_map[key].append(total_counter)
                self.flat_indices.append((session, i))  # Store (session, local_idx)
                total_counter += 1
                
        self.size = len(self.flat_indices)
        print(f"Dataset contains {self.size} images across {len(self.sequence_map)} object sequences")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
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
        session, local_idx = self.flat_indices[idx]
        
        # Get the original image
        img_data = h5_file[session]['images'][local_idx]
        if isinstance(img_data, bytes):
            # If images are stored as bytes
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            # If images are stored as arrays
            image = Image.fromarray(img_data).convert("RGB")
        
        # Get the object_id for this image
        object_id = h5_file[session]['targets'][local_idx]
        
        # Get the sequence key
        sequence_key = (session, int(object_id))
        
        # Find the position of this image in its sequence
        sequence = self.sequence_map[sequence_key]
        position_in_sequence = sequence.index(idx)
        
        # Choose another frame from the same sequence within the time window
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
                paired_idx = sequence[position_in_sequence + offset]
                paired_session, paired_local_idx = self.flat_indices[paired_idx]
                
                # Load the paired image
                paired_img_data = h5_file[paired_session]['images'][paired_local_idx]
                if isinstance(paired_img_data, bytes):
                    paired_image = Image.open(io.BytesIO(paired_img_data)).convert("RGB")
                else:
                    paired_image = Image.fromarray(paired_img_data).convert("RGB")
        else:
            # Fallback: use the same image if no temporal pairing is possible
            paired_image = image
            
        # Apply transformations
        if self.transform is not None:
            # Pass both images to the transform pipeline
            return self.transform(image, paired_image), int(object_id)
        
        return (image, paired_image), int(object_id)