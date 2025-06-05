import io
import os
import random
from typing import Tuple, Callable, Optional, List, Sequence
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from solo.data.custom.temporal_core50 import TemporalCore50

class SelectiveTemporalCore50(TemporalCore50):
    def __init__(
        self,
        h5_path: str,
        transform: Callable,
        time_window: int = 15,
        backgrounds: Optional[Sequence[str]] = None,
        val_backgrounds: Optional[Sequence[str]] = None,
        num_candidates: int = 8,
        is_validation: bool = False,  # Add validation mode flag
    ):
        super().__init__(h5_path, transform, time_window, backgrounds)
        self.num_candidates = num_candidates
        self._h5: Optional[h5py.File] = None  # lazy per-worker handle
        
        # Store val_backgrounds in case it's needed later
        self.val_backgrounds = val_backgrounds
        self.is_validation = is_validation  # Flag to determine validation mode

    # --------------------------------------------------------------------- #
    # helpers
    def _get_file(self) -> h5py.File:
        """Get H5 file handle, handling potential SWMR conflicts with retry mechanism."""
        if self._h5 is None:
            # Try several approaches to open the file, with graceful fallbacks
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # First try with SWMR=True (safe concurrent access)
                    self._h5 = h5py.File(self.h5_path, "r", swmr=True)
                    break
                except OSError as e:
                    if "SWMR" in str(e) and retry < max_retries - 1:
                        # If SWMR conflict and not last retry, try without SWMR
                        try:
                            self._h5 = h5py.File(self.h5_path, "r", swmr=False)
                            print(f"Warning: Opened H5 file without SWMR mode due to conflicts")
                            break
                        except Exception as inner_e:
                            print(f"Warning: Failed to open H5 file without SWMR: {inner_e}")
                    
                    # If last retry or other error, raise warning and try one more approach
                    if retry == max_retries - 1:
                        print(f"Warning: Failed to open H5 file after {max_retries} retries: {e}")
                        # Last attempt - force close any existing handles and retry
                        try:
                            import gc
                            gc.collect()  # Try to collect any unreferenced file handles
                            self._h5 = h5py.File(self.h5_path, "r")  # Simple open
                            print("Successfully opened H5 file with basic mode after cleanup")
                            break
                        except Exception as last_e:
                            print(f"Critical error opening H5 file: {last_e}")
                            raise
        
        return self._h5

    # --------------------------------------------------------------------- #
    def __getitem__(self, idx: int):
        h5 = self._get_file()
        session, local_idx = self.flat_indices[idx]
        obj_id = int(h5[session]["targets"][local_idx])

        # key-frame image ---------------------------------------------------
        key_img = Image.fromarray(h5[session]["images"][local_idx]).convert("RGB")
        key_views = self.transform(key_img)
        key_tensor = key_views[0] if isinstance(key_views, list) else key_views  # (C,H,W)

        # For validation, just return the key tensor and label
        if self.is_validation:
            return key_tensor, torch.tensor(obj_id, dtype=torch.long)

        # find temporal candidates -----------------------------------------
        seq = self.sequence_map[(session, obj_id)]
        pos = seq.index(idx)

        min_off = max(-self.time_window, -pos)
        max_off = min(self.time_window, len(seq) - pos - 1)
        valid_off = [o for o in range(min_off, max_off + 1) if o != 0] or [1, -1]

        k = min(self.num_candidates, len(valid_off))
        offsets = random.sample(valid_off, k) if len(valid_off) >= k else random.choices(valid_off, k=k)

        cand_tensors: List[torch.Tensor] = []
        for off in offsets:
            cand_idx = seq[pos + off]
            s, li = self.flat_indices[cand_idx]
            cand_img = Image.fromarray(h5[s]["images"][li]).convert("RGB")
            cand_views = self.transform(cand_img)
            cand_tensor = cand_views[0] if isinstance(cand_views, list) else cand_views
            cand_tensors.append(cand_tensor)

        cand_tensor = torch.stack(cand_tensors)  # (K,C,H,W)
        # Convert obj_id to torch.long tensor for compatibility with loss functions
        obj_id_tensor = torch.tensor(obj_id, dtype=torch.long)
        return key_tensor, cand_tensor, obj_id_tensor

    def __del__(self):
        """Clean up file handle when dataset is deleted."""
        if hasattr(self, '_h5') and self._h5 is not None:
            try:
                self._h5.close()
                self._h5 = None
            except Exception as e:
                print(f"Warning: Error closing H5 file: {e}")