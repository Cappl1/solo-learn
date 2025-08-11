
import h5py
import cv2
import numpy as np
import urllib.parse

def load_mvimagenet_image(csv_path, partition_id, partition_to_h5_mapping):
    """Load image from H5 file using CSV metadata"""
    
    def parse_csv_path(csv_path):
        parts = csv_path.split('/')
        if len(parts) >= 4:
            category = parts[0]
            object_id = parts[1]
            frame_num = int(parts[3].split('.')[0]) - 1
            return category, object_id, frame_num
        return None, None, None
    
    category, object_id, frame_index = parse_csv_path(csv_path)
    if category is None:
        return None
        
    # Use the actual partition column value to find H5 file
    h5_file = partition_to_h5_mapping.get(partition_id)
    if h5_file is None:
        h5_file = partition_to_h5_mapping.get(urllib.parse.unquote(partition_id))
    
    if h5_file is None:
        return None
        
    try:
        with h5py.File(h5_file, 'r') as f:
            if category in f and object_id in f[category]:
                obj_data = f[category][object_id]
                if isinstance(obj_data, h5py.Dataset) and frame_index < len(obj_data):
                    img_bytes = obj_data[frame_index]
                    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    return img_array
    except:
        pass
    return None
