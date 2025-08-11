import pandas as pd
import h5py
import cv2
import numpy as np
import os
import glob
import time
from collections import defaultdict
import urllib.parse

# Load the full dataset
csv_path = '/home/brothen/solo-learn/mvimagenet_master.parquet'
data_dir = '/home/data/MVImageNet/'

print("=== VERIFYING ALL IMAGES IN DATASET ===")
print(f"Loading full dataset from: {csv_path}")

df = pd.read_parquet(csv_path)
print(f"Total images to verify: {len(df):,}")
print(f"Total clips: {df['clip_id'].nunique():,}")
print(f"Total classes: {df['label'].nunique()}")

# Create H5 file mapping
h5_files = glob.glob(os.path.join(data_dir, 'data*.h5'))
partition_to_h5 = {}

for h5_file in h5_files:
    basename = os.path.basename(h5_file).replace('.h5', '')
    partition_name = basename.replace('data', '', 1)
    partition_decoded = urllib.parse.unquote(partition_name)
    
    partition_to_h5[partition_name] = h5_file
    partition_to_h5[partition_decoded] = h5_file

print(f"H5 mapping created for {len(partition_to_h5)} partitions")

# Image loading functions
def parse_csv_path(csv_path):
    """Convert CSV path to H5 coordinates"""
    parts = csv_path.split('/')
    if len(parts) >= 4:
        category = parts[0]
        object_id = parts[1] 
        frame_num = int(parts[3].split('.')[0]) - 1  # 0-based index
        return category, object_id, frame_num
    return None, None, None

def load_image_from_h5(h5_file, category, object_id, frame_index):
    """Load and decode image from H5 file"""
    try:
        with h5py.File(h5_file, 'r') as f:
            if category in f and object_id in f[category]:
                obj_data = f[category][object_id]
                if isinstance(obj_data, h5py.Dataset) and frame_index < len(obj_data):
                    img_bytes = obj_data[frame_index]
                    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    return img_array is not None
    except Exception:
        pass
    return False

# Verification process
print(f"\n=== STARTING VERIFICATION PROCESS ===")
print(f"This will take a while... estimated time: {len(df) / 1000 / 60:.1f} minutes")

verified_indices = []
failed_indices = []
stats = defaultdict(int)

start_time = time.time()
last_report = start_time

for i, (idx, row) in enumerate(df.iterrows()):
    # Progress reporting
    current_time = time.time()
    if current_time - last_report > 30:  # Report every 30 seconds
        elapsed = current_time - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta_seconds = (len(df) - i) / rate if rate > 0 else 0
        eta_minutes = eta_seconds / 60
        
        progress_pct = (i / len(df)) * 100
        print(f"Progress: {i:,}/{len(df):,} ({progress_pct:.1f}%) - {rate:.0f} imgs/sec - ETA: {eta_minutes:.1f}min")
        print(f"  Success rate so far: {len(verified_indices)}/{i} ({len(verified_indices)/max(i,1)*100:.1f}%)")
        last_report = current_time
    
    # Parse path
    if pd.isna(row['path']) or row['path'] is None:
        stats['null_path'] += 1
        failed_indices.append(idx)
        continue
        
    category, object_id, frame_index = parse_csv_path(row['path'])
    if category is None:
        stats['parse_error'] += 1
        failed_indices.append(idx)
        continue
    
    # Get H5 file
    if pd.isna(row['partition']) or row['partition'] is None:
        stats['null_partition'] += 1
        failed_indices.append(idx)
        continue
        
    partition_name = str(row['partition'])
    h5_file = partition_to_h5.get(partition_name) or partition_to_h5.get(urllib.parse.unquote(partition_name))
    
    if h5_file is None:
        stats['partition_not_found'] += 1
        failed_indices.append(idx)
        continue
    
    # Test image loading
    if load_image_from_h5(h5_file, category, object_id, frame_index):
        stats['success'] += 1
        verified_indices.append(idx)
    else:
        stats['load_failed'] += 1
        failed_indices.append(idx)

total_time = time.time() - start_time
print(f"\nCompleted in {total_time/60:.1f} minutes")

# Results summary
print(f"\n=== VERIFICATION RESULTS ===")
total_checked = len(df)
for stat_name, count in stats.items():
    pct = (count / total_checked) * 100
    print(f"{stat_name}: {count:,} ({pct:.1f}%)")

success_rate = stats['success'] / total_checked * 100
print(f"\nOverall success rate: {success_rate:.1f}%")

# Create verified dataset
if len(verified_indices) > 0:
    print(f"\n=== CREATING VERIFIED DATASET ===")
    verified_df = df.loc[verified_indices].copy().reset_index(drop=True)
    
    print(f"Verified dataset:")
    print(f"  Total images: {len(verified_df):,}")
    print(f"  Total clips: {verified_df['clip_id'].nunique():,}")
    print(f"  Total classes: {verified_df['label'].nunique()}")
    
    # Check for incomplete clips (clips missing frames)
    print(f"\n=== CHECKING CLIP COMPLETENESS ===")
    
    # Group by clip and check frame sequences
    clip_analysis = verified_df.groupby('clip_id').agg({
        'frame_in_clip': ['count', 'min', 'max'],
        'clip_length': 'first',
        'label': 'first'
    })
    clip_analysis.columns = ['actual_frames', 'min_frame', 'max_frame', 'expected_frames', 'label']
    
    # Find clips with missing frames
    complete_clips = clip_analysis[
        (clip_analysis['actual_frames'] == clip_analysis['expected_frames']) &
        (clip_analysis['min_frame'] == 0) &
        (clip_analysis['max_frame'] == clip_analysis['expected_frames'] - 1)
    ]
    
    incomplete_clips = clip_analysis[
        (clip_analysis['actual_frames'] != clip_analysis['expected_frames']) |
        (clip_analysis['min_frame'] != 0) |
        (clip_analysis['max_frame'] != clip_analysis['expected_frames'] - 1)
    ]
    
    print(f"Complete clips: {len(complete_clips):,}")
    print(f"Incomplete clips: {len(incomplete_clips):,}")
    
    if len(incomplete_clips) > 0:
        print(f"Examples of incomplete clips:")
        print(incomplete_clips.head())
        
        # Create dataset with only complete clips
        complete_clip_ids = complete_clips.index
        verified_complete_df = verified_df[verified_df['clip_id'].isin(complete_clip_ids)].copy()
        
        print(f"\nDataset with complete clips only:")
        print(f"  Images: {len(verified_complete_df):,}")
        print(f"  Clips: {verified_complete_df['clip_id'].nunique():,}")
        print(f"  Classes: {verified_complete_df['label'].nunique()}")
    else:
        print("✅ All clips are complete!")
        verified_complete_df = verified_df.copy()
    
    # Final statistics
    print(f"\n=== FINAL DATASET STATISTICS ===")
    print(f"Original dataset: {len(df):,} images, {df['clip_id'].nunique():,} clips")
    print(f"Verified dataset: {len(verified_complete_df):,} images, {verified_complete_df['clip_id'].nunique():,} clips")
    print(f"Data retention: {len(verified_complete_df)/len(df)*100:.1f}%")
    
    # Class distribution
    class_dist = verified_complete_df['label'].value_counts().sort_index()
    print(f"\nClass distribution:")
    print(f"  Classes: {len(class_dist)}")
    print(f"  Min samples per class: {class_dist.min()}")
    print(f"  Max samples per class: {class_dist.max()}")
    print(f"  Mean samples per class: {class_dist.mean():.1f}")
    print(f"  Classes with <100 samples: {(class_dist < 100).sum()}")
    
    # Save verified dataset
    output_path = '/home/brothen/solo-learn/mvimagenet_verified_complete.parquet'
    print(f"\n=== SAVING VERIFIED DATASET ===")
    print(f"Saving to: {output_path}")
    
    verified_complete_df.to_parquet(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024**2)
    print(f"✅ Saved! File size: {file_size:.1f} MB")
    
    # Save detailed statistics
    verification_stats = {
        'verification_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'original_images': len(df),
        'verified_images': len(verified_complete_df),
        'original_clips': df['clip_id'].nunique(),
        'verified_clips': verified_complete_df['clip_id'].nunique(),
        'original_classes': df['label'].nunique(),
        'verified_classes': verified_complete_df['label'].nunique(),
        'success_rate': success_rate,
        'data_retention': len(verified_complete_df)/len(df)*100,
        'verification_stats': dict(stats),
        'class_distribution': class_dist.to_dict(),
        'incomplete_clips_removed': len(incomplete_clips)
    }
    
    import json
    stats_path = '/home/brothen/solo-learn/mvimagenet_verification_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(verification_stats, f, indent=2, default=str)
    
    print(f"📊 Verification statistics saved to: {stats_path}")
    
    # Save failed indices for debugging
    failed_df = df.loc[failed_indices].copy()
    failed_path = '/home/brothen/solo-learn/mvimagenet_failed.parquet'
    failed_df.to_parquet(failed_path, index=False)
    print(f"🔍 Failed images saved to: {failed_path}")
    
    print(f"\n🎉 VERIFICATION COMPLETE!")
    print(f"✅ {len(verified_complete_df):,} verified images ready for training")
    print(f"✅ {verified_complete_df['clip_id'].nunique():,} complete clips")
    print(f"✅ {verified_complete_df['label'].nunique()} classes")
    print(f"✅ 100% guaranteed to load during training!")
    
    # Create simple usage example
    usage_example = f'''
# Load verified dataset
import pandas as pd
df = pd.read_parquet('{output_path}')

# Dataset info
print(f"Images: {{len(df):,}}")
print(f"Clips: {{df['clip_id'].nunique():,}}")
print(f"Classes: {{df['label'].nunique()}}")

# Get clips for a specific class
class_0_clips = df[df['label'] == 0]['clip_id'].unique()

# Get temporal sequence for a clip
def get_clip_sequence(clip_id):
    return df[df['clip_id'] == clip_id].sort_values('frame_in_clip')

# Load images using the loader
from mvimagenet_loader import load_mvimagenet_image
# (Use the H5 loader we created earlier)
'''
    
    with open('/home/brothen/solo-learn/usage_example.py', 'w') as f:
        f.write(usage_example)
    
    print(f"📝 Usage example saved to: /home/brothen/solo-learn/usage_example.py")
    
else:
    print("❌ No images verified successfully!")