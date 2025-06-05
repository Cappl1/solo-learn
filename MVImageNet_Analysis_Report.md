# MVImageNet Dataset Analysis Report

## Overview
The MVImageNet dataset is a **multi-view video dataset with 6DOF pose/camera tracking** containing images organized by objects/scenes with frame sequences.

## Dataset Structure

### File Organization
- **CSV Files**: 75 files with different threshold/parameter combinations (~7.5GB total)
- **H5 Files**: 44 files containing image data (~3GB each, ~132GB total)
- **Parquet Files**: 3 main files (train/test/val splits)
  - `dataset_val_all3.parquet`: 654,768 samples (25.3MB)
  - `dataset_test_all3.parquet`: 647,574 samples (25.1MB) 
  - `dataset_train_all3.parquet`: **CORRUPTED** (189MB)
- **Masked Dataset**: Additional preprocessed data in `/masked_dataset/`

### Data Schema

#### Core Metadata Columns
- `path`: Relative path to images (e.g., `'0/0000f5bc/images/001.jpg'`)
- `category`: Object category (0-based integer, 227 unique categories)
- `object`: Unique object identifier (hex string, ~22K unique objects)
- `frame`: Frame number in sequence (0-40 range)
- `index`: Global sample index
- `partition`: Data split identifier
- `length`: Sequence length
- `partition_cam`: Camera partition info

#### 6DOF Pose Data
- `q0, q1, q2, q3`: Quaternion rotation components
- `t0, t1, t2`: Translation vector components

#### Additional CSV Columns
- `context`, `category_int`, `original_index`
- `crop_id`, `next_obj`, `prev_obj`: Tracking information
- `chosen`: Selection flag (some files only)

## Key Statistics

### Dataset Size
- **Total Samples**: ~1.3M (val + test, train corrupted)
- **Objects**: ~22,000 unique objects
- **Categories**: 227 unique categories
- **Frame Sequences**: 28-32 frames per object on average

### Category Distribution (Top 10)
- Category 144: ~9K samples
- Category 0: ~8.5K samples  
- Category 129: ~8K samples
- Category 1: ~7.5K samples
- Category 156: ~7K samples

### Pose Data Ranges
- **Quaternions (q0-q3)**: Normalized rotation data
  - q0: [-0.500, 1.000] (primary component)
  - q1: [-0.755, 0.842] 
  - q2: [-0.859, 0.982]
  - q3: [-0.863, 0.997]

- **Translation (t0-t2)**: 
  - **Note**: Val data has outliers with very large values (millions)
  - Test data has reasonable ranges: [-415, 115], [-14, 1067], [-918, 21]

## CSV File Parameter Space

### Threshold Values
`['0.0', '0.005', '0.01', '0.05', '0.1', '0.3', '0.5', '0.7', '1.0']`

### Parameter Combinations
- **param1**: `['4', '7']` (likely related to sequence/window size)
- **param2**: `['0', '1', '2']` (likely related to split/variant)

### Most Common Patterns
- `0.1_7_0`: 21 files (most common)
- `0.1_7_1`: 15 files  
- `0.5_7_0`: 4 files
- `1.0_7_0`: 5 files

## H5 File Structure

### Organization
- **Root Level**: Category directories (numerical IDs)
- **Second Level**: Object subdirectories (hex IDs)
- **Data**: Variable-length byte strings (compressed images)

### Issues Identified
- Main `data_all.h5` file has corruption issues
- Individual H5 files are functional (~3GB each)
- Images stored as compressed byte strings in datasets

## Recommendations for Usage

### 1. **For Metadata and Pose Information**
Use the **parquet files** (test/val) as they provide clean, efficient access:
```python
df_test = pd.read_parquet('/home/data/MVImageNet/dataset_test_all3.parquet')
df_val = pd.read_parquet('/home/data/MVImageNet/dataset_val_all3.parquet')
```

### 2. **For Image Data**
Use the individual **H5 files** (not the corrupted main file):
```python
import h5py
with h5py.File('/home/data/MVImageNet/dataf8446572...ff73.h5', 'r') as f:
    # Navigate: f[category][object] to get image data
```

### 3. **For Filtered/Processed Data**
Use **CSV files** with specific threshold parameters based on your needs:
- For loose filtering: `datasetT_0.1_7_0_*.csv`
- For strict filtering: `datasetT_1.0_7_0_*.csv`

### 4. **Handle Data Quality Issues**
- Skip corrupted train parquet file
- Be aware of outliers in translation values (val data)
- Test H5 file accessibility before batch processing

## Next Steps

1. **Fix Training Data**: Investigate train parquet corruption
2. **Data Cleaning**: Address translation outliers in validation set
3. **Create Data Loaders**: Build efficient loaders for your specific use case
4. **Verify Image Quality**: Sample and verify H5 image data integrity

## Technical Notes

- **Total Dataset Size**: ~140GB (mostly H5 image data)
- **Recommended Splits**: Use test/val from parquet, create custom train split from CSV
- **Pose Format**: Standard quaternion + translation 6DOF representation
- **Image Format**: JPEG compressed, stored as byte strings in H5 