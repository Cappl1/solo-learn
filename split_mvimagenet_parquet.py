#!/usr/bin/env python3
"""
Script to split MVImageNet parquet file into separate train/validation files.
This follows the Core50 pattern where train and validation use different background sessions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def split_mvimagenet_parquet(
    input_parquet: str,
    output_dir: str = ".",
    val_split: float = 0.05,
    min_clip_length: int = 10,
    random_seed: int = 42
):
    """
    Split MVImageNet parquet file into train/val splits by clips.
    
    Args:
        input_parquet: Path to input parquet file
        output_dir: Directory to save train/val parquet files
        val_split: Fraction for validation (default 0.05 = 5%)
        min_clip_length: Minimum frames per clip to include
        random_seed: Random seed for reproducible splits
    """
    
    print(f"Loading MVImageNet metadata from: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df):,} total images")
    
    # Filter by minimum clip length
    if min_clip_length > 0:
        clip_lengths = df.groupby('clip_id')['clip_length'].first()
        valid_clips = clip_lengths[clip_lengths >= min_clip_length].index
        df = df[df['clip_id'].isin(valid_clips)].reset_index(drop=True)
        print(f"After filtering (min_length={min_clip_length}): {len(df):,} images")
    
    # Get unique clips with their labels for stratified splitting
    clips_df = df.groupby('clip_id').agg({
        'label': 'first',
        'clip_length': 'first'
    }).reset_index()
    
    print(f"Total clips: {len(clips_df):,}")
    print(f"Unique classes: {clips_df['label'].nunique()}")
    
    # Analyze class distribution before splitting
    class_counts = clips_df['label'].value_counts().sort_index()
    print(f"Class distribution: min={class_counts.min()}, max={class_counts.max()}, mean={class_counts.mean():.1f}")
    
    # Check for classes with very few clips (problematic for stratification)
    min_clips_per_class = max(2, int(1 / val_split))  # Need at least 2 clips per class, or 1/val_split
    sparse_classes = class_counts[class_counts < min_clips_per_class]
    
    if len(sparse_classes) > 0:
        print(f"Found {len(sparse_classes)} classes with < {min_clips_per_class} clips:")
        print(f"  Sparse classes: {sparse_classes.to_dict()}")
        
        # Strategy: Move sparse classes to training set, stratify the rest
        sparse_class_labels = sparse_classes.index.tolist()
        sparse_clips = clips_df[clips_df['label'].isin(sparse_class_labels)]['clip_id'].tolist()
        stratifiable_clips_df = clips_df[~clips_df['label'].isin(sparse_class_labels)]
        
        print(f"  Moving {len(sparse_clips)} clips from sparse classes to training set")
        print(f"  Stratifying remaining {len(stratifiable_clips_df)} clips from {stratifiable_clips_df['label'].nunique()} classes")
        
        if len(stratifiable_clips_df) > 0:
            try:
                # Stratified split on the remaining clips
                strat_train_clips, strat_val_clips = train_test_split(
                    stratifiable_clips_df['clip_id'].tolist(),
                    test_size=val_split,
                    stratify=stratifiable_clips_df['label'].tolist(),
                    random_state=random_seed
                )
                
                # Combine: all sparse clips go to training, stratified clips are split
                train_clips = sparse_clips + strat_train_clips
                val_clips = strat_val_clips
                
                print(f"✓ Used hybrid approach: sparse classes to train + stratified split for rest")
                
            except ValueError as e:
                print(f"⚠️  Hybrid stratification also failed: {e}")
                print("⚠️  Falling back to random split")
                train_clips, val_clips = train_test_split(
                    clips_df['clip_id'].tolist(),
                    test_size=val_split,
                    random_state=random_seed
                )
        else:
            print("⚠️  All classes are sparse, using random split")
            train_clips, val_clips = train_test_split(
                clips_df['clip_id'].tolist(),
                test_size=val_split,
                random_state=random_seed
            )
    else:
        # No sparse classes, try pure stratified split
        try:
            train_clips, val_clips = train_test_split(
                clips_df['clip_id'].tolist(),
                test_size=val_split,
                stratify=clips_df['label'].tolist(),
                random_state=random_seed
            )
            print(f"✓ Used pure stratified split")
        except ValueError as e:
            print(f"⚠️  Pure stratified split failed: {e}")
            print("⚠️  Falling back to random split")
            train_clips, val_clips = train_test_split(
                clips_df['clip_id'].tolist(),
                test_size=val_split,
                random_state=random_seed
            )
    
    # Filter dataframes by selected clips
    train_df = df[df['clip_id'].isin(train_clips)].reset_index(drop=True)
    val_df = df[df['clip_id'].isin(val_clips)].reset_index(drop=True)
    
    print(f"\nSplit results:")
    print(f"  Training: {len(train_df):,} images from {len(train_clips):,} clips")
    print(f"  Validation: {len(val_df):,} images from {len(val_clips):,} clips")
    
    # Verify class distribution
    train_classes = train_df['label'].value_counts().sort_index()
    val_classes = val_df['label'].value_counts().sort_index()
    
    print(f"\nClass distribution:")
    print(f"  Training classes: {len(train_classes)} (min: {train_classes.min()}, max: {train_classes.max()})")
    print(f"  Validation classes: {len(val_classes)} (min: {val_classes.min()}, max: {val_classes.max()})")
    
    # Check class overlap
    train_class_set = set(train_classes.index)
    val_class_set = set(val_classes.index)
    overlap = train_class_set & val_class_set
    print(f"  Overlapping classes: {len(overlap)}/{len(train_class_set | val_class_set)} ({len(overlap)/len(train_class_set | val_class_set)*100:.1f}%)")
    
    # Save split files
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    train_path = output_dir / "mvimagenet_train.parquet"
    val_path = output_dir / "mvimagenet_val.parquet"
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    print(f"\n✓ Saved splits:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {val_path}")
    
    # Create a summary file
    summary = {
        'input_file': str(input_parquet),
        'total_images': len(df),
        'total_clips': len(clips_df),
        'unique_classes': clips_df['label'].nunique(),
        'min_clip_length': min_clip_length,
        'val_split': val_split,
        'random_seed': random_seed,
        'overlapping_classes': len(overlap),
        'train': {
            'images': len(train_df),
            'clips': len(train_clips),
            'classes': len(train_classes),
            'file': str(train_path)
        },
        'val': {
            'images': len(val_df),
            'clips': len(val_clips),
            'classes': len(val_classes),
            'file': str(val_path)
        }
    }
    
    summary_path = output_dir / "mvimagenet_split_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary: {summary_path}")
    
    return train_path, val_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split MVImageNet parquet into train/val")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split fraction")
    parser.add_argument("--min-clip-length", type=int, default=10, help="Minimum clip length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    split_mvimagenet_parquet(
        input_parquet=args.input,
        output_dir=args.output_dir,
        val_split=args.val_split,
        min_clip_length=args.min_clip_length,
        random_seed=args.seed
    ) 