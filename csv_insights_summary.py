#!/usr/bin/env python3
"""
CSV Insights Summary - Key findings from the comprehensive analysis
"""

import json
import pandas as pd

def load_and_summarize():
    """Load the analysis results and provide key insights"""
    
    print("="*80)
    print("MVIMAGENET CSV DATASET INSIGHTS SUMMARY")
    print("="*80)
    
    # Load the analysis summary
    with open('csv_analysis_summary.json', 'r') as f:
        summary = json.load(f)
    
    file_info = summary['file_info']
    threshold_stats = summary['threshold_stats']
    
    # Calculate totals
    total_rows = sum(info['row_count'] for info in file_info.values())
    total_size_gb = sum(info['size_mb'] for info in file_info.values()) / 1024
    
    print(f"📊 DATASET SCALE:")
    print(f"   Total CSV Files: {len(file_info)}")
    print(f"   Total Rows: {total_rows:,}")
    print(f"   Total Size: {total_size_gb:.1f} GB")
    print(f"   Average Rows per File: {total_rows/len(file_info):,.0f}")
    
    # Header patterns insight
    print(f"\n🔍 HEADER PATTERNS INSIGHT:")
    print(f"   Pattern 1 (22 cols): Basic dataset without 'chosen' flag")
    print(f"   Pattern 2 (23 cols): Dataset with 'chosen' selection flag") 
    print(f"   Pattern 3 (23 cols): Dataset with 'original_frame' info")
    print(f"   Pattern 4 (8 cols): Corrupted file")
    
    # Parameter insights
    print(f"\n🎯 PARAMETER INSIGHTS:")
    print(f"   param1 = 4: Fewer files (3), but MUCH larger (~680K rows each)")
    print(f"   param1 = 7: Most files (72), smaller (~430K rows each)")
    print(f"   → param1=4 might be 'full sequences', param1=7 might be 'subsequences'")
    
    print(f"\n   param2 = 0: {sum(1 for f in file_info.keys() if '_0_' in f or f.endswith('_0.csv'))} files, smaller (~384K rows avg)")
    print(f"   param2 = 1: {sum(1 for f in file_info.keys() if '_1_' in f or f.endswith('_1.csv'))} files, medium (~451K rows avg)")
    print(f"   param2 = 2: {sum(1 for f in file_info.keys() if '_2_' in f or f.endswith('_2.csv'))} files, largest (~743K rows avg)")
    print(f"   → param2 might be train(0)/val(1)/test(2) or different viewpoints")
    
    # Threshold analysis
    print(f"\n📈 THRESHOLD ANALYSIS:")
    for thresh in sorted(threshold_stats.keys()):
        stats = threshold_stats[str(thresh)]
        print(f"   {thresh}: {stats['num_files']} files, {stats['total_rows']:,} total rows")
    
    # Most important insight
    print(f"\n💡 KEY INSIGHT:")
    print(f"   threshold=0.1 has the MOST data: {threshold_stats['0.1']['total_rows']:,} rows (38 files)")
    print(f"   This is likely the main training set!")
    
    # Overlap insight
    print(f"\n🔗 OVERLAP INSIGHT:")
    print(f"   High overlap (98%+) between files with same param1/param2, different thresholds")
    print(f"   → Files are filtered versions of the same source data")
    print(f"   → We can use any threshold level and get the same images, just different amounts")
    
    # Recommendations
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print(f"""
   1. IGNORE the corrupted parquet train file completely
   
   2. USE CSV files as the main dataset source:
      - threshold=0.1: Most balanced (15.8M rows across 38 files)
      - param1=7: More files, good for training variety
      - All param2 values: Use all splits (0,1,2) for maximum data
   
   3. DATASET SPLIT STRATEGY:
      - Training: CSV files (15+ million samples)
      - Validation: dataset_val_all3.parquet (655K samples)  
      - Testing: dataset_test_all3.parquet (648K samples)
   
   4. TOTAL USABLE DATASET: ~17M samples (15.8M train + 1.3M val/test)
      - This is MUCH larger than just using parquet files (1.3M)
      - We get 13x more training data!
   
   5. H5 FILE STRATEGY:
      - Verify all paths in CSV files exist in H5 files
      - Build efficient loaders for the individual H5 files
      - Skip the corrupted main H5 file
   """)
    
    # Create recommended file list
    print(f"\n📋 RECOMMENDED FILES FOR TRAINING:")
    
    recommended_files = []
    for filename, info in file_info.items():
        if filename.startswith("datasetT_0.1_7_") and info['row_count'] > 100000:
            recommended_files.append((filename, info['row_count'], info['size_mb']))
    
    recommended_files.sort(key=lambda x: x[1], reverse=True)  # Sort by row count
    
    total_recommended_rows = sum(r[1] for r in recommended_files)
    print(f"   Files with threshold=0.1, param1=7: {len(recommended_files)} files")
    print(f"   Total training samples: {total_recommended_rows:,}")
    print(f"   Total training size: {sum(r[2] for r in recommended_files):.1f} MB")
    
    for filename, rows, size_mb in recommended_files[:10]:  # Show top 10
        print(f"     {filename}: {rows:,} rows ({size_mb:.1f}MB)")
    
    if len(recommended_files) > 10:
        print(f"     ... and {len(recommended_files)-10} more files")

if __name__ == "__main__":
    load_and_summarize() 