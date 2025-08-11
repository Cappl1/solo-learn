#!/usr/bin/env python3
"""
Comprehensive CSV Analysis for MVImageNet Dataset

This script analyzes all CSV files to understand:
1. Header variations across files
2. Entry counts and sizes
3. Overlap in source images/paths
4. Parameter patterns and what they might represent
5. Data distribution across files
"""

import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_all_csv_headers():
    """Analyze headers across all CSV files to find patterns"""
    print("=" * 80)
    print("COMPREHENSIVE CSV HEADER ANALYSIS")
    print("=" * 80)
    
    data_path = "/home/data/MVImageNet"
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Collect all headers
    header_sets = {}
    file_info = {}
    all_headers = set()
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"Processing {filename}...")
        
        try:
            # Read just the header
            df_sample = pd.read_csv(csv_file, nrows=0)
            headers = list(df_sample.columns)
            header_tuple = tuple(sorted(headers))
            
            # Get basic file info
            file_size_mb = os.path.getsize(csv_file) / (1024**2)
            
            # Count rows (efficient way)
            row_count = sum(1 for line in open(csv_file)) - 1  # subtract header
            
            file_info[filename] = {
                'headers': headers,
                'header_tuple': header_tuple,
                'size_mb': file_size_mb,
                'row_count': row_count,
                'num_columns': len(headers)
            }
            
            all_headers.update(headers)
            
            if header_tuple not in header_sets:
                header_sets[header_tuple] = []
            header_sets[header_tuple].append(filename)
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    print(f"\n📊 HEADER PATTERN ANALYSIS")
    print("-" * 50)
    print(f"Total unique headers found: {len(all_headers)}")
    print(f"Unique header combinations: {len(header_sets)}")
    
    # Show header patterns
    for i, (header_tuple, files) in enumerate(header_sets.items(), 1):
        print(f"\n🔍 Header Pattern {i} ({len(files)} files):")
        print(f"  Columns ({len(header_tuple)}): {list(header_tuple)}")
        print(f"  Files: {files[:3]}{'...' if len(files) > 3 else ''}")
    
    return file_info, header_sets, all_headers

def analyze_csv_overlap_and_content(file_info, sample_size=10):
    """Analyze content overlap and patterns in CSV files"""
    print(f"\n📈 CSV CONTENT AND OVERLAP ANALYSIS")
    print("-" * 50)
    
    data_path = "/home/data/MVImageNet"
    
    # Select sample files from different parameter combinations
    files_by_pattern = defaultdict(list)
    for filename, info in file_info.items():
        if filename.startswith("datasetT_"):
            # Extract pattern
            parts = filename.replace("datasetT_", "").replace(".csv", "").split("_")
            if len(parts) >= 3:
                pattern = f"{parts[0]}_{parts[1]}_{parts[2]}"
                files_by_pattern[pattern].append(filename)
    
    # Sample files from different patterns
    sample_files = []
    for pattern, files in sorted(files_by_pattern.items()):
        sample_files.append(files[0])  # Take first file from each pattern
        if len(sample_files) >= sample_size:
            break
    
    print(f"Analyzing {len(sample_files)} representative files:")
    
    file_data = {}
    all_paths = set()
    path_to_files = defaultdict(set)
    
    for filename in sample_files:
        print(f"\n--- {filename} ---")
        file_path = os.path.join(data_path, filename)
        
        try:
            # Read sample of data
            df = pd.read_csv(file_path, nrows=5000)  # Read more rows for better analysis
            
            info = file_info[filename]
            print(f"  Size: {info['size_mb']:.1f}MB, Rows: {info['row_count']:,}, Cols: {info['num_columns']}")
            
            # Analyze key columns
            analysis = {}
            
            if 'path' in df.columns:
                unique_paths = df['path'].nunique()
                sample_paths = df['path'].head(5).tolist()
                analysis['paths'] = {
                    'unique_count': unique_paths,
                    'sample': sample_paths,
                    'all_paths': set(df['path'].tolist())
                }
                
                # Track path overlap
                file_paths = set(df['path'].tolist())
                all_paths.update(file_paths)
                for path in file_paths:
                    path_to_files[path].add(filename)
                
                print(f"  Unique paths: {unique_paths:,}")
                print(f"  Sample paths: {sample_paths[:2]}")
            
            if 'category' in df.columns:
                cat_counts = df['category'].value_counts()
                analysis['categories'] = {
                    'unique_count': len(cat_counts),
                    'distribution': dict(cat_counts.head()),
                    'total_samples': len(df)
                }
                print(f"  Categories: {len(cat_counts)} unique, top: {dict(cat_counts.head(3))}")
            
            if 'object' in df.columns:
                obj_counts = df['object'].value_counts()
                analysis['objects'] = {
                    'unique_count': len(obj_counts),
                    'frames_per_object': obj_counts.describe()
                }
                print(f"  Objects: {len(obj_counts):,} unique, avg frames/obj: {obj_counts.mean():.1f}")
            
            # Analyze parameter meanings by looking at filename
            parts = filename.replace("datasetT_", "").replace(".csv", "").split("_")
            if len(parts) >= 3:
                threshold, param1, param2 = parts[0], parts[1], parts[2]
                analysis['parameters'] = {
                    'threshold': threshold,
                    'param1': param1,
                    'param2': param2
                }
                print(f"  Parameters: threshold={threshold}, param1={param1}, param2={param2}")
            
            file_data[filename] = analysis
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Analyze path overlap
    print(f"\n🔗 PATH OVERLAP ANALYSIS")
    print("-" * 30)
    print(f"Total unique paths across samples: {len(all_paths):,}")
    
    # Find paths that appear in multiple files
    overlapping_paths = {path: files for path, files in path_to_files.items() if len(files) > 1}
    print(f"Paths appearing in multiple files: {len(overlapping_paths):,}")
    
    if overlapping_paths:
        # Sample some overlapping paths
        sample_overlaps = list(overlapping_paths.items())[:5]
        for path, files in sample_overlaps:
            print(f"  {path}: appears in {len(files)} files")
    
    # Calculate overlap matrix between files
    file_pairs_overlap = {}
    sample_file_list = list(file_data.keys())
    for i, file1 in enumerate(sample_file_list):
        for j, file2 in enumerate(sample_file_list[i+1:], i+1):
            if 'paths' in file_data[file1] and 'paths' in file_data[file2]:
                paths1 = file_data[file1]['paths']['all_paths']
                paths2 = file_data[file2]['paths']['all_paths']
                overlap = len(paths1.intersection(paths2))
                total = len(paths1.union(paths2))
                overlap_ratio = overlap / total if total > 0 else 0
                file_pairs_overlap[(file1, file2)] = {
                    'overlap_count': overlap,
                    'overlap_ratio': overlap_ratio
                }
    
    # Show highest overlaps
    sorted_overlaps = sorted(file_pairs_overlap.items(), 
                           key=lambda x: x[1]['overlap_ratio'], reverse=True)
    
    print(f"\n📊 HIGHEST FILE OVERLAPS (top 5):")
    for (file1, file2), overlap_info in sorted_overlaps[:5]:
        ratio = overlap_info['overlap_ratio']
        count = overlap_info['overlap_count']
        print(f"  {os.path.basename(file1)} ↔ {os.path.basename(file2)}: {ratio:.3f} ({count:,} paths)")
    
    return file_data, path_to_files

def analyze_parameter_patterns(file_info):
    """Analyze what the different parameters might represent"""
    print(f"\n🎯 PARAMETER PATTERN ANALYSIS")
    print("-" * 50)
    
    # Group files by parameters
    param_patterns = defaultdict(lambda: defaultdict(list))
    size_by_params = defaultdict(list)
    count_by_params = defaultdict(list)
    
    for filename, info in file_info.items():
        if filename.startswith("datasetT_"):
            parts = filename.replace("datasetT_", "").replace(".csv", "").split("_")
            if len(parts) >= 3:
                threshold, param1, param2 = parts[0], parts[1], parts[2]
                
                param_patterns['threshold'][threshold].append(filename)
                param_patterns['param1'][param1].append(filename)
                param_patterns['param2'][param2].append(filename)
                
                # Track sizes and counts by parameters
                size_by_params[f"thresh_{threshold}"].append(info['size_mb'])
                count_by_params[f"thresh_{threshold}"].append(info['row_count'])
    
    print("📋 Parameter Value Distributions:")
    for param_type, param_dict in param_patterns.items():
        print(f"\n  {param_type.upper()}:")
        for value, files in sorted(param_dict.items()):
            avg_size = np.mean([file_info[f]['size_mb'] for f in files])
            avg_count = np.mean([file_info[f]['row_count'] for f in files])
            print(f"    {value}: {len(files)} files, avg {avg_size:.1f}MB, avg {avg_count:,.0f} rows")
    
    # Analyze trends
    print(f"\n📈 PARAMETER TRENDS:")
    
    # Threshold analysis
    threshold_stats = {}
    for threshold, files in param_patterns['threshold'].items():
        sizes = [file_info[f]['size_mb'] for f in files]
        counts = [file_info[f]['row_count'] for f in files]
        threshold_stats[float(threshold)] = {
            'num_files': len(files),
            'avg_size_mb': np.mean(sizes),
            'avg_rows': np.mean(counts),
            'total_rows': sum(counts)
        }
    
    print("  Threshold Analysis (sorted by value):")
    for threshold in sorted(threshold_stats.keys()):
        stats = threshold_stats[threshold]
        print(f"    {threshold}: {stats['num_files']} files, "
              f"{stats['total_rows']:,} total rows, "
              f"{stats['avg_size_mb']:.1f}MB avg")
    
    # Look for patterns
    print(f"\n🔍 PATTERN INSIGHTS:")
    
    # Check if higher thresholds = fewer samples (filtering effect)
    thresholds_sorted = sorted(threshold_stats.keys())
    rows_trend = [threshold_stats[t]['avg_rows'] for t in thresholds_sorted]
    
    print(f"  Threshold vs Avg Rows: {dict(zip(thresholds_sorted, [f'{r:,.0f}' for r in rows_trend]))}")
    
    if len(rows_trend) > 1:
        if rows_trend[0] > rows_trend[-1]:
            print(f"  → Higher thresholds seem to filter data (fewer rows)")
        elif rows_trend[0] < rows_trend[-1]:
            print(f"  → Higher thresholds seem to expand data (more rows)")
        else:
            print(f"  → No clear threshold-based filtering pattern")
    
    return param_patterns, threshold_stats

def generate_csv_insights_report():
    """Generate final insights about the CSV structure"""
    print(f"\n" + "="*80)
    print("CSV INSIGHTS AND RECOMMENDATIONS")
    print("="*80)
    
    print("""
📋 KEY FINDINGS:

1. PARAMETER MEANINGS (hypothesis):
   - threshold: Likely filtering threshold (0.0 = no filter, 1.0 = strict filter)
   - param1: Possibly sequence length or window size (4 vs 7)
   - param2: Possibly data split or variant (0, 1, 2)

2. DATA ORGANIZATION:
   - Most files follow pattern: datasetT_{threshold}_{param1}_{param2}
   - Different thresholds likely represent different filtering levels
   - Files with same threshold may contain similar data with different processing

3. POTENTIAL USAGE:
   - Use threshold=0.1 files as they are most common (good balance)
   - param1=7 seems more common than param1=4 (possibly better sequences)
   - param2 might represent train/val/test splits or different views

4. OVERLAP PATTERNS:
   - High overlap suggests files are filtered versions of same source
   - Low overlap suggests different data sources or processing

🎯 NEXT STEPS:
1. Pick representative files from different parameter combinations
2. Create unified dataset loader that can handle multiple CSV sources
3. Use CSV files to supplement corrupted parquet training data
4. Verify H5 file accessibility for paths found in CSVs
""")

def main():
    """Run comprehensive CSV analysis"""
    print("Starting comprehensive CSV analysis...")
    
    try:
        # Step 1: Analyze headers
        file_info, header_sets, all_headers = analyze_all_csv_headers()
        
        # Step 2: Analyze content and overlap
        file_data, path_to_files = analyze_csv_overlap_and_content(file_info, sample_size=15)
        
        # Step 3: Analyze parameter patterns
        param_patterns, threshold_stats = analyze_parameter_patterns(file_info)
        
        # Step 4: Generate insights
        generate_csv_insights_report()
        
        print(f"\n✅ Comprehensive CSV analysis completed!")
        
        # Save summary
        summary = {
            'total_files': len(file_info),
            'header_patterns': len(header_sets),
            'file_info': {k: {
                'size_mb': v['size_mb'], 
                'row_count': v['row_count'],
                'num_columns': v['num_columns']
            } for k, v in file_info.items()},
            'threshold_stats': threshold_stats
        }
        
        import json
        with open('csv_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Summary saved to csv_analysis_summary.json")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 