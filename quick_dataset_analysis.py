#!/usr/bin/env python3
import os
import pandas as pd
import h5py
import glob

def quick_analysis():
    data_path = "/home/data/MVImageNet"
    
    print("=== MVImageNet Dataset Quick Analysis ===")
    
    # Count files by type
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    h5_files = glob.glob(os.path.join(data_path, "*.h5"))
    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
    
    print(f"CSV files: {len(csv_files)}")
    print(f"H5 files: {len(h5_files)}")
    print(f"Parquet files: {len(parquet_files)}")
    
    # Analyze first CSV file
    if csv_files:
        print(f"\nAnalyzing first CSV: {os.path.basename(csv_files[0])}")
        try:
            df = pd.read_csv(csv_files[0], nrows=100)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample data:\n{df.head()}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    # Analyze parquet files
    if parquet_files:
        for pf in parquet_files:
            print(f"\nAnalyzing parquet: {os.path.basename(pf)}")
            try:
                df = pd.read_parquet(pf)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"Sample data:\n{df.head()}")
            except Exception as e:
                print(f"Error reading parquet: {e}")

if __name__ == "__main__":
    quick_analysis() 