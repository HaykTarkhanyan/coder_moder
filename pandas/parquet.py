import os
import pandas as pd
from tqdm import tqdm

def convert_csv_to_parquet(directory_convert, directory_save=None):
    """
    !pip install pyarrow
    """
    if directory_save is None:
        directory_save = directory_convert + "_parquet"
    
    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    
    
    for filename in tqdm(os.listdir(directory_convert)):
        try:
            if filename.endswith(".csv"):
                csv_file = os.path.join(directory_convert, filename)
                parquet_file = os.path.join(directory_save, filename.replace(".csv", ".parquet"))
                df = pd.read_csv(csv_file)
                df.to_parquet(parquet_file)
        except Exception as e:
            print(f"Error converting {filename} to parquet: {e}")


def compare_file_sizes_avg(dir_parquet, dir_csv):
    csv_files = os.listdir(dir_csv)
    parquet_files = os.listdir(dir_parquet)
    
    csv_size = 0
    parquet_size = 0
    
    for filename in csv_files:
        if filename.endswith(".csv"):
            size = os.path.getsize(os.path.join(dir_csv, filename))
            csv_size += size 
            print(filename, size)
    
    for filename in parquet_files:
        if filename.endswith(".parquet"):
            parquet_size += os.path.getsize(os.path.join(dir_parquet, filename))
    
    print(f"CSV size: {csv_size / len(csv_files) // 1024 // 1024} MB")
    print(f"Parquet size: {parquet_size / len(parquet_files) // 1024 // 1024} MB")
    print(f"Parquet is {csv_size / parquet_size} times smaller than CSV")

    
import time 

def compare_read_times(dir_csv, dir_parquet):
    csv_files = os.listdir(dir_csv)
    parquet_files = os.listdir(dir_parquet)
    
    csv_time = 0
    parquet_time = 0
    
    for filename in csv_files:
        if filename.endswith(".csv"):
            csv_file = os.path.join(dir_csv, filename)
            start = time.time()
            df = pd.read_csv(csv_file)
            csv_time += time.time() - start
    
    for filename in parquet_files:
        if filename.endswith(".parquet"):
            parquet_file = os.path.join(dir_parquet, filename)
            start = time.time()
            df = pd.read_parquet(parquet_file)
            parquet_time += time.time() - start
    
    print(f"CSV read time: {csv_time}")
    print(f"Parquet read time: {parquet_time}")
    print(f"Parquet is {csv_time / parquet_time} times faster than CSV")