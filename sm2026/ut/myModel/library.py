import os
import urllib.request
import zipfile
import polars as pl
import time

def process_block_io_flags(df: pl.DataFrame) -> pl.DataFrame:
    # Identify unique components
    unique_flags = df['block_io_flags_string'].str.split('-').explode().unique().to_list()

    # Create new columns
    for flag in unique_flags:
        df = df.with_columns(
            pl.col('block_io_flags_string').str.contains(flag).alias(flag)
        )

    # Drop original column
    df = df.drop('block_io_flags_string')

    return df

def time_model_execution(model, inputs_np_array, warm_up_iterations=10000):
    def run_function(model, input_np_array):
      return model.predict(input_np_array.reshape(1,-1))

    print(f"Warming up the model with {warm_up_iterations} iterations...")
    for i in range(warm_up_iterations):
        single_input = inputs_np_array[i % len(inputs_np_array)]
        run_function(model, single_input)
    print("Warm-up complete.")

    print("Timing single prediction in nanoseconds...")
    start_time = time.time()
    model.predict(inputs_np_array)
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_ns = execution_time_seconds * 1e9 # Convert to nanoseconds
    return execution_time_ns / inputs_np_array.shape[0]


def download_data():
    zip_file_url = "https://www.cs.utexas.edu/~amp/rainsong_data.zip"
    zip_file_name = "rainsong_data.zip"

    # Check if the zip file already exists
    if not os.path.exists(zip_file_name):
        print(f"Downloading {zip_file_name}...")
        urllib.request.urlretrieve(zip_file_url, zip_file_name)
        print("Download complete.")
    else:
        print(f"{zip_file_name} already exists. Skipping download.")

    # Unzip the file into the current directory
    if not os.path.exists("rainsong_labeled"):
        print(f"Unzipping {zip_file_name}...")
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Unzipping complete.")
    else:
        print(f"rainsong_labeled already exists. Skipping unzip.")
