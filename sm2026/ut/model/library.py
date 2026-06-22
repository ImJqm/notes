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

# Example usage (optional - you can remove this if you just want the function definition)
# df_processed = process_block_io_flags(df.clone()) # Use a clone to avoid modifying the original df
# display(df_processed.columns)

def time_model_execution(model, inputs_np_array, warm_up_iterations=10_000):
    def run_function(model, input_np_array):
      return model.predict(input_np_array.reshape(1,-1))

    print(f"Warming up the model with {warm_up_iterations} iterations...")
    for i in range(warm_up_iterations):
        single_input = inputs_np_array[i % len(inputs_np_array)]
        run_function(model, single_input)
    print("Warm-up complete.")

    print("Timing single prediction in nanoseconds...")
    start_time = time.time()
    for single_inputs in inputs_np_array:
        run_function(model, single_input)
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_ns = execution_time_seconds * 1e9 # Convert to nanoseconds
    return execution_time_ns / inputs_np_array.shape[0]

# Example usage (assuming 'model' and 'test_features_np' are available from previous cells):
# execution_time_ns = time_model_execution(model, test_features_np)
# print(f"Execution time for a single prediction: {execution_time_ns:.2f} nanoseconds")