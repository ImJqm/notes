import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import norm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

# Model and metric imports
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Custom method imports
from library import process_block_io_flags, time_model_execution


df_read = pl.read_parquet('1.parquet')
print(df_read.columns)


# Only finished I/Os have a real latency we can "look back" at
df_finished = df_read.filter(
    pl.col("finish_ts_uptime_us").is_not_null()
).select([
    "device",
    "ts_uptime_us",
    pl.col("block_io_latency_us").alias("last_latency_block_io_us"),
])

# Sort our dataframe by ts_uptime_us
df_read_sorted = df_read.sort("ts_uptime_us")
df_finished_sorted = df_finished.sort("ts_uptime_us")

# For each row, find the most recently finished read on the same device
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_with_history = df_read_sorted.join_asof(
        df_finished_sorted,
        on="ts_uptime_us",
        by="device",
        strategy="backward",
    )

# Display our new feature
# display(df_with_history.select(
#    ["device", "ts_uptime_us", "last_latency_block_io_us"]
# ).head())

# Drop columns that represent future information or unused target labels
df_drop = df_with_history.drop([
    'collection_id', 'block_io_latency_us', 'block_latency_us',
    'measured_latency_us', 'label85', 'label95', 'finish_ts_uptime_us'
])

# Seperate flags into discrete columns
df = process_block_io_flags(df_drop)

# Only keep 'Read' operations
df_reads = df.filter(pl.col('Read'))

# Handle the the "no previous read yet" rows with median latency
median_latency = df_reads["last_latency_block_io_us"].median()
df_reads = df_reads.with_columns(
    pl.col("last_latency_block_io_us").fill_null(median_latency)
)


# Add a row index column to the DataFrame
df_reads_indexed = df_reads.with_row_index()

# Split the data into training and testing sets (80-20 split)
train_df = df_reads_indexed.sample(
    fraction=0.8, with_replacement=False, seed=42)
test_df = df_reads_indexed.join(
    train_df.select('index'), on='index', how='anti')

# Drop the index column from the split dataframes
train_df = train_df.drop('index')
test_df = test_df.drop('index')

# Confirm our split
print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# Seperate our target
train_target_np = train_df['label90'].to_numpy()
test_target_np = test_df['label90'].to_numpy()

# Two versions of the features: with and without our new history feature
train_features_full = train_df.drop('label90')
test_features_full = test_df.drop('label90')

train_features_baseline = train_features_full.drop('last_latency_block_io_us')
test_features_baseline = test_features_full.drop('last_latency_block_io_us')

# Convert training sets into NumPy arrays
train_full_np = train_features_full.to_numpy()
test_full_np = test_features_full.to_numpy()

train_baseline_np = train_features_baseline.to_numpy()
test_baseline_np = test_features_baseline.to_numpy()

# Normalize our data with history
norm_np = np.abs(train_full_np).max(axis=0) + 0.0001
train_full_np = train_full_np / norm_np
test_full_np = test_full_np / norm_np

# Normalize our data without history
norm_np = np.abs(train_baseline_np).max(axis=0) + 0.0001
train_baseline_np = train_baseline_np / norm_np
test_baseline_np = test_baseline_np / norm_np


# Calculate accuracy and speed evaluation metrics
def evaluate_model(name, model, features_np, target_np):
    predictions = model.predict(features_np)

    accuracy = accuracy_score(target_np, predictions)
    precision = precision_score(target_np, predictions)
    recall = recall_score(target_np, predictions)
    f1 = f1_score(target_np, predictions)
    conf_matrix = confusion_matrix(target_np, predictions)
    ts_ns = time_model_execution(model, features_np)

    print()
    print(f"--- {name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Time per prediction: {ts_ns:.2f}ns, {ts_ns/1000:.2f}us")
    print("Confusion Matrix:")
    print(conf_matrix)
    print()

    return {"name": name, "accuracy": accuracy, "f1": f1, "time_ns": ts_ns}


# We'll collect every model's results here for a final comparison
results = []


# Define the MLPClassifier model with no hidden layers
# baseline_percept = MLPClassifier(hidden_layer_sizes=(
# ), activation='relu', solver='adam', max_iter=500, random_state=42)
#
# Train the baseline Perceptron
# baseline_percept.fit(train_baseline_np, train_target_np)
#
# Save our results
# results.append(evaluate_model("Baseline Perceptron",
#               baseline_percept, test_baseline_np, test_target_np))


# CHANGE THIS to your K value (hint: its probably not 1)
best_k = 9


train_features_full = train_features_full.drop('sector')
test_features_full = test_features_full.drop('sector')

train_features_full = train_features_full.drop('cpu')
test_features_full = test_features_full.drop('cpu')


train_features_full = train_features_full.drop('segments')
test_features_full = test_features_full.drop('segments')

train_features_full = train_features_full.drop('block_io_flags')
test_features_full = test_features_full.drop('block_io_flags')

train_full_np = train_features_full.to_numpy()
test_full_np = test_features_full.to_numpy()


print(train_features_full.columns)


# Uncomment and try one of these out, or add your own code to explore how your decisions impact performance

norm_np = np.abs(train_full_np).max(axis=0) + 0.0001
train_full_np = train_full_np / norm_np
test_full_np = test_full_np / norm_np

new_model = HistGradientBoostingClassifier(max_iter=500)
new_model.fit(train_full_np, train_target_np)
results.append(evaluate_model(
    f"[HistGradientBoostingClassifier]", new_model, test_full_np, test_target_np))

# View the results of all your model calls up to this point
for result in results:
    print(f"{result['name']}  accuracy = {result['accuracy']:.4f}  f1 = {
          result['f1']:.4f}   speed = {result['time_ns']/1000:.2f}us")
