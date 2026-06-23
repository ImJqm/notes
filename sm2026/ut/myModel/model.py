import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import warnings

# Model and metric imports
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Custom method imports
from library import process_block_io_flags, time_model_execution, download_data

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

def handleDf(df):
    df_finished = df.filter(
        pl.col("finish_ts_uptime_us").is_not_null()
    ).select([
        "device",
        pl.col("ts_uptime_us") + 1,
        pl.col("block_io_latency_us").alias("last_latency_block_io_us") ,
    ])
    
    # Sort our dataframe by ts_uptime_us
    # COMMENT THESE OUT FOR FULL DAT SET
    df_read_sorted = df
    df_finished_sorted = df_finished
    
    # For each row, find the most recently finished read on the same device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_with_history = df_read_sorted.join_asof(
            df_finished_sorted,
            on="ts_uptime_us",
            by="device",
            strategy="backward",
        )
    
    df_drop = process_block_io_flags(df_with_history)
    
    
    df_reads = df_drop.filter(pl.col('Read'))
    
    df_reads = df_reads.drop([
        'collection_id', 'block_io_latency_us', 'block_latency_us',
        'measured_latency_us', 'label85', 'label95', 'finish_ts_uptime_us',
        'Idle', 'NoMerge', 'queue_length_segment_ios', 'Read', 'Write', 'segments',
        'cpu', 'sector', 'ts_uptime_us'
    ])
    
    
    median_latency = df_reads["last_latency_block_io_us"].median()
    df_reads = df_reads.with_columns(
            pl.col("last_latency_block_io_us").fill_null(median_latency)
     )
    
    df_reads_indexed = df_reads.with_row_index()

    return df_reads_indexed
    

scaler = StandardScaler()


#model = SGDClassifier(
#    loss="modified_huber",
#    penalty="elasticnet",
#    alpha=1e-6,
#    l1_ratio=0.10,
#    learning_rate="optimal",
#    average=True,
#    random_state=42
#)

model = MLPClassifier(hidden_layer_sizes=(9,8), activation='relu', solver='adam', max_iter=1000, random_state=42)


df_read = handleDf(pl.read_parquet("rainsong_labeled/0.parquet"))
    
# Split the data into training and validation sets (80-20 split)
train_df = df_read.sample(fraction=0.8, with_replacement=False, seed=42)

# Drop the index column from the split dataframes
train_df = train_df.drop('index')
# Separate features and target
train_target_np = train_df['label90'].to_numpy()

train_features = train_df.drop('label90')

train_features_np = train_features.to_numpy()
scaler.partial_fit(train_features_np)
train_features_np = scaler.transform(train_features_np)

print("NaNs:", np.isnan(train_features_np).sum())
print("Infs:", np.isinf(train_features_np).sum())

print("Max:", np.nanmax(train_features_np))
print("Min:", np.nanmin(train_features_np))

model.partial_fit(train_features_np, train_target_np, classes=np.array([0,1]))


for i in range(1,100):
    
    df_read = handleDf(pl.read_parquet(f"rainsong_labeled/{i}.parquet"))
    
    # Split the data into training and validation sets (80-20 split)
    
    # Drop the index column from the split dataframes
    train_df = df_read.drop('index')
    # Separate features and target
    train_target_np = train_df['label90'].to_numpy()

    train_features = train_df.drop('label90')

    train_features_np = train_features.to_numpy()
    # Normalize our data (helps distance-based and margin-based models especially)
    scaler.partial_fit(train_features_np)
    train_features_np = scaler.transform(train_features_np)

    evaluate_model(f"Testing Iteration {i-1}", model, train_features_np, train_target_np)
    print("Partial Fitting to Data", i)
    model.partial_fit(train_features_np, train_target_np)


results = []

#Valdiation is Parquet 100

df_read = handleDf(pl.read_parquet("rainsong_labeled/99.parquet"))
    
# Split the data into training and validation sets (80-20 split)

# Drop the index column from the split dataframes
val_df = df_read.drop('index')
# Separate features and target
val_target_np = val_df['label90'].to_numpy()

val_features = val_df.drop('label90')

val_features_np = val_features.to_numpy()

val_features_np = scaler.transform(val_features_np)


#model.fit(train_features_np, train_target_np)
results.append(evaluate_model("SGD", model, val_features_np, val_target_np))


print(f"{'Model':<28}{'Accuracy':<12}{'F1':<12}{'Speed (us)':<12}")
for r in sorted(results, key=lambda r: r['f1'], reverse=True):
    print(f"{r['name']:<28}{r['accuracy']:<12.4f}{r['f1']:<12.4f}{r['time_ns']/1000:<12.2f}")
