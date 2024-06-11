import numpy as np
import pandas as pd

# load the data
path = 'data_w_features\data_w_fourier.csv'
data = pd.read_csv(path)

# Create a grouping key that changes every 10 rows
grouping_key = np.arange(len(data)) // 10
max_cols = ["amplitude_max", "pitch_max", "sound_intensity_max"]
min_cols = ["amplitude_min", "pitch_min", "sound_intensity_min"]
mean_cols = ["amplitude_mean_kalman", "pitch_mean_kalman", "sound_intensity_mean_kalman"]
freq_cols = [c for c in data.columns if '_freq' in c ]
lag_cols = [c for c in data.columns if '_lag' in c ]

aggregations = dict()
for col in max_cols:
    aggregations[col] = "max"
for col in min_cols:
    aggregations[col] = "min"
for col in mean_cols:
    aggregations[col] = "mean"
    aggregations[col] = "median"
    aggregations[col] = "std"
for col in freq_cols:
    aggregations[col] = "mean"
for col in lag_cols:
    aggregations[col] = "mean"


tmp_data = data.groupby(grouping_key).agg(aggregations).reset_index()
tmp_data.columns
print(tmp_data.columns)
