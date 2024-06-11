import pandas as pd
import numpy as np

data = pd.read_csv('./data_w_features/data_w_all_features.csv')

# Create a grouping key that changes every 10 rows
data['group'] = np.arange(len(data)) // 10

# Define the columns for aggregation
max_cols = ["amplitude_max", "pitch_max", "sound_intensity_max"]
min_cols = ["amplitude_min", "pitch_min", "sound_intensity_min"]
mean_cols = ["amplitude_mean_kalman", "pitch_mean_kalman", "sound_intensity_mean_kalman"]
freq_cols = [c for c in data.columns if '_freq' in c]
lag_cols = [c for c in data.columns if '_lag' in c]

# Define aggregation functions for each column
aggregations = {}
for col in max_cols:
    aggregations[col] = 'max'
for col in min_cols:
    aggregations[col] = 'min'
for col in mean_cols:
    aggregations[col] = ['mean', 'median', 'std']
for col in freq_cols:
    aggregations[col] = 'mean'
for col in lag_cols:
    aggregations[col] = 'mean'

# Include non-aggregated columns by selecting the first occurrence in each group, except for time_1 which we sum
aggregations['time_1'] = 'last'
non_agg_cols = ['language', 'tone', 'participant', 'script']
for col in non_agg_cols:
    aggregations[col] = 'first'

# Aggregate the data
aggregated_data = data.groupby('group').agg(aggregations).reset_index(drop=True)

# Flatten the multi-level columns
aggregated_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_data.columns.values]

# Save the aggregated data to a CSV file
aggregated_data.to_csv("./data_w_features/data_w_all_features_on_10_secs.csv", index=False)