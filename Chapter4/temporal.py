from TemporalAbstraction import NumericalAbstraction
import pandas as pd

# Load the data
data = pd.read_csv('./data_w_features/data_w_temporal_patterns.csv')

# Target columns
# columns = data.columns - ['time_1','language','tone','participant','script']
target_columns = ['amplitude_mean_kalman', 'sound_intensity_mean_kalman', 'pitch_mean_kalman']

# Create an instance of the TemporalAbstraction class
temporal_abstraction = NumericalAbstraction()

# Abstract the numerical data
window_size = 10
for abstration_function in ['mean', 'max', 'min', 'median', 'std', 'slope']:
    data = temporal_abstraction.abstract_numerical(data, target_columns, window_size, abstration_function)

    # Group the data by language, tone, participant, and script
    grouped_data = data.groupby(['language', 'tone', 'participant', 'script'])

# Define a function to process each group
def process_group(group):
    # Remove the first 10 rows of the group
    group = group.iloc[10:]
    return group

# Apply the process_group function to each group and reset the index
processed_data = grouped_data.apply(process_group).reset_index(drop=True)

# Merge the processed data back into the original data
data = pd.concat([data[['language', 'tone', 'participant', 'script']], processed_data], axis=1)

# Save the data
data.to_csv(f'data_w_features/data_w_all_features_final_win{window_size}.csv', index=False)