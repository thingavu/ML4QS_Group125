from TemporalAbstraction import NumericalAbstraction
import pandas as pd

# Load the data
data = pd.read_csv('./data_w_features/data_w_all_features_on_0.5_secs_with_categorical.csv')

# Target columns
# columns = data.columns - ['time_1','language','tone','participant','script']
target_columns = ['amplitude_mean_kalman', 'sound_intensity_mean_kalman', 'pitch_mean_kalman']

# Create an instance of the TemporalAbstraction class
temporal_abstraction = NumericalAbstraction()

# Abstract the numerical data
window_size = 10
for abstration_function in ['mean', 'max', 'min', 'median', 'std', 'slope']:
    data = temporal_abstraction.abstract_numerical(data, target_columns, window_size, abstration_function)

# Save the data
data.to_csv(f'data_temporal_abstraction/data_w_all_features_and_temporal_abstraction_win{window_size}.csv', index=False)