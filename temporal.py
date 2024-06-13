from Chapter4.TemporalAbstraction import NumericalAbstraction
import pandas as pd

# Load the data
data = pd.read_csv('./data_w_features/data_w_all_features.csv')

# Target columns
columns = data.columns - ['time_1','language','tone','participant','script']
target_columns = columns - []

# Create an instance of the TemporalAbstraction class
temporal_abstraction = NumericalAbstraction()

# Abstract the numerical data
for abstration_function in ['mean', 'max', 'min', 'median', 'std', 'slope']:
    data = temporal_abstraction.abstract_numerical(data, target_columns, 5, abstration_function)

# Save the data
data.to_csv('data_temporal_abstraction/data_w_all_features_and_temporal_abstraction.csv', index=False)