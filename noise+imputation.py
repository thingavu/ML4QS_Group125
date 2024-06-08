from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
from Chapter3.DataTransformation import HighPassFilter
import pandas as pd

# Load the data
data = pd.read_csv('data_aggregated\pitch_time_0.1.csv', index_col=0)

# Target columns
target_columns = ['pitch_mean']
# Print the NaN values in target columns
print("NaN values in origin columns:", data[target_columns].isnull().sum())

# Create an instance of the ImputationMissingValues class
imputer = ImputationMissingValues()

# Create an instance of the HighPassFilter class
high_pass = HighPassFilter()

# Create an instance of the KalmanFilters class
kalman = KalmanFilters()

# Define a function to apply the operations on a group
def process_group(group):
    # Call the impute_interpolate function
    for col in target_columns:
        group = imputer.impute_interpolate(group, col)

    # Call the high_pass_filter function
    for col in target_columns:
        group = high_pass.high_pass_filter(group, col, 1, 0.1)

    # Call the apply_kalman_filter function
    for col in target_columns:
        group = kalman.apply_kalman_filter(group, col)
    
    return group

# Apply the operations on each subgroup
data = data.groupby(['language', 'tone', 'participant', 'script']).apply(process_group).reset_index(drop=True)
print("NaN valuesafter imputation:", data[target_columns].isnull().sum())
