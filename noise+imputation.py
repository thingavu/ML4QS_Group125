from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
from Chapter3.DataTransformation import LowPassFilter
import pandas as pd

# Load the data
path = './data_merged/merged_0.1.csv'
data = pd.read_csv(path)

# Target columns
target_columns = ['amplitude_mean', 'pitch_mean', 'sound_intensity_mean']
# Print the NaN values in target columns
for col in target_columns:
    print("original-",col, data[col].isnull().sum())

# Create an instance of the ImputationMissingValues class
imputer = ImputationMissingValues()

# Create an instance of the LowPassFilter class
low_pass = LowPassFilter()

# Create an instance of the KalmanFilters class
kalman = KalmanFilters()

# Define a function to apply the operations on a group
def process_group(group):
    # Call the impute_interpolate function
    for col in target_columns:
        group = imputer.impute_interpolate(group, col)

    # Call the high_pass_filter function
    for col in target_columns:
        group = low_pass.low_pass_filter(group, col, 100, 10)

    # Call the apply_kalman_filter function
    for col in target_columns:
        group = kalman.apply_kalman_filter(group, col)

    # Call the impute_interpolate function again since the Kalman filter might have introduced NaN values
    for col in target_columns:
        group = imputer.impute_interpolate(group, col)
    
    return group

# Apply the operations on each subgroup
data = data.groupby(['language', 'tone', 'participant', 'script']).apply(process_group).reset_index(drop=True)

# Print the NaN values in target columns
for col in target_columns:
    print("final-",col, data[col].isnull().sum())

# Save the data
file_name = "preprocessed_" + path.split('/')[-1] #.split('.')[0]
data.to_csv("./data_preprocessed/" + file_name)
