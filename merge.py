import pandas as pd
import os
import glob

# Function to extract information from the folder path
def extract_info(folder_path):
    parts = folder_path.split('\\')
    sensor = parts[1]
    participant = parts[2]
    language_tone = parts[3].split('_')
    language = language_tone[0]
    tone = language_tone[1]
    script = language_tone[2]
    return sensor, language, tone, participant, script

# List to hold dataframes
df_list = []

# Path where the Autocorrelation.csv files are located
path = 'data\\autocorrelations'

# Loop through all the Autocorrelation.csv files
for folder_path in glob.glob(path + '\\*\\*\\Raw data.csv'):
    # Extract information from the folder path
    sensor, language, tone, participant, script = extract_info(folder_path)
    
    # Read the CSV file
    df = pd.read_csv(folder_path)
    
    # Add new columns
    df['sensor'] = sensor
    df['language'] = language
    df['tone'] = tone
    df['participant'] = participant
    df['script'] = script
    
    # Append the dataframe to the list
    df_list.append(df)

# Merge all dataframes
merged_df = pd.concat(df_list, ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('Merged_RawData.csv', index=False)
