from FrequencyAbstraction import FourierTransformation
from statsmodels.tsa.stattools import acf

import pandas as pd
import numpy as np


# Function to calculate and inspect autocorrelation values
def rolling_autocorrelation(series, window, max_lag):
    autocorr_dict = {f'rolling_autocorr_lag_{lag}': np.full_like(series, fill_value=np.nan, dtype=np.float64) for lag in range(1, max_lag + 1)}
    
    for start in range(len(series) - window + 1):
        end = start + window
        window_series = series[start:end]
        if np.any(np.isnan(window_series)):
            continue
        autocorr_values = acf(window_series, nlags=max_lag)
        for lag in range(1, max_lag + 1):
            autocorr_dict[f'rolling_autocorr_lag_{lag}'][end - 1] = autocorr_values[lag]
    
    return autocorr_dict


data = pd.read_csv('./data_preprocessed/preprocessed_merged_0.5.csv', index_col=0)

trials = data[['language', 'tone', 'participant', 'script']].drop_duplicates()
trials = list(trials.itertuples(index=False, name=None))
print("Number of trials: ", len(trials))

result = pd.DataFrame()
for lang, tone, part, script in trials:
    data_subset = data[(data['language'] == lang) & (data['tone'] == tone) & (data['participant'] == part) & (data['script'] == script)]

    # Apply Fourier transformation to the data
    ft = FourierTransformation()
    transformed_data = ft.abstract_frequency(data_subset, columns=['sound_intensity_mean_kalman', 'amplitude_mean_kalman',\
                                                            'pitch_mean_kalman'], window_size=10, sampling_rate=1)

    # Get autocorrelation values for amplitude and sound intensity
    window_size = 10
    max_lag = 5

    amplitude_autocorr = rolling_autocorrelation(transformed_data['amplitude_mean_kalman'], window_size, max_lag)
    sound_intensity_autocorr = rolling_autocorrelation(transformed_data['sound_intensity_mean_kalman'], window_size, max_lag)

    # Add rolling autocorrelation values to the dataframe
    for lag in range(1, max_lag + 1):
        transformed_data[f'amplitude_autocorr_lag_{lag}'] = amplitude_autocorr[f'rolling_autocorr_lag_{lag}']
        transformed_data[f'sound_intensity_autocorr_lag_{lag}'] = sound_intensity_autocorr[f'rolling_autocorr_lag_{lag}']
    
    result = pd.concat([result, transformed_data])

result.to_csv('./data_w_features_ver2/data_w_fourier_and_autocorr_0.5.csv', index=False)