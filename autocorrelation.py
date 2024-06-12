from statsmodels.tsa.stattools import acf
import pandas as pd
import numpy as np

data = pd.read_csv('./data_w_features/data_w_fourier.csv')

for measurement_col in ['amplitude_mean_kalman', 'sound_intensity_mean_kalman']:
    meas_name = 'amplitude'
    if measurement_col == 'sound_intensity_mean_kalman':
        meas_name = 'sound_intensity'

    measurement = data[measurement_col]

    # Calculate autocorrelation values for lags from 0 to 10
    autocorr_values = acf(measurement.dropna(), nlags=10)

    # Add autocorrelation values to the dataframe
    for lag in range(11):
        data[f'{meas_name}_autocorrelation_lag_{lag}'] = np.nan
        data.loc[data[measurement_col].notna(), f'{meas_name}_autocorrelation_lag_{lag}'] = autocorr_values[lag]

data.to_csv('./data_w_features/data_w_all_features.csv', index=False)
