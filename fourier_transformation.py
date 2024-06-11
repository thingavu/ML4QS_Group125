from Chapter4.FrequencyAbstraction import FourierTransformation
import pandas as pd


ft = FourierTransformation()

data = pd.read_csv('./data_preprocessed/preprocessed_merged_1.csv', index_col=0)
print(data.head())

transformed_data = ft.abstract_frequency(data, columns=['sound_intensity_mean_kalman', 'amplitude_mean_kalman',\
                                                         'pitch_mean_kalman'], window_size=10, sampling_rate=1)
transformed_data.to_csv('./data_w_features/data_w_fourier.csv', index=False)