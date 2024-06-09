import pandas as pd
import glob
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def read_data(path, split="/"):
    data = pd.read_csv(path, sep=',')
    parts = path.split(split)
    trial = parts[-1].replace(".csv", "").split("_")

    data["language"] = trial[0]
    data["tone"] = trial[1]
    data["participant"] = parts[-2]
    data["script"] = trial[2]
        
    supportive_columns = [c for c in data.columns if c not in ["PitchSensor", "DecibelSource"]]
    pitch_data = data[supportive_columns + ["PitchSensor"]]
    pitch_data["sensor"] = "pitch"
    pitch_data = pitch_data.rename(columns={"PitchSensor": "pitch", "relative_time": "time"})

    intensity_data = data[supportive_columns + ["DecibelSource"]]
    intensity_data["sensor"] = "sound_intensity"
    intensity_data = intensity_data.rename(columns={"DecibelSource": "sound_intensity", "relative_time": "time"})
    
    return pitch_data, intensity_data

def aggregation(data, measurement, delta=0.1):
    data['time_in_sec'] = data['time'] / 1000
    data[f'time_{delta}'] = (data['time_in_sec'] // delta).astype(int) * delta

    def calculate_missing_percentage(group):
        total_count = len(group)
        missing_count = group[measurement].isna().sum()
        missing_percentage = (missing_count / total_count) * 100
        sensor = group['sensor'].iloc[0]

        return pd.Series({
            f'{sensor}_max': group[measurement].max(),
            f'{sensor}_min': group[measurement].min(),
            f'{sensor}_mean': group[measurement].mean(),
            f'{sensor}_median': group[measurement].median(),
            f'{sensor}_std': group[measurement].std(),
            f'{sensor}_missing': missing_percentage
        })

    aggregated_data = data.groupby([f'time_{delta}', 'language', 'tone', 'participant', 'script', 'sensor']).apply(calculate_missing_percentage).reset_index()
    return aggregated_data

def main():
    os.makedirs("../data_aggregated", exist_ok=True)

    for delta in tqdm([0.1, 0.5, 1, 2, 5]):
        pitch_merged = pd.DataFrame()
        intensity_merged = pd.DataFrame()        
        for path in glob.glob("data/intensity_pitch/*/*.csv"):
            # if "nga" in path:
            #     continue
            # if "duru" in path:
            #     continue
            pitch_data, intensity = read_data(path)
            pitch_aggregated = aggregation(pitch_data, "pitch", delta=delta)
            intensity_aggregated = aggregation(intensity, "sound_intensity", delta=delta)

            pitch_merged = pd.concat([pitch_merged, pitch_aggregated])
            intensity_merged = pd.concat([intensity_merged, intensity_aggregated])
        pitch_merged.to_csv(f"data_aggregated/pitch_time_{delta}.csv", index=False)
        intensity_merged.to_csv(f"data_aggregated/intensity_time_{delta}.csv", index=False)


if __name__ == '__main__':
    main()