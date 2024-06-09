import matplotlib.pyplot as plt
import pandas as pd
import glob


def line_plots_combined(data, measure='pitch_mean', min_time=None, max_time=None, ):
    sensor = data['sensor'].iloc[0]
    data["language_adj"] = data["language"].apply(lambda x: "non-native" if x == "en" else "native")
    languages = data['language_adj'].unique()
    tones = data['tone'].unique()

    time_col = [c for c in data.columns if 'time' in c][0]
    delta_t = float(time_col.split('_')[-1])

    if min_time is None and max_time is None:
        min_time = data[time_col].min()
        max_time = data[time_col].max()
    
    data = data[(data[time_col] > min_time) & (data[time_col] < max_time)]

    fig, axs = plt.subplots(4, 1, figsize=(18, 6), sharex=True, sharey=True)
    axs = axs.flatten()
    idx = 0

    for language in languages:
        for tone in tones:
            ax = axs[idx]
            filtered_data = data[(data['language_adj'] == language) & (data['tone'] == tone)]
            participants = filtered_data['participant'].unique()

            for participant in participants:
                participant_data = filtered_data[filtered_data['participant'] == participant]
                scripts = participant_data['script'].unique()

                for script in scripts:
                    script_data = participant_data[participant_data['script'] == script]
                    ax.plot(script_data[time_col], script_data[measure], label=f'{participant}-{script}')

            ax.set_xlim(min_time, max_time)
            # if language == "en":
            #     language_label = "Non-mother-tongue"
            # else:
            #     language_label = "Mother-tongue"
            
            tone_label = tone
            if tone == "bus":
                tone_label = "business"
            
            sensor_label = sensor
            if 'intensity' in sensor:
                sensor_label = 'intensity'

            ax.set_ylabel(f'{sensor_label.capitalize()}', fontsize=16)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=9, fontsize=10)
            ax.grid(True)
            ax.set_title(f'{language.capitalize()} - {tone_label.capitalize()}', fontsize=18)
            idx += 1

    # plt.xlabel(f'Time (Δt={delta_t}s)', fontsize=14)
    plt.tight_layout()
    file_name = f'./lineplots/{sensor}_delta{delta_t}.png'
    plt.savefig(file_name)
    plt.close()

# Call the function with the aggregated data
for p in glob.glob("/Users/nag/study/vu_ms_ai/ml4qs/ML4QS_Group125/data_aggregated/*_1.csv"): 
    if 'autocor' not in p:
        data = pd.read_csv(p)
        measure = [c for c in data.columns if 'mean' in c][0]
        line_plots_combined(data, measure=measure, min_time=100, max_time=120)