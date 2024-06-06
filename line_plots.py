import matplotlib.pyplot as plt
import pandas as pd

def line_plots(data, measure='intensity_mean'):
    sensor = data['sensor'].iloc[0]
    languages = data['language'].unique()
    tones = data['tone'].unique()

    time_col = [c for c in data.columns if 'time' in c][0]
    delta_t = float(time_col.split('_')[-1])

    def plot_and_save_intensity_float(data, language, tone):
        plt.figure(figsize=(18, 3))
        filtered_data = data[(data['language'] == language) & (data['tone'] == tone)]
        participants = filtered_data['participant'].unique()
        
        for participant in participants:
            participant_data = filtered_data[filtered_data['participant'] == participant]
            scripts = participant_data['script'].unique()
            
            for script in scripts:
                script_data = participant_data[participant_data['script'] == script]
                plt.plot(script_data[time_col], script_data[measure], label=f'{participant}-{script}')
        
        # plt.xlabel(f'Time (Δt={delta_t})', fontsize=14)
        plt.ylabel(f'{sensor.capitalize()}', fontsize=14)
        # plt.title(f'{sensor.capitalize()} over Time (Δt={delta_t}) for Language: {language}, Tone: {tone}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=8, fontsize=12)
        plt.grid(True)
        
        file_name = f'{sensor}_mean_{language}_{tone}.png'
        plt.savefig(file_name)
        plt.close()

    # Save plots for each combination of language and tone with float time values
    for language in languages:
        for tone in tones:
            plot_and_save_intensity_float(data, language, tone)

# Call the function with the aggregated data
data = pd.read_csv('data_aggregated/pitch_time_0.1.csv')
line_plots(data, measure='pitch_mean')