import matplotlib.pyplot as plt
import pandas as pd


def line_plots_combined(data, measure='pitch_mean'):
    sensor = data['sensor'].iloc[0]
    languages = data['language'].unique()
    tones = data['tone'].unique()

    time_col = [c for c in data.columns if 'time' in c][0]
    delta_t = float(time_col.split('_')[-1])

    # Determine the common x-axis range
    min_time = data[time_col].min()
    max_time = data[time_col].max()

    fig, axs = plt.subplots(4, 1, figsize=(18, 6), sharex=True, sharey=True)
    axs = axs.flatten()
    idx = 0

    for language in languages:
        for tone in tones:
            ax = axs[idx]
            filtered_data = data[(data['language'] == language) & (data['tone'] == tone)]
            participants = filtered_data['participant'].unique()

            for participant in participants:
                participant_data = filtered_data[filtered_data['participant'] == participant]
                scripts = participant_data['script'].unique()

                for script in scripts:
                    script_data = participant_data[participant_data['script'] == script]
                    ax.plot(script_data[time_col], script_data[measure], label=f'{participant}-{script}')

            ax.set_xlim(min_time, max_time)
            if language == "en":
                language_label = "Non-mother-tongue"
            else:
                language_label = "Mother-tongue"
            
            tone_label = tone
            if tone == "bus":
                tone_label = "business"
            ax.set_ylabel(f'{sensor.capitalize()}', fontsize=14)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=8, fontsize=10)
            ax.grid(True)
            ax.set_title(f'{language_label} - {tone_label.capitalize()}', fontsize=14)
            idx += 1

    plt.xlabel(f'Time (Δt={delta_t})', fontsize=14)
    plt.tight_layout()
    file_name = f'./lineplots/{sensor}_delta{delta_t}.png'
    plt.savefig(file_name)
    plt.close()

# def line_plots(data, measure='intensity_mean'):
#     sensor = data['sensor'].iloc[0]
#     languages = data['language'].unique()
#     tones = data['tone'].unique()

#     time_col = [c for c in data.columns if 'time' in c][0]
#     delta_t = float(time_col.split('_')[-1])
    
#     def plot_and_save_intensity_float(data, language, tone):
#         plt.figure(figsize=(18, 3))
#         filtered_data = data[(data['language'] == language) & (data['tone'] == tone)]
#         participants = filtered_data['participant'].unique()

#         min_time = data[time_col].min()
#         max_time = data[time_col].max()
#         for participant in participants:
#             participant_data = filtered_data[filtered_data['participant'] == participant]
#             scripts = participant_data['script'].unique()
            
#             for script in scripts:
#                 script_data = participant_data[participant_data['script'] == script]
#                 plt.plot(script_data[time_col], script_data[measure], label=f'{participant}-{script}')
        
#         plt.xlim(min_time, max_time)
#         # plt.xlabel(f'Time (Δt={delta_t})', fontsize=14)
#         plt.ylabel(f'{sensor.capitalize()}', fontsize=14)
#         # plt.title(f'{sensor.capitalize()} over Time (Δt={delta_t}) for Language: {language}, Tone: {tone}')
#         plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=8, fontsize=12)
#         plt.grid(True)
        
#         file_name = f'./lineplots/{sensor}_delta{delta_t}_{language}_{tone}.png'
#         plt.savefig(file_name)
#         plt.close()

#     # Save plots for each combination of language and tone with float time values
#     for language in languages:
#         for tone in tones:
#             plot_and_save_intensity_float(data, language, tone)

# Call the function with the aggregated data
data = pd.read_csv('data_aggregated/pitch_time_0.1.csv')
line_plots_combined(data, measure='pitch_mean')