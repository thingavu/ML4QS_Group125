import pandas as pd
from tqdm import tqdm

def main(n_aggregate=25):
    # Load the merged CSV file
    df = pd.read_csv('./data/Merged_RawData.csv')
    # df = df[(df['participant'] != 'nga') & (df['participant'] != 'duru')]
    # print(df)

    # Set the number of rows to aggregate
    column_to_aggregate = 'Recording (a.u.)'

    # Function to calculate the aggregation and statistics for 'Autocorrelation (a.u.)'
    def aggregate_rows(n, group, group_name, formatted_x):
        # List to store aggregated data
        agg_list = []
        
        # Calculate the number of chunks
        num_chunks = len(group) // n
        if len(group) % n != 0:
            num_chunks += 1  # Account for the last group with less than n rows
        
        # Iterate over the chunks
        for i in range(num_chunks):
            # Select n rows for the current chunk
            chunk = group.iloc[i*n : min((i+1)*n, len(group))]
            
            # Calculate the percentage of missing values for the specified column
            missing = chunk[column_to_aggregate].isnull().mean() * 100
            
            # Calculate mean, std, min, and max for the specified column
            mean_val = chunk[column_to_aggregate].mean()
            median_val = chunk[column_to_aggregate].median()
            std_val = chunk[column_to_aggregate].std(ddof=0)
            min_val = chunk[column_to_aggregate].min()
            max_val = chunk[column_to_aggregate].max()

            # Calculate the time stamp
            time = "{:.1f}".format((i+1)*float(formatted_x))
            
            # Create a dictionary of the calculated values
            agg_dict = {
                'sensor': group_name[0],
                'language': group_name[1],
                'tone': group_name[2],
                'participant': group_name[3],
                'script': group_name[4],
                f'time_{formatted_x}': time,
                'autocorrelation_mean': mean_val,
                'autocorrelation_median': median_val,
                'autocorrelation_std': std_val,
                'autocorrelation_min': min_val,
                'autocorrelation_max': max_val,
                'autocorrelation_missing': missing
            }
            
            # Append the dictionary to the list
            agg_list.append(agg_dict)
        
        # Convert the list of dictionaries to a DataFrame
        return pd.DataFrame(agg_list)

    # Calculate the time stamp
    x = df["Time (ms)"][n_aggregate]-df["Time (ms)"][1]
    formatted_x = "{:.1f}".format(x)

    # Group by the specified columns and apply the aggregation function
    aggregated_data = []
    for group_name, group in df.groupby(['sensor','language', 'tone', 'participant', 'script']):
        aggregated_group = aggregate_rows(n_aggregate, group, group_name, formatted_x)
        aggregated_data.append(aggregated_group)

    # Concatenate all the group DataFrames
    aggregated_df = pd.concat(aggregated_data, ignore_index=True)

    # Save the aggregated dataframe to a new CSV file
    aggregated_df.to_csv(f'./data_aggregated/rawdata_time_{formatted_x}.csv', index=False)

if __name__ == '__main__':
    for n in [6, 25, 49, 97, 241]:
        main(n)
