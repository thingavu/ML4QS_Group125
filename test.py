import glob
import pandas as pd

for path in glob.glob("data_aggregated/ampli*.csv"):
    data = pd.read_csv(path)
    data['participant'] = data['participant'].replace({'jack': 'subject1', 'yaoyi': 'subject2'})
    data.to_csv(path, index=False)