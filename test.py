import pandas as pd
import glob

for delta in [0.1, 0.5, 1, 2, 5]:
    paths = glob.glob(f"./data_aggregated/*_{delta}.csv")
    print(paths)
    data = pd.read_csv(paths[0])
    for path in paths[1:]:
        data = pd.merge(data, pd.read_csv(path), how='outer', on=[f'time_{delta}', 'language', 'tone', 'participant', 'script', 'sensor'])
    data.to_csv(f"./data_aggregated/merged_{delta}.csv", index=False)