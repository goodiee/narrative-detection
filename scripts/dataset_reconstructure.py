import pandas as pd

df = pd.read_csv("final_main_merged_dataset.csv", header=None)

df = df.iloc[:, :-1]

df.to_csv("final_merged_dataset.csv", index=False, header=False)
