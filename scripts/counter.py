import pandas as pd

csv_filename = "data/newsmediabias-plus/labels/dataset_with_labels.csv"
df = pd.read_csv(csv_filename)

num_titles = df["unique_id"].notna().sum()

print(f"Total titles in CSV file: {num_titles}")
