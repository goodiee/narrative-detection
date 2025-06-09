import pandas as pd

# Load CSV file
csv_filename = "M:/VDU 2024-2025/thesis-project/data/newsmediabias-plus/labels/dataset_with_labels.csv"
df = pd.read_csv(csv_filename)

# Count non-empty titles
num_titles = df["unique_id"].notna().sum()

print(f"Total titles in CSV file: {num_titles}")
