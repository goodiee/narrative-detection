import pandas as pd

df = pd.read_csv('M:/VDU 2024-2025/thesis-project/scripts/wit_v1.train.all-00000-of-00010.tsv', sep='\t')

df.to_csv('dataset_with_labels.csv', index=False)
