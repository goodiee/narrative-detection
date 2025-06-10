import argparse
import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import numpy as np

max_images = 4000

parser = argparse.ArgumentParser(description='WIT dataset image downloader')
parser.add_argument('filepath', type=str, help='Path to the dataset file (TSV format)')
args = parser.parse_args()

df = pd.read_csv(args.filepath, sep="\t", dtype=str)
df = df.replace(np.nan, '', regex=True)

if not os.path.exists("images"):
    os.makedirs("images")

pbar = tqdm(total=max_images)
downloaded = 0

for _, row in df.iterrows():
    if downloaded >= max_images:
        break

    image_url = row["image_url"]
    mime_type = row["mime_type"]

    if image_url and "image" in mime_type:
        try:
            image_name = f"{downloaded}.jpg"
            image_path = os.path.join("images", image_name)
            urllib.request.urlretrieve(image_url, image_path)
            downloaded += 1
        except Exception:
            continue

    pbar.update(1)

pbar.close()
print(f"Total images downloaded: {downloaded}")
