import argparse
import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import numpy as np

# Settings
max_images = 4000  # Limit on the number of images to download

# Parse command-line argument (train, validate, test)
parser = argparse.ArgumentParser(description='WIT dataset image downloader')
parser.add_argument('filepath', type=str, help='Path to the dataset file (TSV format)')
args = parser.parse_args()

# Load dataset
print(f"Loading data from {args.filepath}...")
df = pd.read_csv(args.filepath, sep="\t", dtype=str)
df = df.replace(np.nan, '', regex=True)

# Print the first few rows of the dataset to ensure it's loaded correctly
print("First few rows of the dataset:")
print(df.head())

# Create directory for images
if not os.path.exists("images"):
    os.makedirs("images")

pbar = tqdm(total=max_images)
downloaded = 0

# Start downloading images
print("Starting image download...")

for index, row in df.iterrows():
    if downloaded >= max_images:
        break  # Stop when 4000 images are downloaded

    image_url = row["image_url"]
    mime_type = row["mime_type"]

    # Print URLs and mime types for debugging
    print(f"Processing index {index}...")
    print(f"Image URL: {image_url}, Mime type: {mime_type}")

    # Check if the row contains an image URL and mime_type
    if image_url and "image" in mime_type:
        try:
            image_name = f"{downloaded}.jpg"
            image_path = os.path.join("images", image_name)
            print(f"[{downloaded + 1}/{max_images}] Downloading {image_url} -> {image_path}")

            # Download the image
            urllib.request.urlretrieve(image_url, image_path)
            downloaded += 1
        except Exception as e:
            print(f"Error downloading {image_url}: {str(e)}")

    pbar.update(1)

pbar.close()
print(f"Download complete. Total images downloaded: {downloaded}")
