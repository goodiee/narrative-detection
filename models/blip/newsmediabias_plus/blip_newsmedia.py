import os
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, UnidentifiedImageError
import torch
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

# Paths
csv_path = "M:/VDU 2024-2025/thesis-project/data/newsmediabias-plus/labels/dataset_with_labels.csv"
image_folder = "M:/VDU 2024-2025/thesis-project/data/newsmediabias-plus/images"

# Load dataset
df = pd.read_csv(csv_path)
# Filter valid images
def is_valid_image(unique_id):
    image_filename = f"{unique_id}.jpg"
    image_path = os.path.join(image_folder, image_filename)

    if not os.path.exists(image_path):
        return False
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

print("🔎 Checking valid images...")
tqdm.pandas(desc="🔎 Checking images")
df["is_valid"] = df["unique_id"].progress_apply(is_valid_image)

# Filter dataset to only valid images
valid_df = df[df["is_valid"]].copy()
print(f"✅ Found {len(valid_df)} valid images out of {len(df)} entries.")

# Caption generation function
def generate_caption(unique_id):
    image_filename = f"{unique_id}.jpg"
    image_path = os.path.join(image_folder, image_filename)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Apply captioning with progress bar
tqdm.pandas(desc="🖼 Generating captions")
valid_df["generated_caption"] = valid_df["unique_id"].progress_apply(generate_caption)

# Save
valid_df.drop(columns=["is_valid"], inplace=True)
valid_df.to_csv("dataset_with_captions_first_20000_blip.csv", index=False)
print("✅ Captions generation completed and CSV saved.")

