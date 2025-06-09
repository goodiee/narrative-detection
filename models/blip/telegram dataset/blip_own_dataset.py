import os
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()  # Set model to semantic-similiarity mode

# Load your dataset
csv_path = "/data/telegram dataset/final_merged_dataset.csv"
df = pd.read_csv(csv_path)

# Base directory where the images are stored
base_image_dir = "/data/telegram dataset/"

# Function to generate caption from image
def generate_caption(relative_path):
    image_path = os.path.join(base_image_dir, relative_path)
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return ""

# Generate captions with progress bar
tqdm.pandas(desc="🖼 Generating captions")
df["generated_caption"] = df["image_path"].progress_apply(generate_caption)

# Save the updated CSV
df.to_csv("dataset_with_captions.csv", index=False)
print("✅ Captions added and CSV saved.")
