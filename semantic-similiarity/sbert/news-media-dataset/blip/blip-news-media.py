import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load your dataset
df = pd.read_csv("M:/VDU 2024-2025/thesis-project/models/blip/newsmediabias_plus/dataset_with_captions_first_20000_blip.csv")

# Keep only the required columns
df = df[["unique_id", "image_description", "generated_caption"]].dropna()

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight and fast

# Compute similarity for each row
similarities = []
for _, row in df.iterrows():
    desc = row["image_description"]
    caption = row["generated_caption"]

    # Encode both texts
    embeddings = model.encode([desc, caption], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()

    similarities.append({
        "unique_id": row["unique_id"],
        "image_description": desc,
        "generated_caption": caption,
        "semantic_similarity": similarity_score
    })

# Save results
similarity_df = pd.DataFrame(similarities)
similarity_df.to_csv("sbert_caption_similarity_blip_news-media.csv", index=False)
