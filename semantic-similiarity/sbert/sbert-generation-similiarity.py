import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("path/to/your/input.csv")
df = df[["unique_id", "image_description", "generated_caption"]].dropna()

model = SentenceTransformer("all-MiniLM-L6-v2")

similarities = []
for _, row in df.iterrows():
    desc = row["image_description"]
    caption = row["generated_caption"]
    embeddings = model.encode([desc, caption], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    similarities.append({
        "unique_id": row["unique_id"],
        "image_description": desc,
        "generated_caption": caption,
        "semantic_similarity": similarity_score
    })

similarity_df = pd.DataFrame(similarities)
similarity_df.to_csv("path/to/your/similiarity-score.csv", index=False)
