import matplotlib
matplotlib.use('TkAgg')  # Set the correct Matplotlib backend

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr  # For Pearson correlation

# Load your dataset
file_path = "M:/VDU 2024-2025/thesis-project/models/vit/news-media-dataset/dataset_with_generated_captions_vit_20000.csv"  # Update the path to your CSV file
df = pd.read_csv(file_path)

# Check the first few rows to ensure the structure
print(df.head())

# Initialize Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Get embeddings for each row (image_description and generated_caption)
image_desc_embeddings = model.encode(df['image_description'].dropna().values.tolist(), convert_to_tensor=True)
generated_caption_embeddings = model.encode(df['generated_caption'].dropna().values.tolist(), convert_to_tensor=True)

# Convert embeddings to CPU for similarity calculation
image_desc_embeddings_cpu = image_desc_embeddings.cpu()
generated_caption_embeddings_cpu = generated_caption_embeddings.cpu()

# Compute cosine similarity between image_description and generated_caption
similarity_scores = cosine_similarity(image_desc_embeddings_cpu, generated_caption_embeddings_cpu)

# Add similarity scores to DataFrame
df['semantic_similarity'] = similarity_scores.diagonal()

# Apply threshold for classification (e.g., 0.5)
df['similarity_class'] = (df['semantic_similarity'] >= 0.5).astype(int)

# Save the updated DataFrame
import os

output_path = "/semantic-similiarity/sbert/news-media-dataset/vit/dataset_with_similarity_vit.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['semantic_similarity']], df['similarity_class'], test_size=0.2, random_state=42)

# Train classifier (e.g., Logistic Regression)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Pearson correlation
pearson_corr, _ = pearsonr(df['semantic_similarity'], df['similarity_class'])

# ROC-AUC
roc_auc = roc_auc_score(df['similarity_class'], df['semantic_similarity'])

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Similar", "Similar"],
            yticklabels=["Not Similar", "Similar"],
            annot_kws={'size': 14, 'color': 'black'})  # Аннотации без жирного шрифта
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.show()
