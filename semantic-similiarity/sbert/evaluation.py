import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "path/to/your/dataset.csv"
df = pd.read_csv(file_path)

print(df.head())

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

image_desc_embeddings = model.encode(df['image_description'].dropna().values.tolist(), convert_to_tensor=True)
generated_caption_embeddings = model.encode(df['generated_caption'].dropna().values.tolist(), convert_to_tensor=True)

image_desc_embeddings_cpu = image_desc_embeddings.cpu()
generated_caption_embeddings_cpu = generated_caption_embeddings.cpu()

similarity_scores = cosine_similarity(image_desc_embeddings_cpu, generated_caption_embeddings_cpu)

df['semantic_similarity'] = similarity_scores.diagonal()
df['similarity_class'] = (df['semantic_similarity'] >= 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df[['semantic_similarity']], df['similarity_class'], test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

pearson_corr, _ = pearsonr(df['semantic_similarity'], df['similarity_class'])
roc_auc = roc_auc_score(df['similarity_class'], df['semantic_similarity'])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Similar", "Similar"],
            yticklabels=["Not Similar", "Similar"],
            annot_kws={'size': 14, 'color': 'black'})
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.show()
