# 🧠 Narrative Manipulation Detection via Image and Text Comparison

![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research--project-yellow)

This project detects **narrative manipulation** by comparing **images and text** from propaganda content, with a focus on Telegram channels and media datasets. The goal is to identify misleading narratives where visuals contradict or manipulate the associated captions.


## 📁 Project Structure

- `data/` – Contains dataset-related files and placeholders  
- `models/` – Image-to-text models like BLIP, ViT-GPT2  
  - `models_to_test.ipynb` – Notebook for testing models  
- `results/` – Model output results  
- `scraper/`  
  - `scraper.py` – Scrapes Telegram for image/text content  
- `scripts/`  
  - `converter_paraq_csv.py` – Converts custom data formats into CSV  
  - `counter.py` – Basic statistics (e.g., counts)  
  - `dataset_reconstructure.py` – Reorganizes dataset structure  
  - `image_downloader.py` – Downloads images by URL  
- `semantic-similiarity/`  
  - `sbert/`  
    - `news-media-dataset/` – Embeddings for NewsMediaBias-Plus  
    - `telegram dataset/` – Embeddings for Telegram dataset  
    - `evaluation.py` – Evaluation script for similarity  
    - `sbert-generation-similiarity.py` – Generates SBERT embeddings  
- `translation/`  
  - `translator.py` – For translating non-English text  
- `visualisatons/`  
  - `benchmark-box-plots.py` – Visualization of benchmark results  
  - `data_analysis.ipynb` – Data analysis and insights  
- `requirements.txt` – Python dependencies  
- `.gitignore` – Git ignored files


## 🔧 Installation

To set up the environment:

```bash
git clone https://github.com/yourusername/narrative-detection.git
cd narrative-manipulation-detection
pip install -r requirements.txt
```

## 📊 Datasets

### 📰 NewsMediaBias-Plus  
A benchmark dataset containing media headlines and image pairs with bias annotations.

### 📡 Telegram Propaganda Dataset  
Custom dataset of image-caption pairs collected from Russian Telegram channels known for disinformation.


## 📌 Features

- 🧠 Compare image-generated captions to text using models like **BLIP**, **ViT-GPT2**, and **CLIP**
- 🔎 Use **SBERT** for semantic similarity scoring
- 🌍 Translate multilingual content for fair evaluation
- 📉 Visualize model performance with box plots and metrics
- 🧰 Modular, extensible architecture for plug-and-play with models or datasets

---

## 🛠️ Technologies

- Python 3.8+
- HuggingFace Transformers
- SentenceTransformers (SBERT)
- OpenAI APIs (optional)
- BLIP, ViT, CLIP
- Matplotlib, Seaborn

---

## ✍️ Author

**Maksym Bondar**  
Researcher in multimodal disinformation detection and narrative manipulation using vision-language models.

---

## 📜 License

Licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

---

## 🤝 Contributing

Feel free to contribute! Open an issue, fork the repo, or submit a pull request.
