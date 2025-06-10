import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en", cache_dir="path/to/cache")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en", cache_dir="path/to/cache")

def translate_texts(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    result = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return result

input_file = "path/to/input.csv"
output_file = "path/to/output.csv"

df = pd.read_csv(input_file)

if 'preprocessed_messages' not in df.columns:
    raise ValueError("Missing 'preprocessed_messages' column.")

messages = df['preprocessed_messages'].astype(str).tolist()
translated_messages = []

batch_size = 10
for i in tqdm(range(0, len(messages), batch_size), desc="Translating batches"):
    batch = messages[i:i + batch_size]
    translated_batch = translate_texts(batch)
    translated_messages.extend(translated_batch)

df['message_translated'] = translated_messages
df.to_csv(output_file, index=False, encoding='utf-8')
