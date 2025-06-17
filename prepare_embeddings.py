import json
import numpy as np
from embedding_utils import get_embedding  # assuming you have this
from tqdm import tqdm

# Load your data from brochures.json, articles.json, about.json
with open("brochures.json", encoding="utf-8") as f:
    brochures = json.load(f)
    for b in brochures:
        b["source"] = "brochure"

with open("articles.json", encoding="utf-8") as f:
    articles = json.load(f)
    for a in articles:
        a["source"] = "article"

with open("about.json", encoding="utf-8") as f:
    abouts = json.load(f)
    for ab in abouts:
        ab["source"] = "about"

all_docs = brochures + articles + abouts

print(f"📄 Loaded {len(all_docs)} documents")

embeddings = []
texts = []
sources = []

print("🔄 Generating embeddings...")
for doc in tqdm(all_docs):
    text = doc.get("text") or doc.get("content") or doc.get("description") or ""
    if not text.strip():
        continue
    embedding = get_embedding(text)
    embeddings.append(embedding)
    texts.append(text)
    sources.append(doc["source"])

embeddings = np.array(embeddings, dtype=np.float32)

np.savez_compressed("embeddings_data.npz", embeddings=embeddings, texts=texts, sources=sources)

print(f"✅ Saved compressed embeddings_data.npz with {len(texts)} items")
