import json
import os
import hashlib
from embedding_utils import get_embedding

def hash_content(doc):
    content = doc.get("title", "") + doc.get("content", "") + doc.get("description", "")
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Load and tag source
print("📥 Loading documents...")
brochures = load_json("brochures.json")
for b in brochures:
    b["source"] = "brochure"

articles = load_json("articles.json")
for a in articles:
    a["source"] = "article"

videos = load_json("videos.json")
for v in videos:
    v["source"] = "video"

all_docs = brochures + articles + videos
print(f"📄 Loaded {len(all_docs)} documents")

# Load previous cache (optional but not used for fail-fast)
if os.path.exists("embeddings_cache.json"):
    with open("embeddings_cache.json", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

# Embed with immediate failure if any error
embedded_docs = []

for i, doc in enumerate(all_docs):
    doc_hash = hash_content(doc)

    # Reuse cached if available
    if doc_hash in cache:
        doc["embedding"] = cache[doc_hash]
    else:
        try:
            content = doc.get("title", "") + " " + doc.get("content", "") + " " + doc.get("description", "")
            if not content.strip():
                raise ValueError("Document content is empty")

            emb = get_embedding(content[:1000])
            emb = emb.tolist() if hasattr(emb, "tolist") else emb

            doc["embedding"] = emb
            cache[doc_hash] = emb
        except Exception as e:
            print("\n❌ Failed to embed document:")
            print("🔹 Index:", i)
            print("🔹 Title:", doc.get("title", "ไม่ระบุหัวข้อ"))
            print("🔹 URL:", doc.get("url", "ไม่ระบุ URL"))
            print("🔹 Error:", str(e))
            raise SystemExit("🚨 Terminating early due to embedding failure.")

    embedded_docs.append(doc)

# Save results
save_json(embedded_docs, "embedded_docs.json")
save_json(cache, "embeddings_cache.json")
print(f"✅ Successfully embedded {len(embedded_docs)} documents")