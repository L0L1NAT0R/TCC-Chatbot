import json
import numpy as np
import faiss

with open("embedded_docs.json", encoding="utf-8") as f:
    embedded_docs = json.load(f)

embeddings = []
metadata = []

for doc in embedded_docs:
    if "embedding" in doc:
        embeddings.append(doc["embedding"])
        doc.pop("embedding", None)
        metadata.append(doc)

embedding_matrix = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)
faiss.write_index(index, "index.faiss")

with open("doc_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Index built with {index.ntotal} documents")
