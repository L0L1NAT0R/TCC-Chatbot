# embedding_utils.py
import numpy as np
from openai import OpenAI
import os
import json
from dotenv import load_dotenv, find_dotenv

# ✅ Load environment variables BEFORE using them
load_dotenv(find_dotenv())

# ✅ Safely get API key (strip quotes if needed)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY").strip('"'))

def get_embedding(text, model="text-embedding-3-small"):
    result = client.embeddings.create(
        model=model,
        input=[text]
    )
    return np.array(result.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))