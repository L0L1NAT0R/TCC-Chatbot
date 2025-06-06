from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv
from embedding_utils import get_embedding, cosine_similarity
import numpy as np

load_dotenv()
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data
with open("brochures.json", encoding="utf-8") as f1:
    brochures = json.load(f1)
with open("articles.json", encoding="utf-8") as f2:
    articles = json.load(f2)
with open("about.json", encoding="utf-8") as f3:
    abouts = json.load(f3)

all_docs = brochures + articles

# Embed all documents once
print("🔄 Embedding all documents...")
for doc in all_docs:
    content = doc.get("title", "") + " " + doc.get("content", "") + " " + doc.get("description", "")
    doc["embedding"] = get_embedding(content[:1000])  # limit to 1000 chars

for doc in abouts:
    content = doc.get("title", "") + " " + doc.get("content", "")
    doc["embedding"] = get_embedding(content[:1000])

print("✅ Embeddings ready.")

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[“”\"\'‘’.,!?()\-\–—:;]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def classify_user_intent(user_input):
    messages = [
        {"role": "system", "content": (
            "คุณคือแชทบอทที่ช่วยแยกความตั้งใจของผู้ใช้:\n"
            "- หากต้องการข้อมูลเกี่ยวกับองค์กร ให้ตอบ: องค์กร\n"
            "- หากต้องการคำแนะนำ บทความ หรือข่าว ให้ตอบ: ลิงก์\n"
            "ตอบเฉพาะคำว่า 'องค์กร' หรือ 'ลิงก์' เท่านั้น"
        )},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content.strip()

def get_top_documents_by_similarity(user_input, docs, top_k=30):
    user_embedding = get_embedding(user_input)
    scored = [(cosine_similarity(user_embedding, np.array(doc["embedding"])), doc) for doc in docs]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]

def gpt_rerank_documents(user_input, docs, max_results=3):
    doc_list_text = ""
    for i, doc in enumerate(docs):
        title = doc.get("title", "ไม่ระบุหัวข้อ")
        content = doc.get("content", "") or doc.get("description", "")
        doc_list_text += f"{i+1}. {title.strip()} — {content.strip()[:150]}...\n"

    system_prompt = (
        "คุณคือผู้ช่วยที่ช่วยเลือกลิงก์หรือเนื้อหาที่เกี่ยวข้องมากที่สุดกับคำถามของผู้ใช้ "
        "จากรายการที่ให้ด้านล่าง\n"
        f"{doc_list_text}\n"
        "จากรายการด้านบน ให้เลือก 3 หมายเลขที่ตรงหรือใกล้เคียงกับคำถามของผู้ใช้มากที่สุด "
        "ตอบเป็นตัวเลข เช่น 1, 4, 8"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    answer = response.choices[0].message.content.strip()
    print("📌 GPT selected indexes:", answer)

    indexes = [int(i) - 1 for i in re.findall(r"\d+", answer)]
    return [docs[i] for i in indexes if 0 <= i < len(docs)]

@app.route("/ask", methods=["POST"])
def ask():
    user_msg = request.json["message"]
    intent = classify_user_intent(user_msg)
    print("📌 Intent:", intent)

    if intent == "องค์กร":
        docs = get_top_documents_by_similarity(user_msg, abouts, top_k=10)
        top_docs = gpt_rerank_documents(user_msg, docs)
        prompt = "คุณคือแชทบอทที่ให้คำอธิบายเกี่ยวกับองค์กร จากข้อมูลด้านล่างเท่านั้น\n\n"
        for doc in top_docs:
            prompt += f"- {doc['title']}: {doc.get('content', '')[:500]}...\n"
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}]
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        reply = response.choices[0].message.content
    else:
        docs = get_top_documents_by_similarity(user_msg, all_docs, top_k=30)
        top_docs = gpt_rerank_documents(user_msg, docs)
        if not top_docs:
            reply = "ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาลองใหม่ค่ะ"
        else:
            reply = ""
            for doc in top_docs:
                title = doc.get("title", "ไม่ระบุหัวข้อ")
                url = doc.get("url", "#")
                reply += f'- <a href="{url}" target="_blank">{title}</a><br>'

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
