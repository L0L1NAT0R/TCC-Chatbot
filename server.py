from flask import Flask, request, jsonify #use flask web server
from flask_cors import CORS #allow frontend the input to talk with the server so the server can give output
from openai import OpenAI #use gpt 3.5 to answer question
import os
import json
import re
from dotenv import load_dotenv #load API key in GPT
from embedding_utils import get_embedding, cosine_similarity
import numpy as np #for cosine similarity work with vectors

load_dotenv()
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data from all json files that were compiled using the scrape scripts 
with open("brochures.json", encoding="utf-8") as f1:
    brochures = json.load(f1)
    for b in brochures:
        b["source"] = "brochure"

with open("articles.json", encoding="utf-8") as f2:
    articles = json.load(f2)
    for a in articles:
        a["source"] = "article"

with open("about.json", encoding="utf-8") as f3:
    abouts = json.load(f3)
    for ab in abouts:
        ab["source"] = "about"

# Embed 'abouts' live and ensure embeddings are list-type (not np.array)
for doc in abouts:
    content = doc.get("title", "") + " " + doc.get("content", "")
    emb = get_embedding(content[:1000])
    doc["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb

with open("videos.json", encoding = "utf-8") as f4:
    videos = json.load(f4)
    for v in videos:
        v["source"] = "video"

all_docs = brochures + articles + videos



# Embed all documents into vectors used to compare similarity
# Load precomputed embeddings
print("📥 Loading embedded documents...")
with open("embedded_docs.json", encoding="utf-8") as f:
    all_docs = json.load(f)
print(f"✅ Loaded {len(all_docs)} embedded documents")

#basically removes common punctuations in the string to purely evaluate word meaning
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[“”\"\'‘’.,!?()\-\–—:;]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
#use gpt 4o to perform classification and return a binary choice of reteurn link or user based on the links
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


#this function looks at cosine similarity of embedding and picks the top 30 most similar in meaning and etc.
def get_top_documents_by_similarity(user_input, docs, top_k=30):
    user_embedding = get_embedding(user_input)
    scored = [(cosine_similarity(user_embedding, np.array(doc["embedding"])), doc) for doc in docs]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]
#have the model rerank the top 30 and select the top 3 most relevant
def gpt_rerank_documents(user_input, docs):
    doc_list_text = ""
    for i, doc in enumerate(docs):
        title = doc.get("title", "ไม่ระบุหัวข้อ")
        content = doc.get("content", "") or doc.get("description", "")
        doc_list_text += f"{i+1}. {title.strip()} — {content.strip()[:150]}...\n"

    system_prompt = (
        "คุณคือผู้ช่วยที่ช่วยเลือกลิงก์หรือเนื้อหาที่เกี่ยวข้องมากที่สุดกับคำถามของผู้ใช้ "
        "จากรายการที่ให้ด้านล่าง\n\n"
        f"{doc_list_text}\n\n"
        "จากรายการด้านบน ให้เลือกลิงก์ทั้งหมดที่ตรงหรือใกล้เคียงกับคำถามของผู้ใช้มากที่สุด "
        "ตอบเป็นหมายเลขคั่นด้วยคอมมา เช่น 1, 4, 8 (ไม่จำกัดจำนวน)"
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
#perform this if asked about the company
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
                source = doc.get("source", "")

                if source == "brochure":
                    label = "📊 อินโฟกราฟิกจากเว็บไซต์"
                    reply += f'{label}<br>- <a href="{url}" target="_blank">{title}</a><br><br>'

                elif source == "article":
                    label = "📰 ข่าว/บทความจากเว็บไซต์"
                    reply += f'{label}<br>- <a href="{url}" target="_blank">{title}</a><br><br>'

                elif source == "video":
                    label = "🎥 วิดีโอจาก YouTube ช่อง TCC"
                    video_id = url.split("v=")[-1] if "v=" in url else ""
                    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                    reply += (
                        f'{label}<br>'
                        f'<a href="{url}" target="_blank">'
                        f'<img src="{thumbnail_url}" alt="{title}" style="width:100%;max-width:320px;border-radius:10px;margin-bottom:4px;"><br>'
                        f'{title}</a><br><br>'
                    )

                else:
                    label = "📁 แหล่งข้อมูลอื่น"
                    reply += f'{label}<br>- <a href="{url}" target="_blank">{title}</a><br><br>'



    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)