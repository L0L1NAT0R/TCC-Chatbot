from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv
import numpy as np
import faiss

load_dotenv()
app = Flask(__name__)
app.secret_key = 'your-super-secret-key'
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load and merge all public resources: articles, infographics, videos
with open("articles.json", encoding="utf-8") as f:
    articles = json.load(f)
    for a in articles:
        a["source"] = "article"

with open("infographics.json", encoding="utf-8") as f:
    brochures = json.load(f)
    for b in brochures:
        b["source"] = "brochure"

with open("videos.json", encoding="utf-8") as f:
    videos = json.load(f)
    for v in videos:
        v["source"] = "video"

links = articles + brochures + videos

# Load complaint guide
with open("complaints.json", encoding="utf-8") as f:
    complaint_guide = json.load(f)

# Load complaint type classifier (8 top-level types)
with open("complaint_types.json", encoding="utf-8") as f:
    complaint_types_keywords = json.load(f)


# Load FAISS index and metadata for reranking
print("📅 Loading FAISS index and metadata...")
faiss_index = faiss.read_index("index.faiss")
with open("doc_metadata.json", encoding="utf-8") as f:
    doc_metadata = json.load(f)
print(f"✅ Loaded {len(doc_metadata)} documents into metadata")

# Load about.json separately for in-memory search (small size)
with open("about.json", encoding="utf-8") as f:
    abouts = json.load(f)
    for ab in abouts:
        ab["source"] = "about"

from embedding_utils import get_embedding, cosine_similarity
for doc in abouts:
    content = doc.get("title", "") + " " + doc.get("content", "")
    emb = get_embedding(content[:1000])
    doc["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb



def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[“”\"\'‘’.,!?()\-\–—:;]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def classify_user_intent(user_input):
    messages = [
            {"role": "system", "content": (
        "คุณคือแชทบอทที่ช่วยแยกประเภทความตั้งใจของผู้ใช้ จากข้อความด้านล่าง\n\n"
        "- หากผู้ใช้ถามเกี่ยวกับองค์กร เช่น โครงสร้าง วิสัยทัศน์ วิธีทำงาน การติดต่อ เบอร์โทร อีเมล หรือร้องเรียน ให้ตอบว่า: องค์กร\n"
        "- หากผู้ใช้เขียนเล่าเหตุการณ์หรืออธิบายปัญหาโดยละเอียด เช่น การถูกเอาเปรียบหรือความเสียหาย ให้ตอบว่า: เรื่องร้องเรียน\n"
        "- ในกรณีอื่น ๆ เช่น ต้องการคำแนะนำ แนวทางแก้ปัญหา ข่าว ความรู้ บทความ หรืออินโฟกราฟิก ให้ตอบว่า: ลิงก์\n"
        "ตอบกลับด้วยคำว่า: องค์กร หรือ ลิงก์ หรือ เรื่องร้องเรียน เท่านั้น"
    )},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content.strip()


def select_relevant_infographics(user_msg, infographics):
    titles_text = "\n".join([f"{i+1}. {item['title']}" for i, item in enumerate(infographics)])

    system_prompt = (
        "คุณคือแชทบอทที่ช่วยเลือกอินโฟกราฟิกที่เกี่ยวข้องกับข้อความของผู้ใช้ "
        "จากรายการด้านล่างนี้:\n\n"
        f"{titles_text}\n\n"
        "จากรายการข้างต้น ให้เลือกอินโฟกราฟิกที่เกี่ยวข้องกับคำถามหรือปัญหาของผู้ใช้ "
        "โดยตอบกลับเป็นหมายเลขคั่นด้วยคอมมา เช่น 1, 3, 6 (อย่าอธิบายเพิ่ม)"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    selected_indexes = [int(i.strip()) - 1 for i in re.findall(r"\d+", response.choices[0].message.content)]
    return [infographics[i] for i in selected_indexes if 0 <= i < len(infographics)]

def get_top_documents_by_similarity(user_input, top_k=30):
    user_embedding = get_embedding(user_input).astype("float32")
    D, I = faiss_index.search(np.array([user_embedding]), top_k)
    return [doc_metadata[i] for i in I[0] if 0 <= i < len(doc_metadata)]

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


def gpt_classify_complaint_type(user_msg):
    trait_list = ""
    for category, traits in complaint_types_keywords.items():
        example_traits = " / ".join(traits[:4])
        trait_list += f"- {category}: {example_traits}\n"

    system_prompt = (
        "คุณคือแชทบอทที่ช่วยจัดหมวดหมู่ของเรื่องร้องเรียน โดยพิจารณาจากข้อความของผู้ใช้ "
        "และเปรียบเทียบกับหมวดหมู่ที่กำหนดไว้ด้านล่าง:\n\n"
        f"{trait_list}\n\n"
        "จากข้อความของผู้ใช้ ให้ตอบเพียงชื่อหมวดหมู่ที่ตรงหรือใกล้เคียงที่สุดจากรายการด้านบนเท่านั้น "
        "โดยไม่ต้องอธิบายเพิ่มเติม"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content.strip()

@app.route("/ask", methods=["POST"])
def ask():
    user_msg = request.json["message"]
    intent = classify_user_intent(user_msg)
    print("📌 Intent:", intent)

    if intent == "ร้องเรียน":
        relevant = select_relevant_infographics(user_msg, infographics)

        if not relevant:
            reply = "ไม่พบอินโฟกราฟิกที่เกี่ยวข้องโดยตรงกับปัญหานี้ค่ะ "
        else:
            reply = "📊 อินโฟกราฟิกที่อาจเกี่ยวข้องกับปัญหาของคุณ:\n\n"
            for item in relevant:
                reply += f"- <a href='{item['link']}' target='_blank'>{item['title']}</a>\n"


        reply += (
            "\nหากข้อมูลเหล่านี้ยังไม่เพียงพอ คุณสามารถโทรสายด่วนผู้บริโภคที่เบอร์ "
            "<strong>1502</strong> ในวันและเวลาทำการ หรือร้องเรียนผ่านระบบออนไลน์ที่ "
            "<a href='https://complaint.tcc.or.th/complaint' target='_blank'>complaint.tcc.or.th</a>"
        )
        return jsonify({"reply": reply})


    elif intent == "องค์กร":
    # Use GPT to detect if the user wants to contact or complain
        contact_check = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "ต่อไปนี้คืองานของคุณ:\n"
                    "- หากข้อความของผู้ใช้ต้องการติดต่อองค์กร เช่น การร้องเรียน, ขอข้อมูลเบอร์โทร, อีเมล, เฟซบุ๊ก หรือช่องทางโซเชียลอื่น ๆ ตอบว่า 'ติดต่อ'\n"
                    "- หากไม่เกี่ยวข้องกับการติดต่อหรือร้องเรียน ตอบว่า 'ทั่วไป'\n"
                    "- ห้ามตอบคำอื่นนอกจาก 'ติดต่อ' หรือ 'ทั่วไป'"
                )},
                {"role": "user", "content": user_msg}
            ]
        ).choices[0].message.content.strip()

        if contact_check == "ติดต่อ":
            reply = (
                "หากคุณต้องการร้องเรียนกับสภาผู้บริโภค คุณสามารถใช้ช่องทางดังต่อไปนี้ได้เลยค่ะ:<br><br>"
                "📞 <strong>สายด่วนผู้บริโภค</strong>: 1502 (ในวันและเวลาทำการ)<br>"
                "🌐 <strong>เว็บไซต์ร้องเรียนออนไลน์</strong>: "
                "<a href='https://complaint.tcc.or.th/complaint' target='_blank'>https://complaint.tcc.or.th/complaint</a><br>"
                "🖥 <strong>ช่องทางโซเชียลมีเดีย</strong>:<br>"
                "- Facebook: <a href='https://www.facebook.com/tccthailand' target='_blank'>@tccthailand</a><br>"
                "- Instagram: <a href='https://www.instagram.com/tcc.thailand' target='_blank'>@tccthailand</a><br>"
                "- LINE: <a href='https://line.me/R/ti/p/@tccthailand' target='_blank'>@tccthailand</a><br>"
                "- Twitter/X: <a href='https://twitter.com/tccthailand' target='_blank'>@tccthailand</a><br>"
                "- TikTok: <a href='https://www.tiktok.com/@tccthailand' target='_blank'>@tccthailand</a><br><br>"
                "คุณสามารถติดต่อสอบถามหรือแจ้งปัญหาได้ผ่านช่องทางเหล่านี้ค่ะ 😊"
            )
            return jsonify({"reply": reply})

        # Fallback to document summary if not contact-related
        sorted_docs = sorted(
            abouts,
            key=lambda d: cosine_similarity(get_embedding(user_msg), np.array(d["embedding"])),
            reverse=True
        )[:10]

        prompt = (
            "คุณคือแชทบอทที่ให้คำอธิบายเกี่ยวกับสภาองค์กรของผู้บริโภค โดยใช้ข้อมูลจากด้านล่างเท่านั้น\n"
            "หากข้อความของผู้ใช้เป็นการทักทายหรือเริ่มบทสนทนา ให้คุณตอบกลับด้วยความสุภาพและเป็นธรรมชาติ\n\n"
        )

        for doc in sorted_docs:
            prompt += f"- {doc['title']}: {doc.get('content', '')[:800]}...\n"

        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}]
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    
    elif intent == "เรื่องร้องเรียน":
        category = gpt_classify_complaint_type(user_msg)
        reply = (
            f"📂 จากข้อความของคุณ ระบบคาดว่าเป็นหมวดหมู่: <strong>{category}</strong>\n\n"
            "คุณสามารถดำเนินการร้องเรียนโดยกรอกแบบฟอร์มได้ที่:\n"
            "<a href='https://complaint.tcc.or.th/complaint' target='_blank'>📨 แบบฟอร์มร้องเรียนออนไลน์</a>\n\n"
            "หากหมวดหมู่ไม่ตรง คุณสามารถเลือกใหม่ได้ภายในแบบฟอร์มค่ะ"
        )
        return jsonify({"reply": reply})


    else:
        docs = get_top_documents_by_similarity(user_msg, top_k=30)
        top_docs = gpt_rerank_documents(user_msg, docs)
        if not top_docs:
            return jsonify({"reply": "ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาลองใหม่ค่ะ"})

        reply = ""
        for doc in top_docs:
            title = doc.get("title", "ไม่ระบุหัวข้อ")
            url = doc.get("url") or doc.get("link", "#")
            source = doc.get("source", "")
            if source == "brochure":
                label = "📊 อินโฟกราฟิกจากเว็บไซต์"
            elif source == "article":
                label = "📰 ข่าว/บทความจากเว็บไซต์"
            elif source == "video":
                label = "🎥 วิดีโอจาก YouTube ช่อง TCC"
                video_id = url.split("v=")[-1] if "v=" in url else ""
                thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                reply += f'{label}<br><a href="{url}" target="_blank"><img src="{thumbnail}" alt="{title}" style="width:100%;max-width:320px;border-radius:10px;margin-bottom:4px;"><br>{title}</a><br><br>'
                continue
            else:
                label = "📁 แหล่งข้อมูลอื่น"

            reply += f'{label}<br>- <a href="{url}" target="_blank">{title}</a><br><br>'

        return jsonify({"reply": reply})




@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    if os.path.exists(filename):
        return send_from_directory(".", filename)
    return "File not found", 404


if __name__ == "__main__":
    app.run(debug=True)
