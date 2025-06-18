from flask import Flask, request, jsonify, session
from flask_cors import CORS
from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv
import numpy as np
import faiss
from flask import send_from_directory

load_dotenv()
app = Flask(__name__)
app.secret_key = 'your-super-secret-key'
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load complaint guide
with open("complaints.json", encoding="utf-8") as f:
    complaint_guide = json.load(f)

# Load FAISS index and metadata
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

# Embed 'abouts' live
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
            "คุณคือแชทบอทที่ช่วยแยกความตั้งใจของผู้ใช้:\n"
            "- หากต้องการข้อมูลเกี่ยวกับองค์กร เช่น วิสัยทัศน์, ภารกิจ, วิธีการทำงาน, การติดต่อองค์กร, เบอร์โทรศัพท์, อีเมล หรือแม้แต่การสอบถามช่องทางร้องเรียน ให้ตอบ: องค์กร\n"
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


def detect_complaint_type(user_msg):
    choices = []
    for category, subtypes in complaint_guide.items():
        for subtype in subtypes:
            choices.append(f"{category} > {subtype}")

    choice_list = "\n".join([f"- {c}" for c in choices])

    system = (
        "คุณคือผู้ช่วยจัดหมวดหมู่เรื่องร้องเรียนตามรายการที่มีอยู่ด้านล่าง "
        "จากข้อความของผู้ใช้ ให้เลือกหมวดหมู่ที่ตรงที่สุดจากรายการนี้เท่านั้น "
        "และตอบกลับเฉพาะชื่อหมวดหมู่ในรูปแบบ: ชื่อหมวดหลัก > ชื่อปัญหา\n\n"
        f"{choice_list}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content.strip()

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

@app.route("/ask", methods=["POST"])
def ask():
    user_msg = request.json["message"]
    intent = classify_user_intent(user_msg)
    print("📌 Intent:", intent)


    if session.get("complaint_form") and get_next_field(session["complaint_form"]):
        return complaint_flow()

    if intent == "ร้องเรียน":
        return complaint_flow()

    elif intent == "องค์กร":
        # Prioritize contact section if relevant
        sorted_docs = sorted(
            abouts,
            key=lambda d: (
                any(k in user_msg.lower() for k in ["ติดต่อ", "ร้องเรียน", "เฟซ", "facebook", "ไอจี", "line", "twitter", "tik", "เบอร์", "อีเมล"])
                and d["key"] == "contact",
                cosine_similarity(get_embedding(user_msg), np.array(d["embedding"]))
            ),
            reverse=True
        )[:10]

        # Optimized system prompt
        prompt = (
            "คุณคือแชทบอทที่ให้คำอธิบายเกี่ยวกับสภาองค์กรของผู้บริโภค โดยใช้ข้อมูลจากด้านล่างเท่านั้น\n"
            "หากข้อความของผู้ใช้เป็นการทักทายหรือเริ่มบทสนทนา ให้คุณตอบกลับด้วยความสุภาพและเป็นธรรมชาติ\n"
            "หากผู้ใช้สอบถามเกี่ยวกับการติดต่อหรือการร้องเรียน โปรดสรุปข้อมูลจากหมวด 'contact' โดยเน้น:\n"
            "- สายด่วน 1502\n"
            "- เว็บไซต์ร้องเรียนออนไลน์\n"
            "- เบอร์โทรศัพท์และอีเมล\n"
            "- ชื่อบัญชีของโซเชียลมีเดียทั้งหมด (Facebook, Instagram, LINE, Twitter/X, TikTok)\n"
            "จัดรูปแบบคำตอบให้อ่านง่าย เช่น รายการหัวข้อและอีโมจิหากเหมาะสม\n\n"
        )

        for doc in sorted_docs:
            prompt += f"- {doc['title']}: {doc.get('content', '')[:800]}...\n"

        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}]
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})


    else:
        docs = get_top_documents_by_similarity(user_msg, top_k=30)
        top_docs = gpt_rerank_documents(user_msg, docs)
        if not top_docs:
            return jsonify({"reply": "ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาลองใหม่ค่ะ"})

        reply = ""
        for doc in top_docs:
            title = doc.get("title", "ไม่ระบุหัวข้อ")
            url = doc.get("url", "#")
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


form_fields = [
    "ชื่อบริษัท / บุคคล",
    "วันที่เกิดเหตุ",
    "จังหวัดที่เกิดเหตุ",
    "รายละเอียด",
    "มูลค่าความเสียหาย",
    "แนบไฟล์"
]

def get_next_field(form_data):
    for field in form_fields:
        if form_data.get(field) is None:
            return field
    return None

@app.route("/complaint", methods=["POST"])
def complaint_flow():
    user_msg = request.json["message"]

    if "complaint_form" not in session:
        session["complaint_form"] = {
            "หมวดหมู่ร้องเรียน": None,
            "ประเภทย่อย": None,
            "ชื่อบริษัท / บุคคล": None,
            "วันที่เกิดเหตุ": None,
            "จังหวัดที่เกิดเหตุ": None,
            "รายละเอียด": None,
            "มูลค่าความเสียหาย": None,
            "แนบไฟล์": []
        }

    form = session["complaint_form"]

    if form["หมวดหมู่ร้องเรียน"] is None:
        match = detect_complaint_type(user_msg)
        print("📌 GPT match:", match)
        try:
            category, subtype = [m.strip() for m in match.split(">")]
            info = complaint_guide.get(category, {}).get(subtype, {})
            if info:
                form["หมวดหมู่ร้องเรียน"] = category
                form["ประเภทย่อย"] = subtype
                session["complaint_form"] = form

                doc_list = info.get("required_documents", [])
                guidance = info.get("guidance", [])
                doc_text = "\n".join(["- " + d for d in doc_list])
                guide_text = "\n".join(["- " + g for g in guidance])

                return jsonify({
                    "reply": f"📂 หมวดหมู่ร้องเรียนที่แนะนำ: {category} > {subtype}\n\n"
                    f"📌 เอกสารที่ควรแนบ:\n{doc_text}\n\n"
                    f"✍️ โปรดระบุ: {form_fields[0]}"
                })
            else:
                return jsonify({"reply": "ไม่พบหมวดหมู่ร้องเรียน กรุณาระบุให้ชัดเจนขึ้นค่ะ"})
        except:
            return jsonify({"reply": "ไม่สามารถแยกหมวดหมู่ได้ กรุณาอธิบายเพิ่มเติมค่ะ"})

    next_field = get_next_field(form)
    if next_field:
        form[next_field] = user_msg.strip()
        session["complaint_form"] = form
        next_prompt = get_next_field(form)
        if next_prompt:
            return jsonify({"reply": f"✍️ โปรดระบุ: {next_prompt}"})
        else:
            summary = "\n".join([f"- {k}: {v}" for k, v in form.items() if v])
            return jsonify({
                "reply": f"✅ คุณกรอกข้อมูลครบถ้วนแล้ว:\n\n{summary}\n\n"
                         f"หากต้องการแก้ไข พิมพ์ชื่อหัวข้อ เช่น 'แก้ไข จังหวัดที่เกิดเหตุ'"
            })
    else:
        return jsonify({"reply": "คุณได้กรอกข้อมูลครบถ้วนแล้ว หากต้องการแก้ไขกรุณาระบุหัวข้อที่ต้องการแก้ไขค่ะ"})

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
