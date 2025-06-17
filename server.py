from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv
from embedding_utils import get_embedding, cosine_similarity
import numpy as np
from flask import session

load_dotenv()
app = Flask(__name__)
app.secret_key = 'your-super-secret-key'  # needed for session tracking
CORS(app, supports_credentials= True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
form_fields = [
    "ชื่อบริษัท / บุคคล",
    "วันที่เกิดเหตุ",
    "จังหวัดที่เกิดเหตุ",
    "รายละเอียด",
    "มูลค่าความเสียหาย",
    "แนบไฟล์"
]

def extract_fields(user_msg):
    field_names = form_fields
    prompt = (
        "คุณคือผู้ช่วยที่จะแยกข้อมูลจากข้อความของผู้ใช้เพื่อกรอกฟอร์มร้องเรียน\n"
        f"หัวข้อที่ต้องกรอก ได้แก่: {', '.join(field_names)}\n"
        "กรุณาตอบกลับในรูปแบบ JSON เท่านั้น เช่น:\n"
        '{"ชื่อบริษัท / บุคคล": "Foodland", "จังหวัดที่เกิดเหตุ": "เชียงใหม่"}'
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_msg}
    ]

    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        extracted = json.loads(response.choices[0].message.content)
        return extracted if isinstance(extracted, dict) else {}
    except Exception as e:
        print("❌ Field extraction failed:", e)
        return {}

# Load main content data
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

# Embed 'abouts' live
for doc in abouts:
    content = doc.get("title", "") + " " + doc.get("content", "")
    emb = get_embedding(content[:1000])
    doc["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb

with open("videos.json", encoding="utf-8") as f4:
    videos = json.load(f4)
    for v in videos:
        v["source"] = "video"

all_docs = brochures + articles + videos

# Load precomputed embeddings
# Load compressed precomputed embeddings
print("📥 Loading compressed embedded documents...")
embedding_data = np.load("embeddings_data.npz", allow_pickle=True)
embeddings = embedding_data["embeddings"]
texts = embedding_data["texts"]
sources = embedding_data["sources"]
print(f"✅ Loaded {len(texts)} embedded documents using {embeddings.nbytes / (1024 * 1024):.2f} MB")


# Load complaint types and instructions
with open("complaints.json", encoding="utf-8") as f5:
    complaint_guide = json.load(f5)

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
            "- หากต้องการร้องเรียนปัญหา ให้ตอบ: ร้องเรียน\n"
            "ตอบเฉพาะคำว่า 'องค์กร', 'ลิงก์' หรือ 'ร้องเรียน' เท่านั้น"
        )},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content.strip()

def detect_complaint_type(user_msg):
    # Build the list of valid complaint categories
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


def get_top_documents_by_similarity(user_input, docs=None, top_k=30):
    user_embedding = get_embedding(user_input)
    sims = [cosine_similarity(user_embedding, vec) for vec in embeddings]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    
    top_docs = []
    for i in top_indices:
        top_docs.append({
            "title": texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i],
            "content": texts[i],
            "url": "#",  # You can modify this if you saved actual URLs
            "source": sources[i]
        })
    return top_docs


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
    
    # Check for edit command first
    if session.get("complaint_form") and user_msg.startswith("แก้ไข"):
        field_to_edit = user_msg.replace("แก้ไข", "").strip()
        if field_to_edit in form_fields:
            session["complaint_form"][field_to_edit] = None
            next_field = get_next_field(session["complaint_form"])
            return jsonify({
                "reply": f"✍️ โปรดระบุ {field_to_edit} ใหม่:"
            })
    
    # Check if we're already inside an active complaint session
    if session.get("complaint_form"):
        return complaint_flow()
    
    intent = classify_user_intent(user_msg)
    print("📌 Intent:", intent)

    # If user just started a complaint
    if intent == "ร้องเรียน":
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
        return complaint_flow()
    
    elif intent == "องค์กร":
        docs = get_top_documents_by_similarity(user_msg, abouts, top_k=10)
        top_docs = gpt_rerank_documents(user_msg, docs)
        prompt = "คุณคือแชทบอทที่ให้คำอธิบายเกี่ยวกับองค์กร จากข้อมูลด้านล่างเท่านั้น\n\n"
        for doc in top_docs:
            prompt += f"- {doc['title']}: {doc.get('content', '')[:500]}...\n"
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}]
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    else:
        docs = get_top_documents_by_similarity(user_msg, all_docs, top_k=30)
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
    
# 🔄 Multi-step complaint assistant form (structured response)

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

    # STEP 0: Classification
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

    # STEP 1+: Collect fields
        # STEP 1+: Try extracting multiple fields from user input
    extracted_fields = extract_fields(user_msg)
    for field, value in extracted_fields.items():
        if field in form and form[field] is None:
            form[field] = value.strip()

    session["complaint_form"] = form

    next_field = get_next_field(form)
    if next_field:
        examples = {
            "ชื่อบริษัท / บุคคล": "เช่น Foodland, Shopee",
            "วันที่เกิดเหตุ": "เช่น 5 มิ.ย. 2025",
            "จังหวัดที่เกิดเหตุ": "เช่น กรุงเทพมหานคร",
            "รายละเอียด": "กรุณาเล่าปัญหาที่เกิดขึ้น",
            "มูลค่าความเสียหาย": "เช่น 1,500 บาท",
            "แนบไฟล์": "สามารถแนบภาพหลักฐาน ใบเสร็จ แชท ฯลฯ"
        }
        example = examples.get(next_field, "")
        reply = f"✍️ โปรดระบุ: {next_field}"
        if example:
            reply += f"\n{example}"
        return jsonify({"reply": reply})
    else:
        # All fields are filled
        summary = "\n".join([f"- {k}: {v}" for k, v in form.items() if v])
        return jsonify({
            "reply": f"✅ คุณกรอกข้อมูลครบถ้วนแล้ว:\n\n{summary}\n\n"
                     f"หากต้องการแก้ไข พิมพ์ชื่อหัวข้อ เช่น 'แก้ไข จังหวัดที่เกิดเหตุ'"
        })


#neeeded for the application to run properly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
