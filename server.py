from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import re
from pythainlp.tokenize import word_tokenize
from difflib import SequenceMatcher

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

# Normalize text
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[“”\"\'‘’.,!?()\-\–—:;]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Matching logic
def filter_documents(user_msg, docs, max_docs=30, skip_title_filter=False):
    user_msg_clean = normalize(user_msg)
    user_tokens = word_tokenize(user_msg_clean, engine="newmm")

    filler_words = {
        "ที่", "ของ", "กับ", "โดย", "เพื่อ", "ซึ่ง", "เป็น", "ให้", "แล้ว", "จะ",
        "ก็", "ยัง", "และ", "หรือ", "แต่", "เพราะ", "จึง", "ดังนั้น", "แม้", "หาก",
        "เมื่อ", "จน", "ตาม", "ขณะ", "เนื่องจาก", "ทำ", "มี", "อยู่", "ไป", "มา",
        "ใช้", "ช่วย", "บอก", "รู้", "ควร", "สามารถ", "ต้อง", "ได้", "ไม่", "ฉัน",
        "คุณ", "เรา", "เขา", "เธอ", "มัน", "หน่อย", "นะ", "ค่ะ", "ครับ", "จ้า",
        "จ๊ะ", "หนะ", "ล่ะ", "เอง", "สิ่ง", "เรื่อง", "อย่าง", "รายการ", "ข้อมูล",
        "คำ", "แบบ", "ชนิด", "อะไร", "อย่างไร", "ทำไม", "ไหน", "ใคร", "เมื่อไหร่", "ที่ไหน"
    }

    signal_words = [
        w for w in user_tokens if w.strip() and w not in filler_words and len(w.strip()) >= 2
    ]

    semantic_triggers = {
        "overview": [
            "คืออะไร", "เกี่ยวกับ", "หน้าที่", "ทำอะไร", "องค์กรอะไร", "overview", "ทำหน้าที่"
        ],
        "mission": [
            "พันธกิจ", "เป้าหมาย", "ทำเพื่ออะไร", "บทบาท", "ภารกิจ", "mission"
        ],
        "history": [
            "ประวัติ", "ก่อตั้ง", "เมื่อไหร่", "ปี", "เริ่มต้น", "history"
        ],
        "revenue": [
            "รายได้", "การเงิน", "งบประมาณ", "เงิน", "income", "revenue", "ค่าธรรมเนียม", "เงินอุดหนุน"
        ],
        "contact": [
            "ติดต่อ", "เบอร์", "อีเมล", "ที่อยู่", "สำนักงาน", "contact", "phone", "address"
        ],
        "vision": [
            "วิสัยทัศน์", "เป้าหมายสูงสุด", "อนาคต", "เห็นภาพ", "มุ่งหวัง", "vision"
        ],
        "strategy": [
            "กลยุทธ์", "ยุทธศาสตร์", "แผนงาน", "การดำเนินงาน", "strategy", "ระยะยาว", "แผน 5 ปี"
        ]
    }


    prompt_boost_category = None
    for section, keywords in semantic_triggers.items():
        for kw in keywords:
            if kw in user_msg_clean or any(kw in token for token in user_tokens):
                prompt_boost_category = section
                break
        if prompt_boost_category:
            break

    scored = []

    for doc in docs:
        title = doc.get("title", "")
        content = doc.get("content", "")
        title_norm = normalize(title)
        content_norm = normalize(content)

        title_tokens = word_tokenize(title_norm, engine="newmm")
        content_tokens = word_tokenize(content_norm, engine="newmm") if content else []

        if not skip_title_filter and not any(word in title_tokens for word in signal_words):
            continue

        score = 0

        if prompt_boost_category and normalize(title) == prompt_boost_category:
            score += 5


        title_score = sum(
            2 if word == token else 1
            for word in signal_words
            for token in title_tokens
            if word in token or token in word
        )
        score += 2 * title_score

        content_score = sum(
            1
            for word in signal_words
            for token in content_tokens[:30]
            if word in token or token in word
        )
        score += min(content_score, 5)

        if skip_title_filter and content:
            similarity = SequenceMatcher(None, user_msg_clean, content_norm).ratio()
            if similarity >= 0.5:
                score += 3

        # ✅ Hashtag match boost — apply only if present
        # ✅ Hashtag match boost — apply only if present
        hashtags = doc.get("hashtags", [])
        hashtag_score = 0
        for tag in hashtags:
            tag_norm = normalize(tag)

            # Check direct inclusion (cleaned)
            if tag_norm in user_msg_clean:
                print(f"🎯 Exact hashtag matched: {tag_norm}")
                hashtag_score += 12
                continue

            # Check token overlap if not full phrase match
            tag_tokens = [tok for tok in word_tokenize(tag_norm, engine="newmm") if tok.strip()]
            match_count = sum(1 for tok in tag_tokens if any(tok in utok or utok in tok for utok in user_tokens))

            if match_count == len(tag_tokens):
                print(f"✅ Full token overlap with hashtag: {tag_norm}")
                hashtag_score += 10
            elif match_count >= 1:
                print(f"➕ Partial token overlap with hashtag: {tag_norm}")
                hashtag_score += 5

        score += hashtag_score


        if score >= 3 or skip_title_filter:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    print("✅ MATCHED TITLES:")
    for score, doc in scored[:max_docs]:
        print(f"- ({score}) {doc.get('title', '❌ no title')}")

    return scored[:max_docs]

@app.route("/ask", methods=["POST"])
def ask():
    user_msg = request.json["message"]
    user_msg_clean = normalize(user_msg)  # move this BEFORE tokenization
    user_tokens = [tok for tok in word_tokenize(user_msg_clean, engine="newmm") if tok.strip()]

    about_keywords = [
        "สภาผู้บริโภค", "tcc", "องค์กร", "บริษัท", "เกี่ยวกับบริษัท", "หน้าที่", "ประวัติ", "ก่อตั้ง",
        "กฎหมาย", "ติดต่อ", "เบอร์", "อีเมล", "สำนักงาน", "เลขาธิการ", "สำนักงานใหญ่"
    ]

    matched_about = [
        kw for kw in about_keywords
        if kw in user_msg_clean or any(kw in token for token in user_tokens)
    ]

    fuzzy_match = any(
        SequenceMatcher(None, kw, user_msg_clean).ratio() >= 0.85
        for kw in about_keywords
    )

    is_about_company = len(matched_about) > 0 or fuzzy_match

    print("🔍 TOKENS:", user_tokens)
    print("🏢 Matched about-org words:", matched_about)
    print("🧠 Fuzzy match:", fuzzy_match)
    print("🏢 Is about company:", is_about_company)

    if is_about_company:
        docs = abouts
    else:
        docs = brochures + articles

    filtered_docs = filter_documents(user_msg, docs, max_docs=3, skip_title_filter=is_about_company)

    if not filtered_docs:
        reply = "ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาลองใหม่ค่ะ"
    else:
        if is_about_company:
            top = filtered_docs[0] if filtered_docs else None
            if top and top[0] >= 3:
                score, doc = top
                title = doc.get("title", "")
                content = doc.get("content", "")
                reply = f"<strong>{title}</strong><br>{content[:800]}..."
            else:
                reply = "ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาลองใหม่ค่ะ"
        else:
            reply = ""
            for score, doc in filtered_docs[:3]:
                title = doc.get("title", "ไม่ระบุหัวข้อ")
                url = doc.get("url", "#")
                reply += f'- <a href="{url}" target="_blank">{title}</a><br>'
            if not reply:
                reply = "ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาลองใหม่ค่ะ"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("✅ Server running at http://localhost:5000")
    app.run(debug=True)
    