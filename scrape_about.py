import requests
from bs4 import BeautifulSoup
import json
import re

url = "https://www.tcc.or.th/aboutus/"
res = requests.get(url)
soup = BeautifulSoup(res.text, "html.parser")

# Get main content
main = soup.select_one("main.tcc-main-content .entry-content")
if not main:
    raise ValueError("❌ Cannot find about content")

text = main.get_text(separator="\n", strip=True)

# Normalize spacing
text = re.sub(r"\n+", "\n", text)

# Define split points
sections = {
    "overview": r"^สภาองค์กรของผู้บริโภค",
    "vision": r"^วิสัยทัศน์",
    "mission": r"^พันธกิจ",
    "strategy": r"^ยุทธศาสตร์",
    "revenue": r"^ที่มาของรายได้",
    "history": r"^กว่าจะเป็นสภาองค์กรของผู้บริโภค",
    "contact": r"^ติดต่อสภาผู้บริโภค"
}

# Use ordered positions to extract blocks
matches = [(key, re.search(pattern, text, re.M)) for key, pattern in sections.items()]
matches = [(k, m.start()) for k, m in matches if m]
matches.sort(key=lambda x: x[1])
matches.append(("end", len(text)))

# Slice the text
output = []
for i in range(len(matches) - 1):
    key = matches[i][0]
    start = matches[i][1]
    end = matches[i + 1][1]
    output.append({
        "title": key,
        "content": text[start:end].strip()
    })

with open("about.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("✅ Saved structured 'about.json' with", len(output), "sections")
