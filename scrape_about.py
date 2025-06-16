import requests
from bs4 import BeautifulSoup
import json
import re
import time
from requests.exceptions import RequestException

url = "https://www.tcc.or.th/aboutus/"
MAX_RETRIES = 3
TIMEOUT = 30

# Fetch page with retry logic
for attempt in range(MAX_RETRIES):
    try:
        res = requests.get(url, timeout=TIMEOUT)
        res.raise_for_status()
        break
    except RequestException as e:
        print(f"⚠️ Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
        time.sleep(3)
else:
    raise RuntimeError("❌ Failed to fetch about page after retries.")

soup = BeautifulSoup(res.text, "html.parser")

# Get the full raw content text
main = soup.select_one("article .entry-content")
if not main:
    raise ValueError("❌ Cannot find about content")

text = main.get_text(separator="\n", strip=True)
text = re.sub(r"\n+", "\n", text)

# Define the 7 fixed section markers
titles = {
    "overview": r"สภาองค์กรของผู้บริโภค",
    "vision": r"วิสัยทัศน์",
    "mission": r"พันธกิจ",
    "strategy": r"ยุทธศาสตร์",
    "revenue": r"ที่มาของรายได้",
    "history": r"กว่าจะเป็นสภาองค์กรของผู้บริโภค",
    "contact": r"ติดต่อสภาผู้บริโภค"
}

# Find all matching positions in the text
matches = []
for key, title in titles.items():
    match = re.search(rf"^{title}", text, re.M)
    if match:
        matches.append((key, match.start()))
    else:
        print(f"⚠️ Section not found: {title}")

# Sort matches in text order
matches.sort(key=lambda x: x[1])
matches.append(("end", len(text)))

# Slice into chunks
sections = []
for i in range(len(matches) - 1):
    key = matches[i][0]
    start = matches[i][1]
    end = matches[i + 1][1]
    chunk = text[start:end].strip()
    first_line, *rest = chunk.splitlines()
    sections.append({
        "key": key,
        "title": first_line.strip(),
        "content": "\n".join(rest).strip()
    })

# Save as JSON
with open("about.json", "w", encoding="utf-8") as f:
    json.dump(sections, f, ensure_ascii=False, indent=2)

print(f"✅ Done — {len(sections)} sections saved to about.json")
