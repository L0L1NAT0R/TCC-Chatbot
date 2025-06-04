import requests
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# =========================
# 🧾 PART 1: TCC Media Pages
# =========================
base_url = "https://www.tcc.or.th/tcc-media-type/infographic/page/"
page = 1
brochures = []

while True:
    url = base_url + str(page) + "/"
    print(f"🔎 Fetching TCC page: {url}")
    try:
        res = requests.get(url, timeout=10)
    except Exception as e:
        print(f"❌ Failed to fetch TCC page: {e}")
        break

    if res.status_code != 200:
        break

    soup = BeautifulSoup(res.text, "html.parser")
    items = soup.select("div.media1-grid-item")
    if not items:
        break

    for item in items:
        link = item.select_one("a")["href"]
        title = item.select_one("h3.media1-title").text.strip()
        brochures.append({
            "title": title,
            "url": link
        })

    page += 1

# ============================
# 🧾 PART 2: CRM Infographic Page (JS-rendered via Selenium)
# ============================
print("🌐 Launching headless browser for CRM infographics...")
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)

crm_url = "https://crm.tcc.or.th/portal/public/?entryPoint=Portal&action=knowledgeBaseArticle&problem_type"
driver.get(crm_url)
time.sleep(3)  # Wait for JS to load content

soup = BeautifulSoup(driver.page_source, "html.parser")

faq_items = soup.select("div.faq-item")
for item in faq_items:
    # Title
    title_tag = item.select_one("a.faq-question")
    title = title_tag.text.strip() if title_tag else "ไม่พบชื่อ"

    # Hashtags
    hashtags = [span.text.strip() for span in item.select("span.tag-item")]

    # Image URL (from expanded answer panel)
    collapse_id = title_tag.get("href").replace("#", "") if title_tag else ""
    collapse_div = soup.find("div", id=collapse_id)
    img_tag = collapse_div.select_one("img") if collapse_div else None

    if not img_tag:
        continue

    src = img_tag.get("src")
    if not src.startswith("http"):
        src = "https://crm.tcc.or.th" + src

    brochures.append({
        "title": title,
        "url": src,
        "hashtags": hashtags
    })

driver.quit()

# ==================
# Save to File
# ==================
with open("brochures.json", "w", encoding="utf-8") as f:
    json.dump(brochures, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(brochures)} items to brochures.json")
