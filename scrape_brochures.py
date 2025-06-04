import requests
from bs4 import BeautifulSoup
import json

base_url = "https://www.tcc.or.th/tcc-media-type/infographic/page/"
page = 1
brochures = []

while True:
    url = base_url + str(page) + "/"
    print(f"🔎 Fetching: {url}")
    res = requests.get(url)
    if res.status_code != 200:
        print("❌ Page not found or blocked")
        break

    soup = BeautifulSoup(res.text, "html.parser")

    items = soup.select("div.media1-grid-item")
    if not items:
        print("✅ No more infographics found.")
        break

    for item in items:
        link = item.select_one("a")["href"]
        title = item.select_one("h3.media1-title").text.strip()

        brochures.append({
            "title": title,
            "url": link
        })

    page += 1

# Save as JSON
with open("brochures.json", "w", encoding="utf-8") as f:
    json.dump(brochures, f, ensure_ascii=False, indent=2)

print(f"✅ Done. Saved {len(brochures)} brochures to brochures.json")
