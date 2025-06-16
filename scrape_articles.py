import requests
from bs4 import BeautifulSoup
import json
import time
from requests.exceptions import RequestException

# List of categories and their base URLs
categories = {
    "news": "https://www.tcc.or.th/category/news/",
    "banking": "https://www.tcc.or.th/category/banking/",
    "transport": "https://www.tcc.or.th/category/transport/",
    "real-estate": "https://www.tcc.or.th/category/real-estate/",
    "food-and-drug": "https://www.tcc.or.th/category/food-and-drug/",
    "health": "https://www.tcc.or.th/category/health/",
    "product": "https://www.tcc.or.th/category/product/",
    "telecom": "https://www.tcc.or.th/category/telecom/",
    "environment": "https://www.tcc.or.th/category/environment/",
    "education": "https://www.tcc.or.th/category/education/"
}

all_articles = []

MAX_RETRIES = 3
TIMEOUT = 30
RETRY_DELAY = 3

for category, base_url in categories.items():
    print(f"\n📂 Scraping category: {category}")
    page = 1

    while True:
        url = f"{base_url}page/{page}/" if page > 1 else base_url
        print(f"  → Page {page}: {url}")

        retries = 0
        while retries < MAX_RETRIES:
            try:
                res = requests.get(url, timeout=TIMEOUT)
                res.raise_for_status()
                break
            except RequestException as e:
                retries += 1
                print(f"    ⚠️ Retry {retries}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY)
        else:
            print("  ❌ Gave up after max retries.")
            break

        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.select("div.grid-article-item")
        if not articles:
            print("  ✅ No more articles. Stopping pagination.")
            break

        for item in articles:
            title_tag = item.select_one("h3.grid-article-title a")
            if title_tag:
                title = title_tag.get_text(strip=True)
                link = title_tag["href"]
                all_articles.append({
                    "category": category,
                    "title": title,
                    "url": link
                })

        page += 1
        time.sleep(1)  # Be polite

# Save to JSON
with open("articles.json", "w", encoding="utf-8") as f:
    json.dump(all_articles, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Saved {len(all_articles)} articles to articles.json")
