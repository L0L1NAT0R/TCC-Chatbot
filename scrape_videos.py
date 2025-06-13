import json
import requests

API_KEY = "AIzaSyDT_jDTNaAXFC-v9SCD9ZiE-WzTV5XBhS4" #must make a new one for the company
CHANNEL_ID = "UCxU2cUr78h5bb8xfq60_T1A"  # This is @tccthailand

def get_uploads_playlist_id():
    url = f"https://www.googleapis.com/youtube/v3/channels?part=contentDetails&id={CHANNEL_ID}&key={API_KEY}"
    res = requests.get(url)
    
    try:
        data = res.json()
    except Exception as e:
        print("❌ Failed to decode JSON from response")
        print("Raw response:", res.text)
        raise e

    print("🔍 API response:", json.dumps(data, indent=2, ensure_ascii=False))  # This is what we need

    if "items" not in data or not data["items"]:
        raise ValueError("❌ API call failed. Check API key and channel ID.")

    return data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]



def fetch_videos(playlist_id):
    videos = []
    next_page = None

    while True:
        url = (
            f"https://www.googleapis.com/youtube/v3/playlistItems?"
            f"part=snippet&playlistId={playlist_id}&maxResults=50&key={API_KEY}"
        )
        if next_page:
            url += f"&pageToken={next_page}"

        res = requests.get(url).json()
        for item in res.get("items", []):
            snippet = item["snippet"]
            video_id = snippet["resourceId"]["videoId"]
            title = snippet["title"]
            description = snippet.get("description", "")
            videos.append({
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "description": description,
                "source": "video"
            })

        next_page = res.get("nextPageToken")
        if not next_page:
            break

    return videos

if __name__ == "__main__":
    playlist_id = get_uploads_playlist_id()
    video_data = fetch_videos(playlist_id)
    with open("videos.json", "w", encoding="utf-8") as f:
        json.dump(video_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(video_data)} videos to videos.json")
