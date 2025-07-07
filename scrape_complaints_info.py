from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

# Your public folder ID
FOLDER_ID = '1k6EzRF0DpXDS5_iJN4nHWR2JrmPmVQi9'

# Path to service account key
SERVICE_ACCOUNT_FILE = 'service-account.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_service():
    print("🔑 Authenticating with Google Drive service account...")
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    print("✅ Authentication successful.")
    return build('drive', 'v3', credentials=creds)

def list_public_png_links(service, folder_id):
    print(f"\n📂 Searching for PNG files in folder: {folder_id}")
    query = f"'{folder_id}' in parents and mimeType='image/png' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    print(f"🔍 Found {len(files)} PNG file(s).")
    data = []

    for i, file in enumerate(files, start=1):
        view_link = f"https://drive.google.com/file/d/{file['id']}/view"
        data.append({
            "title": file['name'],
            "link": view_link
        })
        print(f"📝 [{i}/{len(files)}] Embedded: {file['name']}")

    return data

def main():
    service = get_service()
    links = list_public_png_links(service, FOLDER_ID)

    output_file = "infographics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved {len(links)} links to '{output_file}'.")

if __name__ == '__main__':
    main()
