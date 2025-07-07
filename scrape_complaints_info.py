from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

# 👇 Your public folder ID
FOLDER_ID = '1k6EzRF0DpXDS5_iJN4nHWR2JrmPmVQi9'

# 👇 Path to service account key
SERVICE_ACCOUNT_FILE = 'service-account.json'

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_public_png_links(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='image/png' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    data = []
    for file in files:
        view_link = f"https://drive.google.com/file/d/{file['id']}/view"
        data.append({
            "title": file['name'],
            "link": view_link
        })
    return data

def main():
    service = get_service()
    links = list_public_png_links(service, FOLDER_ID)

    print(json.dumps(links, ensure_ascii=False, indent=2))

    with open("infographics.json", "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
