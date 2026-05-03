"""
fetch_youtube_transcript.py — Official YouTube API Transcript Downloader.
Uses your channel owner permissions to get the highest quality transcripts.
"""
import os
import json
import argparse
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import io

# We need force-ssl or youtubepartner to download the actual caption content
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

def get_service():
    creds = None
    if os.path.exists('yt_token.json'):
        # Note: We may need to re-auth if the existing token doesn't have the force-ssl scope
        creds = Credentials.from_authorized_user_file('yt_token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except:
                creds = None
        
        if not creds:
            if not os.path.exists('client_secrets.json'):
                print("❌ Error: client_secrets.json not found.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open('yt_token.json', 'w') as token:
            token.write(creds.to_json())

    return build('youtube', 'v3', credentials=creds)

def extract_id(url):
    import re
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else url

def fetch_transcript(video_url, output_path=None, preferred_langs=['hi', 'en']):
    video_id = extract_id(video_url)
    youtube = get_service()
    if not youtube: return False

    print(f"🔍 Checking captions for video: {video_id}...")

    # 1. List all caption tracks
    request = youtube.captions().list(part="snippet", videoId=video_id)
    response = request.execute()
    items = response.get("items", [])

    if not items:
        print("❌ No caption tracks found for this video.")
        return False

    # 2. Smart Selection Logic
    # We want: Manual Hindi > Auto Hindi > Manual English > Auto English > Anything else
    target_track = None
    
    # Debug print all found tracks
    for item in items:
        snip = item["snippet"]
        print(f"   found: {snip['language']} (type: {snip['trackKind']})")

    # Try to find the best match based on our priority list
    for lang in preferred_langs:
        # First pass: Manual tracks only
        for item in items:
            if item["snippet"]["language"].startswith(lang) and item["snippet"]["trackKind"] == "standard":
                target_track = item
                break
        if target_track: break
        
        # Second pass: Auto-generated tracks
        for item in items:
            if item["snippet"]["language"].startswith(lang):
                target_track = item
                break
        if target_track: break

    if not target_track:
        print("⚠️ No preferred language found. Using the first available track...")
        target_track = items[0]

    track_id = target_track["id"]
    track_lang = target_track["snippet"]["language"]
    track_kind = target_track["snippet"]["trackKind"]
    print(f"📥 Downloading: {track_lang} (type: {track_kind}, id: {track_id})...")

    # 3. Download the caption content in SRT format
    request = youtube.captions().download(id=track_id, tfmt="srt")
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    
    srt_content = fh.getvalue().decode('utf-8')

    # 4. Convert SRT to our Pipeline JSON format
    # This is a basic parser for SRT -> JSON [{start, end, text}, ...]
    import re
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    results = []
    
    def time_to_sec(t_str):
        h, m, s = t_str.replace(',', '.').split(':')
        return int(h)*3600 + int(m)*60 + float(s)

    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            times = re.findall(r'(\d+:\d+:\d+,\d+)', lines[1])
            if len(times) == 2:
                text = " ".join(lines[2:]).strip()
                results.append({
                    "start": time_to_sec(times[0]),
                    "end": time_to_sec(times[1]),
                    "text": text
                })

    if not output_path:
        output_path = f"transcripts/{video_id}.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✨ Success! Transcript saved to: {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube Video URL")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--lang", "-l", default="hi,en", help="Preferred languages (comma-separated, e.g. hi,en)")
    args = parser.parse_args()
    
    langs = args.lang.split(",")
    fetch_transcript(args.url, args.output, preferred_langs=langs)
