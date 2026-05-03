"""
setup_youtube.py — Guided Setup Wizard for YouTube Shorts Automation.
"""
import os
import sys
import time

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def wizard():
    clear()
    print("====================================================")
    print("🚀 YOUTUBE SHORTS AUTOMATION SETUP WIZARD")
    print("====================================================")
    print("\nThis wizard will help you link your YouTube channel:")
    print("Channel: cricketwithprajjwal@gmail.com")
    print("\n----------------------------------------------------")
    print("STEP 1: Create your Google Cloud Project")
    print("----------------------------------------------------")
    print("1. Go to: https://console.cloud.google.com/")
    print("2. Click 'Select a project' > 'New Project'. Name it 'yt-clips'.")
    print("3. In the search bar at the top, search for 'YouTube Data API v3' and click ENABLE.")
    
    input("\nPress ENTER once you have enabled the API...")
    
    clear()
    print("----------------------------------------------------")
    print("STEP 2: Create Credentials")
    print("----------------------------------------------------")
    print("1. Go to 'APIs & Services' > 'OAuth consent screen'.")
    print("2. Choose 'External' and click CREATE.")
    print("3. Fill in 'App name' (yt-clips) and 'User support email' (your email).")
    print("4. Add your email (cricketwithprajjwal@gmail.com) as a 'Test User'.")
    print("5. Go to 'Credentials' > 'Create Credentials' > 'OAuth client ID'.")
    print("6. Choose 'Desktop App' and click CREATE.")
    print("7. Click DOWNLOAD JSON on the right side of the ID you just created.")
    
    input("\nPress ENTER once you have downloaded the JSON file...")
    
    clear()
    print("----------------------------------------------------")
    print("STEP 3: Link the File")
    print("----------------------------------------------------")
    print(f"Please move the downloaded file into this folder: {os.getcwd()}")
    print("And rename it to: client_secrets.json")
    
    while not os.path.exists('client_secrets.json'):
        print("\nChecking... ❌ File 'client_secrets.json' not found yet.")
        print("Waiting 10 seconds... (Or press Ctrl+C to stop)")
        time.sleep(10)
    
    print("\n✅ Found client_secrets.json!")
    print("\n----------------------------------------------------")
    print("STEP 4: Authenticate with YouTube")
    print("----------------------------------------------------")
    print("A browser window will now open. Please log in with:")
    print("👉 cricketwithprajjwal@gmail.com")
    print("\nIf you see a 'Google hasn't verified this app' warning, click 'Advanced' > 'Go to yt-clips (unsafe)'.")
    
    input("\nPress ENTER to open the browser and login...")
    
    try:
        from upload import get_authenticated_service
        service = get_authenticated_service()
        if service:
            print("\n🎉 SUCCESS! Your YouTube channel is now linked.")
            print("A 'yt_token.json' has been created. You are ready for Version 3 autonomy!")
    except Exception as e:
        print(f"\n❌ Error during login: {e}")

if __name__ == "__main__":
    wizard()
