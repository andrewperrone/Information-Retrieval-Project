"""
Test Set Downloader

This helper script downloads the specific books required for the evaluation
(Alice in Wonderland, Dracula, etc.) to ensure the test cases actually work.
"""
import requests
import os
import re

# Specific Gutenberg IDs for the books in test_cases.json
# 11: Alice in Wonderland
# 345: Dracula
# 1342: Pride and Prejudice
# 84: Frankenstein
# 67098: Winnie the Pooh (Public Domain version)
# 2701: Moby Dick (You likely already have this)
TARGET_IDS = [11, 345, 1342, 84, 67098, 2701]

SAVE_DIR = "gutenberg_corpus"

def download_specific_books():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    for book_id in TARGET_IDS:
        print(f"Fetching metadata for ID {book_id}...")
        try:
            # 1. Get Metadata
            api_url = f"https://gutendex.com/books/{book_id}"
            meta = requests.get(api_url).json()
            title = meta['title']
            
            # 2. Find Text Link
            text_url = None
            for mime, url in meta['formats'].items():
                if 'text/plain' in mime:
                    text_url = url
                    break
            
            if not text_url:
                print(f"  [Skipped] No text format found for {title}")
                continue
                
            # 3. Download Text
            print(f"  Downloading '{title}'...")
            text_resp = requests.get(text_url)
            text_resp.encoding = 'utf-8-sig' # Fix encoding
            
            # 4. Save
            safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title).strip().replace(' ', '_')
            filename = f"{book_id}_{safe_title}.txt"
            path = os.path.join(SAVE_DIR, filename)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text_resp.text)
                
            print(f"  [Success] Saved to {filename}")
            
        except Exception as e:
            print(f"  [Error] Failed on ID {book_id}: {e}")

if __name__ == "__main__":
    download_specific_books()