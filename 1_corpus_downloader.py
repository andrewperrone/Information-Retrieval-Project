"""
Gutenberg Corpus Downloader

This script downloads and processes public domain books from Project Gutenberg's collection.
It fetches books using the Gutenberg API, cleans the text by removing headers/footers,
and saves them as individual text files for further processing.

Inputs:
- Target book count (hardcoded, default=1000)
- Save directory (hardcoded, default="gutenberg_corpus")
- Gutenberg API endpoints (hardcoded)

Outputs:
- Text files in the specified directory, named as "{book_id}_{cleaned_title}.txt"
- Console logs of download progress and any errors encountered

Process:
1. Creates a retry-enabled HTTP session for robust downloads
2. Checks for and loads existing book files to resume partial downloads
3. Fetches book metadata in pages from the Gutenberg API
4. For each book:
   - Downloads the text content (preferring plain text, falling back to HTML)
   - Cleans the text by removing Gutenberg headers/footers
   - Skips books matching denylist criteria (collections, references, etc.)
   - Saves cleaned text to a file with a sanitized filename
5. Continues until target book count is reached or no more books available
6. Handles rate limiting and network errors with retries
"""

import requests
import re
import time
import os
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def strip_gutenberg_headers(text):
    """
    Attempts to strip the Project Gutenberg header and footer from a plain text file.
    """
    start_match = re.search(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if start_match:
        text = text[start_match.end():]
    
    end_match = re.search(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if end_match:
        text = text[:end_match.start()]
        
    return text.strip()

def create_retry_session():
    """
    Creates a requests.Session() object with automatic retries
    for temporary server errors.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1, 
        # Added 429 to the list of codes to retry on
        status_forcelist=[429, 500, 502, 503, 504], 
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def download_and_clean_book(book_id, session):
    """
    Downloads and cleans a single book given its ID.
    """
    api_url = f"https://gutendex.com/books/{book_id}"
    text_url = None
    html_url = None
    book_title = f"book_{book_id}" 

    try:
        # 1. Get Book Metadata
        # Small sleep to be polite
        time.sleep(0.5) 
        response = session.get(api_url)
        response.raise_for_status()
        book = response.json()
        book_title = book['title']
        
        # 2. Triage formats
        for mimetype, url in book['formats'].items():
            if 'text/plain' in mimetype and (url.endswith('.txt') or url.endswith('.txt.utf-8')):
                text_url = url
                break
            elif 'text/html' in mimetype:
                html_url = url
        
        # 3. Download and Parse
        clean_text = None
        
        if text_url:
            book_response = session.get(text_url)
            book_response.encoding = 'utf-8-sig'
            if book_response.status_code == 200:
                clean_text = strip_gutenberg_headers(book_response.text)
        
        elif html_url:
            book_response = session.get(html_url)
            if book_response.status_code == 200:
                soup = BeautifulSoup(book_response.text, 'html.parser')
                clean_text = soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text(separator=' ', strip=True)
        
        if clean_text:
            return book_title, clean_text
        else:
            print(f"  Warning: No usable text found for ID {book_id}")
            return book_title, None

    except requests.exceptions.RequestException as e:
        print(f"  Error: Failed to download metadata/text for ID {book_id}: {e}")
        return None, None

def save_book(book_id, title, text, directory="gutenberg_corpus"):
    """
    Saves the text to a file with a unique, clean filename.
    """
    safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title).strip().replace(' ', '_')
    if not safe_title:
        safe_title = "unknown_title"
    
    safe_filename = f"{book_id}_{safe_title}"
    safe_filename = (safe_filename[:150] + ".txt")
    
    filepath = os.path.join(directory, safe_filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except IOError as e:
        print(f"  Error: Could not save file {filepath}: {e}")
        return False

# --- Main Controller ---
if __name__ == "__main__":
    
    TARGET_BOOK_COUNT = 1000
    SAVE_DIRECTORY = "gutenberg_corpus"
    
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        print(f"Created directory: {SAVE_DIRECTORY}")
        
    # --- Check for existing files ---
    print(f"Checking for existing files in {SAVE_DIRECTORY}...")
    try:
        existing_files = os.listdir(SAVE_DIRECTORY)
        existing_ids = set()
        for f in existing_files:
            if f.endswith('.txt') and '_' in f:
                book_id = f.split('_')[0]
                if book_id.isdigit():
                    existing_ids.add(book_id)
        
        success_count = len(existing_ids)
        print(f"Found {success_count} existing books.")
    except Exception as e:
        print(f"Warning: Could not list existing files. Assuming 0. Error: {e}")
        existing_ids = set()
        success_count = 0
    
    fail_count = 0
    session = create_retry_session()
    
    next_page_url = "https://gutendex.com/books?sort=popular&bookshelf=Fiction"
    
    print(f"Starting download process. Goal: {TARGET_BOOK_COUNT} total books.")
    
    while success_count < TARGET_BOOK_COUNT and next_page_url:
        print(f"Fetching next page of results: {next_page_url}")
        
        # --- Page Fetch Retry Loop ---
        data = None
        page_retries = 0
        MAX_PAGE_RETRIES = 5
        
        while page_retries < MAX_PAGE_RETRIES:
            try:
                page_response = session.get(next_page_url)
                page_response.raise_for_status()
                data = page_response.json()
                break # Success! Exit the retry loop
            except requests.exceptions.RequestException as e:
                print(f"  Error fetching page (Attempt {page_retries+1}/{MAX_PAGE_RETRIES}): {e}")
                
                # If we are rate limited (429), wait a LONG time (60s)
                if "429" in str(e):
                    print("  >>> Rate Limit Hit (429). Sleeping for 60 seconds to cool down...")
                    time.sleep(60)
                else:
                    time.sleep(5)
                page_retries += 1
        
        if not data:
            print("Critical Error: Could not fetch page after multiple retries. Saving progress and stopping.")
            break
        
        # Get the URL for the *next* page
        next_page_url = data.get('next')
        if not next_page_url:
            print("--- Reached the last page of results ---")
            
        # --- Expanded Denylist (Matches corpus_cleaner.py) ---
        TITLE_DENYLIST = [
            # --- Collections / Compilations ---
            "complete_works", "collected_works", "compilation", "anthology",
            "short_stories", "best_russian_short_stories", "tales", "fables",
            "reader", "works_of", "series",
            
            # --- Massive/Complete Editions ---
            "complete", "volume", "vol_", "books_1",
            
            # --- Reference / Non-Narrative ---
            "dictionary", "thesaurus", "encyclopaedia", "encyclopedia", "factbook",
            "index_of", "handbook", "manual", "guide", "quotations", "atlas",
            "grammar", "roget", "webster", "digest", "roll_of", "record",
            "register", "yearbook", "report", "census", "gazetteer",
            
            # --- History / Biography / Philosophy / Science ---
            "memoirs", "biography", "autobiography", "life_of", "history_of",
            "chronicle", "letters_of", "essays", "treatise", "dialogues",
            "discourses", "commentaries", "diary", "journal", "lives_of",
            "philosophy", "psychology", "science", "theory", "principles",
            "inquiry", "study_of", "narrative_of", # Often non-fiction travelogues
            
            # --- Specific Failures we found ---
            "radiolaria", "what_is_art", "systematic", "botany", "zoology",
            
            # --- Epics / Religious ---
            "mahabharata", "ramayana", "bible", "testament", "psalms",
            "sermons", "divine_comedy", "nibelungenlied"
        ]
        
        # Loop through the books on this page
        for book in data['results']:
            book_id_str = str(book['id'])
            book_title_lower = book['title'].lower()
            
            # Skip existing
            if book_id_str in existing_ids:
                continue 
            
            # Skip denylist
            if any(word in book_title_lower for word in TITLE_DENYLIST):
                print(f"  Skipping ID {book_id_str}: Title '{book['title']}' is on denylist.")
                continue
                
            # Skip non-English
            if 'en' not in book['languages']:
                continue

            print(f"Attempting download for ID {book_id_str}...")
            
            title, text = download_and_clean_book(book['id'], session)
            
            if title and text:
                if save_book(book['id'], title, text, SAVE_DIRECTORY):
                    success_count += 1
                    existing_ids.add(book_id_str)
                    print(f"  Success ({success_count}/{TARGET_BOOK_COUNT}): Saved '{title}' (ID: {book_id_str})")
                else:
                    fail_count += 1
            else:
                fail_count += 1
            
            if success_count >= TARGET_BOOK_COUNT:
                print("Download target reached!")
                break
        
        if success_count >= TARGET_BOOK_COUNT:
            next_page_url = None 
    
    print("\n--- Download Complete! ---")
    print(f"Successfully downloaded: {success_count} books")
    print(f"Failed or skipped:     {fail_count} books")
    print(f"All files are saved in the '{SAVE_DIRECTORY}' folder.")