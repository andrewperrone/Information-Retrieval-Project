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
   - **NEW: Skips books that are too short (< 20,000 characters)**
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

# --- DENYLIST ---
# This list matches the logic in corpus_cleaner.py
DENYLIST = [
    # Collections / Compilations
    "complete works", "collected works", "compilation", "anthology",
    "short stories", "best russian short stories", "tales", "fables",
    "reader", "works of", "series",
    
    # Massive/Complete Editions
    "complete", "volume", "vol.", "books 1",
    
    # Reference / Non-Narrative
    "dictionary", "thesaurus", "encyclopaedia", "encyclopedia", "factbook",
    "index of", "handbook", "manual", "guide", "quotations", "atlas",
    "grammar", "roget", "webster", "digest", "roll of", "record",
    "register", "yearbook", "report", "census", "gazetteer",
    
    # History / Biography / Philosophy / Science
    "memoirs", "biography", "autobiography", "life of", "history of",
    "chronicle", "letters of", "essays", "treatise", "dialogues",
    "discourses", "commentaries", "diary", "journal", "lives of",
    "philosophy", "psychology", "science", "theory", "principles",
    "inquiry", "study of", "narrative of",
    
    # Specific Failures
    "radiolaria", "what is art", "systematic", "botany", "zoology", 
    "language of flowers", "crimes of", "preachers", "commentary", "prayers",
    
    # Epics / Religious
    "mahabharata", "ramayana", "bible", "testament", "psalms",
    "sermons", "divine comedy", "nibelungenlied",

    # Extra
    "illustrations", # Catches "The Tenniel Illustrations..."
    "index",         # Catches "Index of..." files
    "readme"        # Catches Audio Book readmes
]

def strip_gutenberg_headers(text):
    """
    Removes Project Gutenberg header and footer from the text content.
    
    Args:
        text (str): The raw text content from a Gutenberg book
        
    Returns:
        str: Cleaned text with Gutenberg headers and footers removed
        
    This function specifically targets the standard Gutenberg header/footer format:
    - Header: *** START OF THIS PROJECT GUTENBERG EBOOK ... ***
    - Footer: *** END OF THIS PROJECT GUTENBERG EBOOK ... ***
    
    The function preserves only the actual book content between these markers.
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
    Creates a requests Session with automatic retry logic for failed requests.
    
    Returns:
        requests.Session: A configured session object with retry capabilities
        
    This function sets up a session that will automatically retry failed requests
    with exponential backoff. It's for handling server rate limits such as 
    (HTTP 429) and server errors (5xx). The session will:
    - Retry up to 5 times
    - Use exponential backoff starting at 1 second
    - Retry on status codes: 429, 500, 502, 503, 504
    - Apply to both HTTP and HTTPS requests
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
    Downloads and processes a single book from Project Gutenberg.
    
    Args:
        book_id (int/str): The Gutenberg ID of the book to download
        session (requests.Session): The session object to use for HTTP requests
        
    Returns:
        tuple: (book_title, cleaned_text) if successful, (None, None) on failure
        
    This function:
    1. Fetches book metadata from the Gutenberg API
    2. Tries to download the plain text version first, falls back to HTML if needed
    3. Cleans the text by removing Gutenberg headers/footers
    4. Checks if text is long enough to be a real book
    5. Returns the book title and cleaned text content
    
    The function includes error handling for network issues and invalid responses.
    A small delay (0.5s) is added between requests to be respectful to the server.
    """
    api_url = f"https://gutendex.com/books/{book_id}"
    text_url = None
    html_url = None
    book_title = f"book_{book_id}" 

    try:
        time.sleep(0.5)  # Be nice to the server
        response = session.get(api_url)
        response.raise_for_status()
        book = response.json()
        book_title = book['title']
        
        # Find the best available format (prefer plain text over HTML)
        for mimetype, url in book['formats'].items():
            if 'text/plain' in mimetype and (url.endswith('.txt') or url.endswith('.txt.utf-8')):
                text_url = url
                break
            elif 'text/html' in mimetype:
                html_url = url
        
        clean_text = None
        
        # Try plain text first
        if text_url:
            book_response = session.get(text_url)
            book_response.encoding = 'utf-8-sig'  # Handle Byte Order Mark (BOM) if present
            if book_response.status_code == 200:
                clean_text = strip_gutenberg_headers(book_response.text)
        
        # Fall back to HTML if plain text not available
        elif html_url:
            book_response = session.get(html_url)
            if book_response.status_code == 200:
                soup = BeautifulSoup(book_response.text, 'html.parser')
                clean_text = soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text(separator=' ', strip=True)
        
        # --- LENGTH CHECK ---
        # 20,000 characters is roughly 3,000 words. 
        # Anything shorter is likely a short story collection intro, an index, or a stub.
        MIN_CHAR_LENGTH = 20000 
        
        if clean_text and len(clean_text) < MIN_CHAR_LENGTH:
            print(f"  Skipping ID {book_id}: Text too short ({len(clean_text)} chars). Likely a stub/index.")
            return None, None
        # --------------------

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
    Saves the book text to a file with a sanitized filename.
    
    Args:
        book_id (int/str): The Gutenberg ID of the book
        title (str): The book's title
        text (str): The cleaned text content to save
        directory (str): The directory to save the file in (default: "gutenberg_corpus")
        
    Returns:
        bool: True if save was successful, False otherwise
        
    The function:
    1. Creates a filesystem-safe filename from the book ID and title
    2. Handles edge cases like empty titles
    3. Limits the filename length to prevent filesystem issues
    4. Saves the text with UTF-8 encoding
    5. Returns success/failure status
    """
    # Create a safe filename by removing special characters and normalizing spaces
    safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title).strip().replace(' ', '_')
    if not safe_title:
        safe_title = "unknown_title"
    
    # Combine ID and title, ensuring reasonable length
    safe_filename = f"{book_id}_{safe_title}"
    safe_filename = (safe_filename[:150] + ".txt")  # Truncate if too long
    
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
    # Configuration: Set target number of books and output directory
    TARGET_BOOK_COUNT = 1000
    SAVE_DIRECTORY = "gutenberg_corpus"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        print(f"Created directory: {SAVE_DIRECTORY}")
    
    # Scan for existing downloaded books to enable resuming
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
    
    # Initialize API endpoint with popular fiction books
    next_page_url = "https://gutendex.com/books?sort=popular&bookshelf=Fiction"
    print(f"Starting download process. Goal: {TARGET_BOOK_COUNT} total books.")
    
    # Main download loop: Continue until target count is reached or no more pages
    while success_count < TARGET_BOOK_COUNT and next_page_url:
        print(f"Fetching next page of results: {next_page_url}")
        
        # Fetch and parse the current page of results with retry logic
        data = None
        page_retries = 0
        MAX_PAGE_RETRIES = 5
        
        # Retry logic for fetching the current page
        while page_retries < MAX_PAGE_RETRIES:
            try:
                page_response = session.get(next_page_url)
                page_response.raise_for_status()
                data = page_response.json()
                break 
            except requests.exceptions.RequestException as e:
                print(f"  Error fetching page (Attempt {page_retries+1}/{MAX_PAGE_RETRIES}): {e}")
                if "429" in str(e):
                    print("  >>> Rate Limit Hit (429). Sleeping for 60 seconds to cool down...")
                    time.sleep(60)
                else:
                    time.sleep(5)
                page_retries += 1
        
        # Exit if we couldn't fetch the page after retries
        if not data:
            print("Critical Error: Could not fetch page after multiple retries. Saving progress and stopping.")
            break
        
        # Get the next page URL for pagination
        next_page_url = data.get('next')
        if not next_page_url:
            print("--- Reached the last page of results ---")
        
        # Process each book in the current page of results
        for book in data['results']:
            book_id_str = str(book['id'])
            book_title_lower = book['title'].lower()
            
            # Skip already downloaded books
            if book_id_str in existing_ids:
                continue 
            
            # Skip books matching denylist criteria
            if any(keyword in book_title_lower for keyword in DENYLIST):
                print(f"  Skipping ID {book_id_str}: Title '{book['title']}' matches denylist.")
                continue
                
            # Skip non-English books
            if 'en' not in book['languages']:
                continue

            # Download and process the current book
            print(f"Attempting download for ID {book_id_str}...")
            title, text = download_and_clean_book(book['id'], session)
            
            # Save the book if download was successful
            if title and text:
                if save_book(book['id'], title, text, SAVE_DIRECTORY):
                    success_count += 1
                    existing_ids.add(book_id_str)
                    print(f"  Success ({success_count}/{TARGET_BOOK_COUNT}): Saved '{title}' (ID: {book_id_str})")
                else:
                    fail_count += 1
            else:
                fail_count += 1
            
            # Exit if we've reached our target count
            if success_count >= TARGET_BOOK_COUNT:
                print("Download target reached!")
                break
        
        # Clean exit if target count is reached
        if success_count >= TARGET_BOOK_COUNT:
            next_page_url = None 
    
    # Print final statistics
    print("\n--- Download Complete! ---")
    print(f"Successfully downloaded: {success_count} books")
    print(f"Failed or skipped:     {fail_count} books")
    print(f"All files are saved in the '{SAVE_DIRECTORY}' folder.")