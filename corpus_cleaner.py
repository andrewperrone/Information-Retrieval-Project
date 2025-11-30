"""
Corpus Cleaner for Project Gutenberg Files

This script filters out unwanted documents from a Project Gutenberg corpus by removing files
whose names contain specific keywords. It helps maintain a clean dataset by excluding
collections, references, and non-narrative works that may not be suitable for sentiment analysis.

Inputs:
- Directory containing Project Gutenberg text files (default: "gutenberg_corpus")
- Hardcoded list of keywords for filtering

Outputs:
- Modified file system (deletes matching files)
- Console summary of actions taken (files deleted/kept)

Process:
1. Scans the specified directory for .txt files
2. For each file, checks if its name contains any of the blacklisted keywords
3. Deletes files matching any keyword
4. Generates a summary report of files deleted and kept

The script uses a comprehensive denylist that includes:
- Collections and compilations
- Reference works and non-narrative content
- Historical and biographical works
- Scientific and philosophical texts
- Religious and epic literature
"""

import os

# --- Configuration ---
CORPUS_DIR = "gutenberg_corpus"

# Case-insensitive keywords that indicate a file should be deleted
# These must match the denylist in download_corpus.py to prevent re-download loops
DELETE_KEYWORDS = [
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

def clean_corpus(directory):
    print(f"Scanning '{directory}' for files to remove...")
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    files = os.listdir(directory)
    deleted_count = 0
    kept_count = 0
    
    for filename in files:
        if not filename.endswith(".txt"):
            continue
            
        file_path = os.path.join(directory, filename)
        lower_name = filename.lower()
        
        # Check against keywords
        should_delete = False
        for keyword in DELETE_KEYWORDS:
            if keyword in lower_name:
                should_delete = True
                print(f"Deleting: {filename} (Matched: '{keyword}')")
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error deleting {filename}: {e}")
                break 
        
        if not should_delete:
            kept_count += 1

    print("-" * 30)
    print(f"Cleanup Complete.")
    print(f"Deleted: {deleted_count} files")
    print(f"Remaining: {kept_count} files")

if __name__ == "__main__":
    clean_corpus(CORPUS_DIR)