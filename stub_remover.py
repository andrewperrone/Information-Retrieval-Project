"""
Gutenberg Corpus Stub Remover

This script cleans the downloaded corpus by removing "stub" files—text files that are too small 
to be full novels (e.g., indexes, short readmes, image caption lists). This ensures that the 
analysis pipeline focuses only on substantial narrative content.

Action:
It scans the corpus directory and deletes any text file smaller than a specified threshold 
(default: 20KB).

Connection:
This is a maintenance utility. It cleans the 'gutenberg_corpus' directory, ensuring that 
downstream scripts (corpus_processor.py, book_level_emotion_analyzer.py) do not process 
garbage data or metadata files that often skew statistics with low word counts.

Inputs:
- Directory containing .txt files (default: "gutenberg_corpus")
- Minimum file size in KB (default: 20KB)

Outputs:
- Deletes invalid files directly from the filesystem
- Console logs of every deleted file
- Final summary statistics (deleted count vs. remaining count)

Process:
1. Scans the target directory for all .txt files.
2. For each file, checks the file size in bytes using the OS filesystem.
3. Compares the size against the threshold (20KB * 1024 bytes).
4. If the file is too small, it is deleted immediately.
5. Prints a summary report of the cleaning operation.
"""

import os

# --- Configuration ---
CORPUS_DIR = "gutenberg_corpus"
MIN_FILE_SIZE_KB = 20  # Files smaller than 20KB are likely stubs/indexes
# ---------------------

def remove_stubs(directory, min_kb):
    print(f"Scanning '{directory}' for files smaller than {min_kb}KB...")
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    files = os.listdir(directory)
    deleted_count = 0
    kept_count = 0
    
    min_bytes = min_kb * 1024
    
    for filename in files:
        if not filename.endswith(".txt"):
            continue
            
        file_path = os.path.join(directory, filename)
        
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            
            if file_size < min_bytes:
                print(f"Deleting STUB: {filename} ({file_size/1024:.1f} KB)")
                os.remove(file_path)
                deleted_count += 1
            else:
                kept_count += 1
                
        except Exception as e:
            print(f"Error checking {filename}: {e}")

    print("-" * 30)
    print(f"Stub Removal Complete.")
    print(f"Deleted: {deleted_count} files (Under {min_kb}KB)")
    print(f"Remaining: {kept_count} files")

if __name__ == "__main__":
    remove_stubs(CORPUS_DIR, MIN_FILE_SIZE_KB)