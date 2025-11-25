"""
Corpus File Lister

This script scans a specified directory (default: 'gutenberg_corpus') and generates a text file
containing the names of all files found, one per line. Useful for creating an inventory of files
in the corpus for reference or further processing.

Inputs:
- Directory to scan (configurable via TARGET_DIRECTORY, default: 'gutenberg_corpus')
- Output filename (configurable via OUTPUT_FILE, default: 'corpus_file_list.txt')

Outputs:
- Text file listing all filenames from the target directory
- Console output showing scan progress and results

Process:
1. Verifies the existence of the target directory
2. Retrieves a list of all files in the directory
3. Writes each filename to the output file, one per line
4. Provides status updates and success/error messages in the console
"""

import os

# --- Configuration ---
# 1. Set the directory you want to scan
TARGET_DIRECTORY = "gutenberg_corpus" 

# 2. Set the name of the output file
OUTPUT_FILE = "corpus_file_list.txt"
# ---------------------

def compile_filenames(directory, output_file):
    """
    Scans a target directory, gets all filenames, and writes them
    to a new text file, one per line.
    """
    print(f"Scanning directory: {directory}...")
    
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory not found: '{directory}'")
        print("Please make sure this script is in the same folder as your corpus directory.")
        return

    try:
        # 1. Get a list of all filenames in the directory
        filenames = os.listdir(directory)
        
        # Optional: Filter for only .txt files
        # filenames = [f for f in filenames if f.endswith('.txt')]
        
        print(f"Found {len(filenames)} files. Writing to {output_file}...")

        # 2. Open the output file in 'write' mode
        with open(output_file, 'w', encoding='utf-8') as f:
            # 3. Loop and write each filename
            for name in filenames:
                f.write(name + '\n')
        
        print("\n--- Success! ---")
        print(f"All filenames have been saved to {output_file}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

# --- Main execution ---
if __name__ == "__main__":
    compile_filenames(TARGET_DIRECTORY, OUTPUT_FILE)