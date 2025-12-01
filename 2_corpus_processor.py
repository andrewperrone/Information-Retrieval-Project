"""
Gutenberg Corpus Processor

This script processes raw text files from Project Gutenberg by tokenizing and normalizing the text.
It creates a searchable corpus that can be saved/loaded for efficient access in subsequent runs.

Action:
It reads books from the local directory, tokenizes them into words, converts them to lowercase, 
and filters out non-alphabetic tokens (numbers/punctuation).

Connection:
This script generates 'processed_corpus.pkl', which is the required input data for the 
indexer script (indexer.py) to build the search index.

Inputs:
- Directory containing .txt files (default: "gutenberg_corpus")
- NLTK data (automatically downloaded if missing: 'punkt')

Outputs:
- Processed corpus as a Python dictionary: {doc_id: [list_of_tokens]}
- Saves corpus as a pickle file (default: "processed_corpus.pkl")
- Console logs of processing progress and statistics

Process:
1. Sets up required NLTK data (punkt tokenizer).
2. Checks configuration flag FORCE_REBUILD.
3. If FORCE_REBUILD is False and a saved file exists, it loads the data from disk.
4. If FORCE_REBUILD is True or no file exists:
   - Scans directory for .txt files.
   - For each file:
     - Reads the raw text.
     - Runs the processing pipeline (tokenize -> lowercase -> alpha-only).
     - Stores processed tokens with the document ID.
   - Saves the resulting dictionary to a pickle file for future use.
5. Displays validation output (sample tokens) to confirm success.
"""

import nltk
import os
import glob
from nltk.tokenize import word_tokenize
import time
import pickle

# --- CONFIGURATION ---
# Directory where raw text files are stored
CORPUS_DIRECTORY = "gutenberg_corpus"
# Filename for the saved pickle output
CORPUS_SAVE_FILE = "processed_corpus.pkl"
# Set to True to ignore existing pickle file and re-process all texts
FORCE_REBUILD = True  
# ---------------------

def setup_nltk():
    """
    Checks for and downloads the necessary NLTK data packages.
    Specifically ensures the 'punkt' tokenizer is available.
    """
    print("Checking NLTK data...")
    try:
        # Check if the tokenizer is already locally available
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # If not found, download it
        print("Downloading punkt tokenizer...")
        nltk.download('punkt')
    print("NLTK setup complete.")

def process_text_pipeline(raw_text):
    """
    Runs a single string of raw text through the cleaning pipeline.
    
    Steps:
    1. Tokenize (split string into words based on linguistic rules).
    2. Lowercase (normalization).
    3. Filter (keep only alphabetic tokens, removing numbers and punctuation).
    
    Args:
        raw_text (str): The full text content of a book.
        
    Returns:
        list: A list of clean string tokens.
    """
    # 1. Tokenize (split into words)
    tokens = word_tokenize(raw_text)
    processed_tokens = []
    
    for token in tokens:
        # 2. Normalize (lowercase)
        token = token.lower()
        
        # 3. Filter (Alphabetic only - removes punctuation & numbers)
        if token.isalpha():
            processed_tokens.append(token)
            
    return processed_tokens

def build_corpus_from_files(corpus_dir):
    """
    Iterates through all .txt files in the specified directory, processes them,
    and builds a dictionary mapping document IDs to token lists.
    
    Args:
        corpus_dir (str): Path to the folder containing .txt files.
        
    Returns:
        dict: { 'filename.txt': ['word1', 'word2', ...] } or None on error.
    """
    print(f"Starting to build corpus from directory: {corpus_dir}")
    processed_corpus = {}
    
    # Locate all text files in the directory
    file_paths = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    if not file_paths:
        print(f"Error: No .txt files found in '{corpus_dir}'.")
        return None
        
    print(f"Found {len(file_paths)} .txt files. Starting processing...")
    start_time = time.time()
    
    # Process files one by one
    for i, filepath in enumerate(file_paths):
        # Extract filename to use as the unique Document ID
        doc_id = os.path.basename(filepath)
        try:
            # Read the raw text content
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Apply the cleaning pipeline
            processed_tokens = process_text_pipeline(raw_text)
            
            # Store result in the dictionary
            processed_corpus[doc_id] = processed_tokens
            
            # Progress log every 100 files
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(file_paths)} files...")

        except Exception as e:
            # Catch file read errors without stopping the whole process
            print(f"Warning: Error processing file {doc_id} ({e}). Skipping.")
            
    end_time = time.time()
    print(f"\n--- Corpus Processing Complete ---")
    print(f"Successfully processed {len(processed_corpus)} documents.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    return processed_corpus

if __name__ == "__main__":
    # Ensure dependencies are ready
    setup_nltk()
    
    corpus = None
    
    # 1. ATTEMPT LOAD: Check if we can skip processing
    # We only load if FORCE_REBUILD is False AND the file exists
    if not FORCE_REBUILD and os.path.exists(CORPUS_SAVE_FILE):
        print(f"Found saved corpus file: {CORPUS_SAVE_FILE}")
        print("Loading corpus from disk... (This is fast)")
        try:
            with open(CORPUS_SAVE_FILE, 'rb') as f:
                corpus = pickle.load(f)
            print("Corpus loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load {CORPUS_SAVE_FILE} ({e}). Re-building corpus.")

    # 2. BUILD: Run if we couldn't load or if FORCE_REBUILD is True
    if corpus is None:
        if FORCE_REBUILD:
            print("Force Rebuild is ON. Ignoring existing pickle files.")
        else:
            print("No saved corpus found.")
            
        print("Building from files... (This is slow)")
        corpus = build_corpus_from_files(CORPUS_DIRECTORY)
        
        # Save the newly built corpus to disk
        if corpus:
            print(f"Saving processed corpus to {CORPUS_SAVE_FILE}...")
            try:
                with open(CORPUS_SAVE_FILE, 'wb') as f:
                    pickle.dump(corpus, f)
                print("Corpus saved successfully.")
            except Exception as e:
                print(f"Error: Could not save corpus to {CORPUS_SAVE_FILE} ({e})")
    
    # 3. VALIDATION: Print a sample to verify correct processing
    if corpus:
        print("\n--- Example: Processing Results ---")
        # Grab the first document key available
        example_doc_id = list(corpus.keys())[0]
        example_tokens = corpus[example_doc_id]
        
        print(f"Document ID: {example_doc_id}")
        print(f"Total processed tokens: {len(example_tokens)}")
        print(f"First 20 processed tokens: {example_tokens[:20]}")
    else:
        print("No corpus was built or loaded.")