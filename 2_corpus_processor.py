"""
Gutenberg Corpus Processor

This script processes raw text files from Project Gutenberg by tokenizing and normalizing the text.
It creates a searchable corpus that can be saved/loaded for efficient access in subsequent runs.

Inputs:
- Directory containing .txt files (default: "gutenberg_corpus")
- NLTK data (automatically downloaded if missing)

Outputs:
- Processed corpus as a Python dictionary: {doc_id: [tokens]}
- Saves corpus as a pickle file (default: "processed_corpus.pkl")
- Console logs of processing progress and statistics

Process:
1. Sets up required NLTK data (punkt tokenizer)
2. Checks for existing processed corpus to load (saves time on subsequent runs)
3. If no saved corpus exists:
   - Scans directory for .txt files
   - For each file:
     - Reads the raw text
     - Tokenizes and normalizes the text (lowercase, alphabetic only)
     - Stores processed tokens with document ID
4. Optionally saves the processed corpus as a pickle file
5. Displays example processing results
"""

import nltk
import os
import glob
import time
import pickle

# --- Configuration ---
CORPUS_DIRECTORY = "gutenberg_corpus"
CORPUS_SAVE_FILE = "processed_corpus.pkl"

# New Settings for Chunking
CHUNK_SIZE = 1000      # Words per segment
OVERLAP = 100          # Words of overlap to maintain context

def setup_nltk():
    """
    Downloads the necessary NLTK data packages if not already present.
    """
    print("Checking NLTK data...")
    try:
        # Check for the punkt tokenizer data
        nltk.data.find('tokenizers/punkt')
        # Some newer versions of NLTK also require 'punkt_tab'
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            pass 
            
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt')
        try:
            nltk.download('punkt_tab')
        except:
            pass
    print("NLTK setup complete.")

def create_chunks(tokens, chunk_size, overlap):
    """
    Splits a list of tokens into overlapping chunks.
    """
    if len(tokens) <= chunk_size:
        return [tokens]
    
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
        # Prevent creating tiny chunks at the very end
        if i + chunk_size >= len(tokens):
            break
            
    return chunks

def build_corpus_from_files(corpus_dir):
    """
    Reads all .txt files, splits them into chunks (segments),
    and returns a dictionary of {doc_id_chunk: [tokens]}.
    """
    print(f"Starting to build Segment-Level corpus from: {corpus_dir}")
    
    processed_corpus = {}
    file_paths = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    if not file_paths:
        print(f"Error: No .txt files found in '{corpus_dir}'.")
        return None
        
    print(f"Found {len(file_paths)} .txt files. Processing into chunks...")
    start_time = time.time()
    
    for i, filepath in enumerate(file_paths):
        # Extract base ID (e.g. "2701" from "2701_Moby_Dick.txt")
        filename = os.path.basename(filepath)
        book_id = filename.split('_')[0] 
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # 1. Tokenize & Normalize
            # Uses the safe nltk.word_tokenize import
            tokens = nltk.word_tokenize(raw_text)
            
            # Keep alpha words AND negation contraction "n't"
            clean_tokens = [t.lower() for t in tokens if t.isalpha() or t == "n't"]
            
            # 2. Create Segments (Chunks)
            chunks = create_chunks(clean_tokens, CHUNK_SIZE, OVERLAP)
            
            # 3. Store each chunk as a unique "Document"
            for chunk_idx, chunk_tokens in enumerate(chunks):
                # New ID Format: "BookID_ChunkNum" (e.g., "2701_0", "2701_1")
                unique_id = f"{book_id}_{chunk_idx}"
                processed_corpus[unique_id] = chunk_tokens
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1} books -> {len(processed_corpus)} total segments...")

        except Exception as e:
            print(f"Warning: Error processing file {filename} ({e}). Skipping.")
            
    end_time = time.time()
    print(f"\n--- Corpus Processing Complete ---")
    print(f"Original Books: {len(file_paths)}")
    print(f"Total Segments Created: {len(processed_corpus)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    return processed_corpus

# --- Main execution ---
if __name__ == "__main__":
    setup_nltk()
    
    corpus = None
    
    # Check if exists #TODO: Review code, remove if unneeded
    if os.path.exists(CORPUS_SAVE_FILE):
        print(f"Found saved corpus: {CORPUS_SAVE_FILE}")
        print("Re-building corpus to ensure Chunking is applied...")
        corpus = None

    if corpus is None:
        print("Building corpus from files...")
        corpus = build_corpus_from_files(CORPUS_DIRECTORY)
        
        if corpus:
            print(f"Saving processed corpus to {CORPUS_SAVE_FILE}...")
            with open(CORPUS_SAVE_FILE, 'wb') as f:
                pickle.dump(corpus, f)
            print("Corpus saved successfully.")
    
    # Verification
    if corpus:
        print("\n--- Example: Processing Results ---")
        first_id = list(corpus.keys())[0]
        print(f"Segment ID: {first_id}")
        print(f"Token Count: {len(corpus[first_id])}")
        print(f"Sample: {corpus[first_id][:10]}...")