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
import os         # For file/folder operations
import glob       # For finding all .txt files
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
import time
import pickle     # For saving/loading the corpus

def setup_nltk():
    """
    Downloads the necessary NLTK data packages if not already present.
    """
    print("Checking NLTK data...")
    try:
        # Check if packages are downloaded, if not, download them
        # Stopwords download removed
        nltk.data.find('corpora/stopwords')
    except LookupError:
        # This check ensures necessary NLTK data is downloaded
        pass
        
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt')
    print("NLTK setup complete.")

def process_text_pipeline(raw_text):
    """
    Runs a single string of raw text through the full NLTK processing pipeline.
    Returns a list of processed tokens. (Stemming and Stopword Removal removed)
    """
    # 1. Tokenize (split into words)
    tokens = word_tokenize(raw_text)
    
    processed_tokens = []
    for token in tokens:
        # 2. Normalize (lowercase) and check if alphabetic
        token = token.lower()
        if token.isalpha():
            processed_tokens.append(token)
                
    return processed_tokens

def build_corpus_from_files(corpus_dir):
    """
    Reads all .txt files in a directory, processes them, 
    and returns a dictionary of {doc_id: [tokens]}.
    """
    print(f"Starting to build corpus from directory: {corpus_dir}")
    
    # Store processed data
    processed_corpus = {}
    
    # --- Find all .txt files ---
    file_paths = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    if not file_paths:
        print(f"Error: No .txt files found in '{corpus_dir}'.")
        print("Please make sure you ran the download script first.")
        return None
        
    print(f"Found {len(file_paths)} .txt files. Starting processing...")
    
    start_time = time.time()
    
    # --- Loop, Read, and Process each file ---
    for i, filepath in enumerate(file_paths):
        # The filename is Document ID
        doc_id = os.path.basename(filepath)
        
        # Use a try...except block in case one file is corrupted.
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Run the text through our NLP pipeline
            processed_tokens = process_text_pipeline(raw_text)
            
            # Store the result
            processed_corpus[doc_id] = processed_tokens
            
            # Print a progress update every 100 files
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(file_paths)} files...")

        except UnicodeDecodeError:
            print(f"Warning: Could not read file {doc_id} (UnicodeDecodeError). Skipping.")
        except Exception as e:
            print(f"Warning: Error processing file {doc_id} ({e}). Skipping.")
            
    end_time = time.time()
    print(f"\n--- Corpus Processing Complete ---")
    print(f"Successfully processed {len(processed_corpus)} documents.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    return processed_corpus

# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Download NLTK data (stopwords, punkt)
    setup_nltk()
    
    # 2. Define corpus directory and the new save file path
    CORPUS_DIRECTORY = "gutenberg_corpus"
    CORPUS_SAVE_FILE = "processed_corpus.pkl"  #File to save/load corpus
    
    corpus = None
    
    # --- Check if the processed corpus already exists ---
    if os.path.exists(CORPUS_SAVE_FILE):
        print(f"Found saved corpus file: {CORPUS_SAVE_FILE}")
        print("Loading corpus from disk... (This is fast)")
        try:
            with open(CORPUS_SAVE_FILE, 'rb') as f:
                corpus = pickle.load(f)
            print("Corpus loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load {CORPUS_SAVE_FILE} ({e}). Re-building corpus.")

    # --- If corpus wasn't loaded, build it from scratch ---
    if corpus is None:
        print("No saved corpus found. Building from files... (This is slow)")
        corpus = build_corpus_from_files(CORPUS_DIRECTORY)
        
        # --- Save the newly built corpus ---
        if corpus:
            print(f"Saving processed corpus to {CORPUS_SAVE_FILE}...")
            try:
                with open(CORPUS_SAVE_FILE, 'wb') as f:
                    pickle.dump(corpus, f)
                print("Corpus saved successfully.")
            except Exception as e:
                print(f"Error: Could not save corpus to {CORPUS_SAVE_FILE} ({e})")
    
    # 4. Show an example of what was built/loaded
    if corpus:
        print("\n--- Example: Processing Results ---")
        
        # Get one of the document IDs from our new corpus
        # list(corpus.keys())[0] gets the first doc_id
        example_doc_id = list(corpus.keys())[0]
        example_tokens = corpus[example_doc_id]
        
        print(f"Document ID: {example_doc_id}")
        print(f"Total processed tokens: {len(example_tokens)}")
        print(f"First 20 processed tokens: {example_tokens[:20]}")
    else:
        print("No corpus was built or loaded.")