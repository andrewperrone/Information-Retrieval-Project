"""
Gutenberg Index Builder

This script builds a search index from processed text documents to enable efficient
information retrieval. It creates an inverted index, calculates IDF scores, and
tracks document lengths for use in ranking search results.

Action:
It counts word frequencies (TF), calcualtes word rarity (IDF), and maps words to books (Inverted Index).
It also counts total words per book (Doc Lengths).

Connection:
This provides the Text Search capability to the IR System, by generating search_index.pkl.

Inputs:
- processed_corpus.pkl: A pickled dictionary of {doc_id: [tokens]} created by 2_corpus_processor.py

Outputs:
- search_index.pkl: A pickled dictionary containing:
  - 'inverted_index': {term: {doc_id: term_frequency, ...}}
  - 'idf_scores': {term: idf_score}
  - 'doc_lengths': {doc_id: total_word_count}

Process:
1. Loads the processed corpus from the input pickle file
2. Builds an inverted index mapping terms to documents and their frequencies
3. Calculates Inverse Document Frequency (IDF) scores for each term
4. Tracks document lengths for normalization in search results
5. Saves the index components to a single pickle file for later use
6. Provides statistics about the indexing process
"""

import pickle
import math
import os
import time
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

def build_index(processed_corpus_file: str) -> tuple[dict, dict, dict]:
    """
    Builds search index components from a processed corpus.
    
    Args:
        processed_corpus_file (str): Path to the pickled corpus file containing 
            {doc_id: [tokens]} mapping
    
    Returns:
        tuple: A tuple containing three elements:
            - inverted_index (dict): {term: {doc_id: term_frequency, ...}}
            - idf_scores (dict): {term: idf_score} where idf_score = log(N/(df + 1))
            - doc_lengths (dict): {doc_id: total_word_count}
    """
    print(f"Loading corpus from {processed_corpus_file}...")
    
    if not os.path.exists(processed_corpus_file):
        print("Error: Processed corpus not found.")
        return None, None, None

    with open(processed_corpus_file, 'rb') as f:
        corpus = pickle.load(f)

    num_documents = len(corpus)
    print(f"Loaded {num_documents} documents. Building index...")
    
    start_time = time.time()

    # --- Data Structures ---
    inverted_index = defaultdict(dict) 
    doc_frequency = defaultdict(int)
    doc_lengths = {}

    # --- Step 1: Build Index ---
    for doc_id, tokens in corpus.items():
        
        doc_lengths[doc_id] = len(tokens)
        
        term_counts = Counter(tokens)
        
        for token, count in term_counts.items():
            inverted_index[token][doc_id] = count
            doc_frequency[token] += 1

    # --- Step 2: Calculate IDF ---
    # IDF (Inverse Document Frequency) measures how important a term is across documents
    # We use log(N/(df + 1)) where:
    # - N: total number of documents
    # - df: document frequency of the term
    # The +1 in denominator is for smoothing (avoiding division by zero)
    idf_scores = {}
    for token, freq in doc_frequency.items():
        idf_scores[token] = math.log(num_documents / (freq + 1))

    end_time = time.time()
    
    print("\n--- Indexing Complete ---")
    print(f"Total unique terms indexed: {len(idf_scores)}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    return inverted_index, idf_scores, doc_lengths

def save_index(inverted_index: dict, idf_scores: dict, doc_lengths: dict, 
               filename: str = "search_index.pkl") -> None:
    """
    Saves the search index components to a pickle file for later use.
    
    Args:
        inverted_index (dict): Inverted index mapping terms to document frequencies
        idf_scores (dict): IDF scores for each term
        doc_lengths (dict): Length of each document in tokens
        filename (str): Path where the index will be saved (default: "search_index.pkl")
    
    The saved file contains a dictionary with three keys:
    - 'inverted_index': The main inverted index
    - 'idf_scores': Pre-computed IDF scores
    - 'doc_lengths': Document lengths for normalization
    """
    data = {
        'inverted_index': inverted_index,
        'idf_scores': idf_scores,
        'doc_lengths': doc_lengths # <-- NEW
    }
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving index: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    CORPUS_FILE = "processed_corpus.pkl"  # Input: Output from corpus processor
    INDEX_FILE = "search_index.pkl"       # Output: Will contain the search index
    
    # 1. Build the index components
    print("Starting index construction...")
    inv_index, idfs, lengths = build_index(CORPUS_FILE)
    
    # 2. Save the index if all components were built successfully
    if inv_index and idfs and lengths:
        print(f"Saving index to {INDEX_FILE}...")
        save_index(inv_index, idfs, lengths, INDEX_FILE)
    else:
        print("Error: Failed to build one or more index components")