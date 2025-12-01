"""
Book-Level Emotion Analyzer with Negation Handling

This script analyzes the emotional content of books from Project Gutenberg using the NRC Emotion Lexicon.
It processes each book in chunks (paragraphs) and implements negation handling to improve emotion detection accuracy.
Emotional words that appear after negation terms (e.g., 'not happy') are excluded from the analysis.

Action:
It scans books for emotion words using the NRC Lexicon and handles negation (skipping "not happy").

Connection:
This provides the Raw Emotion Data to the IR System, by generating emotion_results.pkl.

Key Features:
- Token-level emotion analysis with negation detection
- Configurable lookback window for negation terms
- Handles common negation terms and contractions
- Efficient processing of large texts using chunking

Inputs:
- gutenberg_corpus/: Directory containing .txt files of books from Project Gutenberg
- NRC Emotion Lexicon (automatically downloaded via nrclex)
- NLTK punkt tokenizer (automatically downloaded if missing)

Outputs:
- emotion_results.pkl: A pickled list of (doc_id, emotion_vector) tuples where:
  - doc_id is the book's filename
  - emotion_vector is a dictionary of emotion scores (e.g., {'joy': 150, 'anger': 80, ...})
- Console output showing analysis progress and top results

Process:
1. Scans the corpus directory for .txt files (books)
2. For each book:
   - Splits text into chunks (paragraphs separated by double newlines)
   - Analyzes each chunk with negation-aware emotion detection:
     * Tokenizes text and checks for emotional words
     * Looks for negation terms in a configurable window (default: 2 words)
     * Excludes emotional words that follow negation terms
   - Aggregates emotion scores across all chunks
3. Saves the aggregated emotion data to a pickle file
4. Prints top results and summary statistics
"""

import nltk
import glob
import os
import time
import pickle
from nrclex import NRCLex
from collections import defaultdict
from nltk.tokenize import word_tokenize
from typing import List, Tuple, Dict, Any

# --- Configuration ---
# Terms that negate the emotional meaning of subsequent words
# These are used to identify and exclude negated emotional expressions
NEGATION_TERMS = {
    'not', 'never', 'no', 'nothing', 'neither', 'nor', 
    'nowhere', 'hardly', 'scarcely', 'barely', 'didnt', 
    'dont', 'doesnt', 'wont', 'wouldnt', 'couldnt', 
    'shouldnt', 'cant', 'cannot', "n't"
}

# Number of previous words to check for negation terms
# This defines the window size for negation detection
NEGATION_WINDOW_SIZE = 2

def get_negation_aware_emotions(text_chunk: str) -> dict:
    """
    Analyzes a text chunk for emotions while handling negation contexts.
    
    Args:
        text_chunk (str): A segment of text to analyze for emotional content
        
    Returns:
        dict: A dictionary mapping emotion types to their scores in the text chunk,
              with negated emotional words excluded
              
    Example:
        >>> text = "I am happy but not sad"
        >>> get_negation_aware_emotions(text)
        {'joy': 1}  # 'sad' is negated and excluded
    """
    # Tokenize the chunk so we can look at previous words
    tokens = word_tokenize(text_chunk.lower())
    chunk_vector = defaultdict(int)
    
    # Define a 'window' to look back. 
    LOOKBACK_WINDOW = NEGATION_WINDOW_SIZE 
    
    for i, word in enumerate(tokens):
        # 1. Check if the word has emotion
        # Create new NRCLex instance for a SINGLE word to check its dict
        word_obj = NRCLex(word)
        word_emotions = word_obj.raw_emotion_scores
        
        # If this word has no emotional content, skip it
        if not word_emotions:
            continue
            
        # 2. Check for Negation in the previous N words
        is_negated = False
        start_index = max(0, i - LOOKBACK_WINDOW)
        previous_words = tokens[start_index:i]
        
        for prev_word in previous_words:
            if prev_word in NEGATION_TERMS:
                is_negated = True
                break
        
        # 3. Add to vector ONLY if not negated
        if not is_negated:
            for emotion, score in word_emotions.items():
                chunk_vector[emotion] += score
    
    return chunk_vector

def analyze_corpus_emotions_by_book(corpus_dir: str) -> list[tuple[str, dict]]:
    """
    Analyzes emotional content of all text files in a directory using the NRC Emotion Lexicon.
    
    Args:
        corpus_dir (str): Path to directory containing .txt files to analyze
        
    Returns:
        list[tuple[str, dict]]: List of (document_id, emotion_vector) tuples where:
            - document_id (str): The filename of the analyzed document
            - emotion_vector (dict): Dictionary mapping emotion types to their scores
            
    The function processes each document in chunks, applies negation-aware emotion detection,
    and aggregates scores across all chunks. Results are automatically saved to 'emotion_results.pkl'.
    """
    print(f"Starting NRC Emotion Lexicon (Book-Level + Negation) analysis...")
    
    # This list will store tuples: (doc_id, {'joy': 150, 'anger': 80, ...})
    all_book_emotions = []
    
    # Find all .txt files
    file_paths = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    if not file_paths:
        print(f"Error: No .txt files found in '{corpus_dir}'.")
        return None
        
    print(f"Found {len(file_paths)} files. Starting analysis...")
    start_time = time.time()
    
    # Loop, Read, and Analyze each file
    for i, filepath in enumerate(file_paths):
        doc_id = os.path.basename(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # --- The Chunking Method ---
            chunks = [chunk for chunk in raw_text.split('\n\n') if chunk.strip()]
            
            if not chunks:
                # print(f"Warning: Could not find any text chunks in {doc_id}. Skipping.")
                continue
                
            # --- The Aggregation Method ---
            book_emotion_vector = defaultdict(int)
            
            # Analyze each chunk using the NEW NEGATION FUNCTION
            for chunk in chunks:
                # --- Use custom function instead of raw NRCLex, includes negation ---
                chunk_scores = get_negation_aware_emotions(chunk)
                
                # Add these counts to the book's total vector
                for emotion, score in chunk_scores.items():
                    book_emotion_vector[emotion] += score
            
            # Store the final vector for the book
            if book_emotion_vector:
                all_book_emotions.append((doc_id, dict(book_emotion_vector)))
            
            if (i + 1) % 100 == 0:
                print(f"  Analyzed {i+1}/{len(file_paths)} files...")

        except Exception as e:
            print(f"Warning: Error processing file {doc_id} ({e}). Skipping.")
            
    end_time = time.time()
    print(f"\n--- Emotion Analysis Complete ---")
    print(f"Successfully analyzed {len(all_book_emotions)} documents.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    # --- Save results to the new pickle file ---
    NEW_SAVE_FILE = 'emotion_results.pkl'
    try:
        with open(NEW_SAVE_FILE, 'wb') as f:
            pickle.dump(all_book_emotions, f)
        print(f"Emotion results saved to '{NEW_SAVE_FILE}'")
    except Exception as e:
        print(f"Warning: Could not save emotion results. {e}")
        
    return all_book_emotions

# --- Main execution ---
if __name__ == "__main__":
    # Configuration
    CORPUS_DIRECTORY = "gutenberg_corpus"  # Directory containing text files to analyze
    
    # Ensure required NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')

    # Run emotion analysis on the corpus
    print("Starting emotion analysis...")
    results = analyze_corpus_emotions_by_book(CORPUS_DIRECTORY)
    
    # Display top results
    if results:
        # Sort by joy score in descending order and show top 5
        print("\n--- Top 5 Most 'Joyful' Books (Negation Aware) ---")
        results.sort(key=lambda x: x[1].get('joy', 0), reverse=True)
        for i, (doc_id, vector) in enumerate(results[:5], 1):
            print(f"{i}. {doc_id}")
            print(f"   Joy: {vector.get('joy', 0):<5} | "
                  f"Sadness: {vector.get('sadness', 0):<5} | "
                  f"Anger: {vector.get('anger', 0):<5} | "
                  f"Fear: {vector.get('fear', 0):<5}")