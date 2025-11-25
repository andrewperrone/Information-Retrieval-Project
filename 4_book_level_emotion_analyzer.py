"""
Book-Level Emotion Analyzer with Negation Handling

This script analyzes the emotional content of books from Project Gutenberg using the NRC Emotion Lexicon.
It processes each book in chunks (paragraphs) and implements negation handling to improve emotion detection accuracy.
Emotional words that appear after negation terms (e.g., 'not happy') are excluded from the analysis.

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

# --- Negation Configuration ---
NEGATION_TERMS = {
    'not', 'never', 'no', 'nothing', 'neither', 'nor', 
    'nowhere', 'hardly', 'scarcely', 'barely', 'didnt', 
    'dont', 'doesnt', 'wont', 'wouldnt', 'couldnt', 
    'shouldnt', 'cant', 'cannot', "n't"
}

def get_negation_aware_emotions(text_chunk):
    """
    Analyzes a text chunk for emotions, but IGNORES words
    that are immediately preceded by a negation term.
    """
    # Tokenize the chunk so we can look at previous words
    tokens = word_tokenize(text_chunk.lower())
    chunk_vector = defaultdict(int)
    
    # Define a 'window' to look back. 
    LOOKBACK_WINDOW = 2 
    
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

def analyze_corpus_emotions_by_book(corpus_dir):
    """
    Reads all .txt files, analyzes them chunk-by-chunk for 8 core emotions
    (with negation handling), and returns a list of (doc_id, vector) tuples.
    """
    print(f"Starting NRC Emotion Lexicon (Book-Level + Negation) analysis...")
    
    # This list will store tuples: (doc_id, {'joy': 150, 'anger': 80, ...})
    all_book_emotions = []
    
    # 2. Find all .txt files
    file_paths = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    if not file_paths:
        print(f"Error: No .txt files found in '{corpus_dir}'.")
        return None
        
    print(f"Found {len(file_paths)} files. Starting analysis...")
    start_time = time.time()
    
    # 3. Loop, Read, and Analyze each file
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
            
            # 5. Analyze each chunk using the NEW NEGATION FUNCTION
            for chunk in chunks:
                # --- CHANGED: Use custom function instead of raw NRCLex ---
                chunk_scores = get_negation_aware_emotions(chunk)
                
                # Add these counts to the book's total vector
                for emotion, score in chunk_scores.items():
                    book_emotion_vector[emotion] += score
            
            # 6. Store the final vector for the book
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
    
    CORPUS_DIRECTORY = "gutenberg_corpus"
    
    # Make sure NLTK tokenizer data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    results = analyze_corpus_emotions_by_book(CORPUS_DIRECTORY)
    
    # 3. Show the most "emotional" books
    if results:
        print("\n--- Top 5 Most 'Joyful' Books (Negation Aware) ---")
        results.sort(key=lambda x: x[1].get('joy', 0), reverse=True)
        for doc_id, vector in results[:5]:
            print(f"  Joy Count: {vector.get('joy', 0):<5} | File: {doc_id}")