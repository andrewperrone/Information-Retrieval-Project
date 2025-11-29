"""
Chunk-Level Emotion Analyzer with Negation Handling

This script analyzes the emotional content of pre-processed text segments (chunks) using the 
NRC Emotion Lexicon. Unlike the previous book-level version, this script maintains separate 
emotion scores for every chunk, enabling precise retrieval of specific passages.

It implements negation handling to improve accuracy: Emotional words appearing immediately 
after negation terms (e.g., 'not happy') are excluded from the analysis.

Key Features:
- Segment-level analysis (score per chunk, not per book)
- Token-level emotion detection with lookback negation
- Configurable lookback window (default: 3 words)
- Handles common negation terms and contractions

Inputs:
- processed_corpus.pkl: Dictionary of pre-tokenized chunks (from 2_corpus_processor.py)
- NRC Emotion Lexicon (automatically downloaded via nrclex)

Outputs:
- emotion_results.pkl: A pickled list of (doc_id, emotion_vector) tuples where:
  - doc_id is the unique segment ID (e.g., '2701_0', '2701_1')
  - emotion_vector is a dictionary of scores (e.g., {'joy': 2, 'fear': 0, ...})

Process:
1. Loads the pre-processed segmented corpus (processed_corpus.pkl)
2. For each segment (chunk):
   - Reconstructs text or iterates tokens
   - Checks for emotional words while looking back for negation terms
   - Excludes emotional words that follow negation terms
   - Stores the emotion vector for that specific chunk
3. Saves the list of chunk-level emotion vectors to emotion_results.pkl
"""

import pickle
import time
from collections import defaultdict
from nrclex import NRCLex

# Inputs/Outputs
CORPUS_FILE = "processed_corpus.pkl"
EMOTION_FILE = "emotion_results.pkl"

# --- Negation Configuration ---
# These must match what is preserved in the processor (alpha words + n't)
NEGATION_TERMS = {
    'not', 'never', 'no', 'nothing', 'neither', 'nor', 
    'hardly', 'scarcely', 'barely', 'didnt', 'dont', 
    'doesnt', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 
    'cant', 'cannot', "n't"
}
LOOKBACK_WINDOW = 3  # How many words back to check for negation

def get_negation_aware_emotions(tokens):
    """
    Analyzes a list of tokens for emotions, ignoring words
    preceded by a negation term.
    """
    chunk_vector = defaultdict(int)
    
    for i, word in enumerate(tokens):
        # 1. Check if the word has emotion (using NRCLex on single word)
        # NRCLex is slightly slow per-word, but accurate for this logic
        word_obj = NRCLex(word)
        word_emotions = word_obj.raw_emotion_scores
        
        # If no emotion, skip
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

def analyze_chunks():
    print(f"Loading segmented corpus from {CORPUS_FILE}...")
    try:
        with open(CORPUS_FILE, 'rb') as f:
            corpus = pickle.load(f)
    except FileNotFoundError:
        print("Error: processed_corpus.pkl not found. Run 2_corpus_processor.py first.")
        return

    print(f"Analyzing {len(corpus)} segments with Negation Handling...")
    results = [] # List of (doc_id, emotion_dict)
    
    start_time = time.time()
    
    for i, (doc_id, tokens) in enumerate(corpus.items()):
        
        # Use the custom negation function on the token list
        raw_scores = get_negation_aware_emotions(tokens)
        
        if raw_scores:
            results.append((doc_id, dict(raw_scores)))
            
        if (i+1) % 500 == 0:
            print(f"  Analyzed {i+1} segments...")
            
    end_time = time.time()
    
    print(f"\n--- Emotion Analysis Complete ---")
    print(f"Segments with emotion: {len(results)}")
    print(f"Time taken: {end_time - start_time:.2f} s")
    
    print(f"Saving to {EMOTION_FILE}...")
    with open(EMOTION_FILE, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    analyze_chunks()