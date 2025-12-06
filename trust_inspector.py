"""
Emotion Word Inspector (Trust Bias Analyzer)

This diagnostic script performs an audit on a specific book's emotion score. 
It identifies exactly which words are contributing to a specific emotion classification, 
helping to explain statistical anomalies (e.g., why "The Art of War" scores high in "Trust").

Action:
It reads a raw text file, checks every word against the NRC Emotion Lexicon, and counts 
occurrences of words associated with a target emotion (defaulting to 'Trust').

Connection:
This is a standalone debugging tool used to validate findings from the Archetype Percentile Test.
It does not generate data for the pipeline but helps us understand the "Black Box" 
of the lexicon-based analysis.

Inputs:
- Directory containing .txt files (default: "gutenberg_corpus")
- User input: Book title fragment (e.g., "Art of War")

Outputs:
- Console report showing the total count of words for the target emotion.
- A ranked table of the Top 30 words driving that score, including their 
  raw count and percentage contribution to the total.

Process:
1. Performs a fuzzy scan on the corpus directory to find the requested book file.
2. Reads and tokenizes the full text (simple linguistic cleaning).
3. Iterates through the unique vocabulary of the book.
4. Queries the NRCLex library for each word to see if it maps to the target emotion.
5. Aggregates counts and calculates the percentage influence of each word.
6. Prints a sorted leaderboard of the most influential words.
"""

import os
import collections
from nltk.tokenize import word_tokenize
from nrclex import NRCLex
# Run once
# nltk.download('punkt_tab')

# --- Configuration ---
CORPUS_DIR = "gutenberg_corpus"
# ---------------------

def inspect_trust_words(target_title_fragment):
    # 1. Find the file
    found_filename = None
    if not os.path.exists(CORPUS_DIR):
        print("Error: Corpus directory not found.")
        return

    # Fuzzy search for the file
    fragment_clean = target_title_fragment.lower().replace(" ", "")
    for filename in os.listdir(CORPUS_DIR):
        clean_name = filename.lower().replace("_", "").replace("-", "")
        if fragment_clean in clean_name:
            found_filename = filename
            break
    
    if not found_filename:
        print(f"Error: Could not find book matching '{target_title_fragment}'")
        return

    print(f"\nAnalyzing: {found_filename}")
    print("Scanning for 'Trust' words...")

    # 2. Read and Tokenize
    filepath = os.path.join(CORPUS_DIR, found_filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Simple cleaning (matching corpus_processor logic to keep things constistent)
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    
    # 3. The Inspector Logic
    trust_word_counts = collections.Counter()
    total_trust_words = 0
    
    # We instantiate NRCLex once to get the lexicon dictionary
    # Note: nrclex doesn't expose the raw dict easily, so we iterate and check.
    # Optimization: Create a memoized lookup or just check word-by-word.
    
    # Iterate through the book's unique words, then multiply by count
    unique_counts = collections.Counter(tokens)
    
    for word, frequency in unique_counts.items():
        # Create a tiny NRCLex object for just this word
        emotion_obj = NRCLex(word)
        
        # Check if 'trust' is in this word's emotion list
        # emotion_obj.affect_list returns ['joy', 'trust', ...]
        if 'trust' in emotion_obj.affect_list:
            trust_word_counts[word] += frequency
            total_trust_words += frequency

    # 4. Print Results
    print(f"\nTotal 'Trust' words found: {total_trust_words}")
    print("-" * 40)
    print(f"{'Word':<20} | {'Count':<10} | {'% of Trust Score'}")
    print("-" * 40)
    
    for word, count in trust_word_counts.most_common(30):
        percentage = (count / total_trust_words) * 100
        print(f"{word:<20} | {count:<10} | {percentage:.1f}%")


if __name__ == "__main__":
    while True:
        # Ask user for book title to search
        query = input("\nEnter book title to inspect for TRUST (or 'exit'): ").strip()
        if query.lower() == 'exit':
            break
        inspect_trust_words(query)