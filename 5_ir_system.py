"""
Hybrid Information Retrieval System with Emotion Analysis

This script implements a search engine that combines traditional TF-IDF text search with emotion analysis.
It allows users to search documents by keywords, filter by emotional content, or discover documents with specific emotional profiles.

Action: It waits for a user query.
1. It uses the Index to find books matching keywords (BM25(pending)/TF-IDF(current)).
2. It uses Emotion Results + Stats to re-rank those books based on emotional intensity (Z-Score).

Key Features:
1. Text Search: Traditional TF-IDF with length normalization
2. Emotion Filtering: Filter search results by emotional content
3. Emotion Discovery: Find documents with highest density of specific emotions
4. Hybrid Scoring: Combines text relevance with emotional characteristics

Inputs:
- search_index.pkl: Pre-built inverted index with TF-IDF scores and document lengths
- emotion_results.pkl: Pre-computed emotion analysis results for documents

Outputs:
- Search results displayed in console, including:
  - Document IDs
  - Relevance scores (TF-IDF)
  - Emotion density scores (percentage)
  - Combined scores for hybrid search

Process:
1. Initialization: Loads pre-computed indices and emotion data
2. Query Processing: Tokenizes and normalizes search queries
3. Text Search: Performs TF-IDF search with length normalization
4. Emotion Analysis: Filters and ranks results based on emotional content
5. Result Presentation: Displays formatted search results
"""

import pickle
import os
import re
import time
import numpy as np
import nltk
from nltk.corpus import wordnet
from collections import defaultdict

# --- Configuration ---
INDEX_FILE = "search_index.pkl"
EMOTION_FILE = "emotion_results.pkl"
STATS_FILE = "emotion_stats.pkl" # <-- NEW: Needed for Z-Scores
# ---------------------

class IRSystem:
    def __init__(self):
        print("--- Initializing IR System ---")
        self.inverted_index = {}
        self.idf_scores = {}
        self.doc_lengths = {}
        self.emotion_data = {}
        self.emotion_stats = {} # <-- NEW
        self.doc_ids = []
        
        # Ensure WordNet is available for synonyms
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading WordNet data...")
            nltk.download('wordnet')
        
        self._load_data()
        
    def _load_data(self):
        """Loads the pickle files into memory once."""
        start_time = time.time()
        
        # 1. Load Text Index
        if os.path.exists(INDEX_FILE):
            print(f"Loading Text Index from {INDEX_FILE}...")
            with open(INDEX_FILE, 'rb') as f:
                data = pickle.load(f)
                self.inverted_index = data['inverted_index']
                self.idf_scores = data['idf_scores']
                self.doc_lengths = data.get('doc_lengths', {}) 
        else:
            print(f"CRITICAL WARNING: {INDEX_FILE} not found.")

        # 2. Load Emotion Data
        if os.path.exists(EMOTION_FILE):
            print(f"Loading Emotion Data from {EMOTION_FILE}...")
            with open(EMOTION_FILE, 'rb') as f:
                raw_data = pickle.load(f)
                self.emotion_data = {item[0]: item[1] for item in raw_data}
                self.doc_ids = list(self.emotion_data.keys())
        else:
            print(f"WARNING: {EMOTION_FILE} not found.")

        # 3. Load Emotion Statistics (Z-Scores)
        if os.path.exists(STATS_FILE):
            print(f"Loading Emotion Stats from {STATS_FILE}...")
            with open(STATS_FILE, 'rb') as f:
                self.emotion_stats = pickle.load(f)
        else:
            print(f"WARNING: {STATS_FILE} not found. Z-Score ranking unavailable.")
            
        print(f"System loaded in {time.time() - start_time:.2f} seconds.")

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                clean_syn = lemma.name().lower().replace('_', ' ')
                if ' ' not in clean_syn: 
                    synonyms.add(clean_syn)
        return synonyms

    def process_query(self, query_text):
        raw_tokens = re.findall(r'\b[a-z]+\b', query_text.lower())
        expanded_tokens = set(raw_tokens)
        
        for token in raw_tokens:
            syns = self.get_synonyms(token)
            expanded_tokens.update(syns)
            
        return list(expanded_tokens)

    def text_search(self, query_text):
        """
        Performs a Length-Normalized TF-IDF search.
        """
        tokens = self.process_query(query_text)
        if not tokens:
            return []
            
        doc_scores = defaultdict(float)
        
        for token in tokens:
            if token in self.inverted_index:
                idf = self.idf_scores.get(token, 0)
                matching_docs = self.inverted_index[token]
                
                for doc_id, tf in matching_docs.items():
                    score = tf * idf
                    doc_scores[doc_id] += score
        
        final_scores = []
        for doc_id, raw_score in doc_scores.items():
            length = self.doc_lengths.get(doc_id, 1)
            norm_factor = np.sqrt(length)
            if norm_factor < 1: norm_factor = 1
            
            normalized_score = raw_score / norm_factor
            final_scores.append((doc_id, normalized_score))
            
        ranked_results = sorted(final_scores, key=lambda x: x[1], reverse=True)
        return ranked_results

    def filter_by_emotion(self, text_results, emotion, min_score=0, text_weight=1.0, emotion_weight=1.0):
        """
        Filters results using Z-Score normalization for emotions.
        """
        filtered_results = []
        
        if text_results:
            candidates = text_results
        else:
            candidates = [(doc, 0) for doc in self.doc_ids]
        
        # Get Corpus Stats for this emotion
        stats = self.emotion_stats.get(emotion)
        
        for doc_id, text_score in candidates:
            if doc_id in self.emotion_data:
                raw_emotion_count = self.emotion_data[doc_id].get(emotion, 0)
                length = self.doc_lengths.get(doc_id, 1)
                
                # Calculate Density
                emotion_density = raw_emotion_count / length
                
                # --- NEW: Z-Score Calculation ---
                if stats and stats['std'] > 0:
                    # Z = (Value - Mean) / StdDev
                    z_score = (emotion_density - stats['mean']) / stats['std']
                else:
                    z_score = 0 
                
                # Treat anything below average (negative Z) as 0 benefit for ranking
                effective_emotion_score = max(0, z_score)

                # Combined Score: Text + (Z-Score * Weight)
                combined_score = (text_score * text_weight) + (effective_emotion_score * emotion_weight)
                
                # Filter logic (still use raw count for strict "min_score" filtering)
                if raw_emotion_count >= min_score:
                    # Return (doc_id, combined_score, Z-SCORE)
                    filtered_results.append((doc_id, combined_score, z_score))
        
        return sorted(filtered_results, key=lambda x: x[1], reverse=True)

# --- Main Loop ---
if __name__ == "__main__":
    system = IRSystem()
    
    print("\n" + "="*40)
    print("   DEV LEVEL SEARCH ENGINE (Z-Score)")
    print("   (Type 'exit' to quit)")
    print("="*40)
    
    while True:
        print("\nOptions:")
        print("1. Text Search Only")
        print("2. Text + Emotion Filter")
        print("3. Emotion Only (Discovery Mode)")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice.lower() == 'exit': 
            break
            
        if choice == '1':
            query = input("Enter search terms: ").strip()
            results = system.text_search(query)
            
            print(f"\nFound {len(results)} matching documents.")
            print("--- Top 10 Results ---")
            for doc, score in results[:10]:
                print(f"[{score:.2f}] {doc}")
                
        elif choice == '2':
            query = input("Enter search terms: ").strip()
            emotion = input("Enter emotion (joy, fear, anger, etc.): ").strip().lower()
            
            # Using weights derived from your grid search (Text heavy)
            text_results = system.text_search(query)
            final_results = system.filter_by_emotion(text_results, emotion, text_weight=0.5, emotion_weight=1.0)
            
            print(f"\nFound {len(final_results)} documents matching '{query}' with '{emotion}'.")
            print("--- Top 10 Results ---")
            for doc, comb_score, z_score in final_results[:10]:
                # Updated print to show Z-Score
                print(f"[Comb: {comb_score:.2f} | {emotion} Z-Score: {z_score:+.2f}] {doc}")

        elif choice == '3':
            emotion = input("Enter emotion to explore (joy, fear, etc.): ").strip().lower()
            
            # Emotion Only mode
            final_results = system.filter_by_emotion([], emotion, text_weight=0.0, emotion_weight=1.0)
            
            print(f"\nFound {len(final_results)} documents ranked by '{emotion}'.")
            print(f"--- Top 10 Most '{emotion.title()}' Books ---")
            for doc, comb_score, z_score in final_results[:10]:
                # Updated print to show Z-Score
                print(f"[Z-Score: {z_score:+.2f}] {doc}")
                
        else:
            print("Invalid choice.")