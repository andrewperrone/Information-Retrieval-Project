"""
Hybrid Information Retrieval System (Weighted)

This updated version includes a 'TEXT_WEIGHT' parameter to prevent emotion scores
from overpowering the keyword relevance. It also handles the Title Mapping.
"""

import pickle
import os
import re
import time
import glob
import numpy as np
from collections import defaultdict

# --- Configuration ---
INDEX_FILE = "search_index.pkl"
EMOTION_FILE = "emotion_results.pkl"
CORPUS_DIR = "gutenberg_corpus"
TEXT_WEIGHT = 50.0  # <--- NEW: Boosts text score x50 to match emotion scale
# ---------------------

class IRSystem:
    def __init__(self):
        self.inverted_index = {}
        self.idf_scores = {}
        self.doc_lengths = {}
        self.emotion_data = {}
        self.doc_ids = []
        self.title_map = {}
        
        self._load_data()
        self._load_titles()
        
    def _load_data(self):
        # 1. Load Text Index
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, 'rb') as f:
                data = pickle.load(f)
                self.inverted_index = data['inverted_index']
                self.idf_scores = data['idf_scores']
                self.doc_lengths = data.get('doc_lengths', {}) 
        
        # 2. Load Emotion Data
        if os.path.exists(EMOTION_FILE):
            with open(EMOTION_FILE, 'rb') as f:
                raw_data = pickle.load(f)
                self.emotion_data = {item[0]: item[1] for item in raw_data}
                self.doc_ids = list(self.emotion_data.keys())

    def _load_titles(self):
        # Scans the corpus directory to map Book IDs (e.g. '2701') 
        # to readable Titles (e.g. 'Moby Dick').
        file_paths = glob.glob(os.path.join(CORPUS_DIR, "*.txt"))
        
        for path in file_paths:
            filename = os.path.basename(path)
            parts = filename.split('_', 1)
            
            if len(parts) == 2:
                book_id = parts[0]
                raw_title = parts[1].replace('_', ' ').replace('.txt', '')
                self.title_map[book_id] = raw_title

    def get_readable_title(self, doc_id):
        parts = doc_id.split('_')
        if len(parts) >= 2:
            book_id = parts[0]
            chunk_num = parts[1]
            title = self.title_map.get(book_id, "Unknown Book")
            return f"{title} (Seg {chunk_num})"
        return doc_id

    def text_search(self, query_text):
        tokens = re.findall(r'\b[a-z]+\b', query_text.lower())
        if not tokens:
            return []
            
        doc_scores = defaultdict(float)
        
        for token in tokens:
            if token in self.inverted_index:
                idf = self.idf_scores.get(token, 0)
                matching_docs = self.inverted_index[token]
                
                for doc_id, tf in matching_docs.items():
                    doc_scores[doc_id] += tf * idf
        
        final_scores = []
        for doc_id, raw_score in doc_scores.items():
            length = self.doc_lengths.get(doc_id, 1)
            norm_factor = np.sqrt(length)
            if norm_factor < 1: norm_factor = 1
            
            # Apply the TEXT_WEIGHT boost here
            normalized_score = (raw_score / norm_factor) * TEXT_WEIGHT
            final_scores.append((doc_id, normalized_score))
            
        return sorted(final_scores, key=lambda x: x[1], reverse=True)

    def filter_by_emotion(self, text_results, emotion, min_score=0):
        # If no text results, search everything (Discovery Mode)
        if text_results:
            candidates = text_results
        else:
            candidates = [(doc, 0.0) for doc in self.doc_ids]
        
        filtered_results = []
        
        for doc_id, text_score in candidates:
            if doc_id in self.emotion_data:
                raw_emotion_count = self.emotion_data[doc_id].get(emotion, 0)
                length = self.doc_lengths.get(doc_id, 1)
                
                # Calculate density (0-100)
                emotion_density = (raw_emotion_count / length) * 100 if length > 0 else 0
                
                # Combined Score
                combined_score = text_score + emotion_density
                
                if raw_emotion_count >= min_score:
                    filtered_results.append((doc_id, combined_score, emotion_density))
        
        return sorted(filtered_results, key=lambda x: x[1], reverse=True)

# --- Main Loop ---
if __name__ == "__main__":
    system = IRSystem()
    
    print("\n" + "="*40)
    print("   SEGMENT-LEVEL SEARCH ENGINE READY")
    print("   (Type 'exit' to quit)")
    print("="*40)
    
    while True:
        print("\nOptions:")
        print("1. Text Search Only")
        print("2. Text + Emotion Filter")
        print("3. Emotion Only")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice.lower() == 'exit': 
            break
            
        elif choice == '1':
            query = input("Enter search terms: ").strip()
            results = system.text_search(query)
            
            print(f"\nFound {len(results)} matching segments.")
            print("--- Top 10 Results ---")
            for doc, score in results[:10]:
                title = system.get_readable_title(doc)
                print(f"[{score:.2f}] {title}")
                
        elif choice == '2':
            query = input("Enter search terms: ").strip()
            emotion = input("Enter emotion (joy, fear, anger, etc.): ").strip().lower()
            
            text_results = system.text_search(query)
            final_results = system.filter_by_emotion(text_results, emotion)
            
            print(f"\nFound {len(final_results)} segments matching '{query}' + '{emotion}'.")
            print("--- Top 10 Results ---")
            for doc, comb_score, emo_score in final_results[:10]:
                title = system.get_readable_title(doc)
                print(f"[Comb: {comb_score:.2f} | {emotion}: {emo_score:.2f}%] {title}")
        
        elif choice == '3':
            emotion = input("Enter emotion to explore: ").strip().lower()
            final_results = system.filter_by_emotion([], emotion)
            print(f"\nFound {len(final_results)} segments ranked by '{emotion}'.")
            for doc, comb_score, emo_score in final_results[:10]:
                title = system.get_readable_title(doc)
                print(f"[Density: {emo_score:.2f}%] {title}")