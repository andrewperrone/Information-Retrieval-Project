"""
Emotion Validator for Book Corpus

This script evaluates the accuracy of emotion detection by comparing the system's output against
predefined emotional archetypes from classic literature. It calculates percentiles to show how strongly
the detected emotions match the expected emotional profiles of well-known books.

Action:
It tests if the emotion detection system correctly identifies the dominant emotions in classic books
with well-established emotional tones (e.g., fear in "Dracula", joy in "Winnie the Pooh").

Connection:
Imports IRSystem from 5_ir_system.py to access emotion data and document information.

Inputs:
- Pre-loaded IR system with emotion data
- Hardcoded list of literary archetypes and their expected emotions

Outputs:
- Console table showing each book's expected emotion vs actual dominant emotions
- Percentile scores indicating how strongly each book matches its expected emotion
- Visual indicators of analysis quality (Excellent/Good/Mismatch)

Process:
1. Loads the IR system and its emotion data
2. Calculates emotion densities across the entire corpus
3. For each archetypal book, determines the percentile of its expected emotion
4. Displays a comparison of expected vs actual dominant emotions
5. Provides interpretation guidelines for the results
"""

import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("ir_system", "5_ir_system.py")
ir_system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir_system)
IRSystem = ir_system.IRSystem
# Above replaces this faulty import:
# from 5_ir_system import IRSystem
from typing import List, Tuple, Dict, Any

# --- Configuration: The "Golden Standard" for Overall feel ---
# We list books that are UNDENIABLY associated with specific emotions.
ARCHETYPES = [
    {"title": "Dracula", "expected_emotion": "fear"},
    {"title": "Frankenstein", "expected_emotion": "fear"},
    {"title": "The War of the Worlds", "expected_emotion": "fear"},
    {"title": "Winnie the Pooh", "expected_emotion": "trust"},
    {"title": "Pride and Prejudice", "expected_emotion": "trust"},
    {"title": "Alice in Wonderland", "expected_emotion": "surprise"},
    {"title": "The Adventures of Tom Sawyer", "expected_emotion": "joy"},
    {"title": "Romeo and Juliet", "expected_emotion": "sadness"},
    {"title": "The Picture of Dorian Gray", "expected_emotion": "disgust"},
    {"title": "The Art of War", "expected_emotion": "fear"}
]

class EmotionValidator:
    def __init__(self):
        self.system = IRSystem()
        
    def find_doc_id(self, fragment):
        """Helper to find the full filename from a partial title."""
        fragment = fragment.lower().replace(" ", "")
        for doc_id in self.system.doc_ids:
            clean_id = doc_id.lower().replace("_", "").replace("-", "")
            if fragment in clean_id:
                return doc_id
        return None

    def calculate_percentiles(self):
        print(f"\n{'='*85}")
        print(f"STARTING ARCHETYPE PERCENTILE TEST")
        print(f"{'='*85}")
        # Added 'Dominant Emotion' column
        print(f"{'Book Title':<25} | {'Exp. Emotion':<12} | {'Pct':<6} | {'Rank':<9} | {'ACTUAL Dominant (Top 3)':<30}")
        print("-" * 105)

        # We need to pre-calculate the density distribution for every emotion
        # to know what "High" looks like.
        
        # 1. Build a dictionary of {emotion: [list_of_all_densities]}
        emotion_distributions = {
            'fear': [], 'joy': [], 'trust': [], 'sadness': [], 
            'surprise': [], 'anger': [], 'disgust': [], 'anticipation': [],
            'positive': [], 'negative': [] # Tracking these can help debug too
        }
        
        # Pre-calculate densities for the whole corpus
        doc_densities = {} # {doc_id: {'fear': 0.015, 'joy': 0.002...}}
        
        for doc_id in self.system.doc_ids:
            doc_densities[doc_id] = {}
            length = self.system.doc_lengths.get(doc_id, 1)
            
            # Get raw vectors
            if doc_id in self.system.emotion_data:
                vector = self.system.emotion_data[doc_id]
                for emo in emotion_distributions.keys():
                    count = vector.get(emo, 0)
                    density = count / length
                    
                    emotion_distributions[emo].append(density)
                    doc_densities[doc_id][emo] = density

        # Sort distributions so we can calculate percentiles
        for emo in emotion_distributions:
            emotion_distributions[emo].sort()

        # 2. Test our Archetypes
        for case in ARCHETYPES:
            title = case['title']
            target_emo = case['expected_emotion']
            
            doc_id = self.find_doc_id(title)
            if not doc_id:
                print(f"{title[:25]:<25} | {target_emo:<12} | {'N/A':<6} | {'N/A':<9} | NOT FOUND")
                continue
                
            # Get the density of the target book for the target emotion
            actual_density = doc_densities[doc_id].get(target_emo, 0)
            
            # Compare against the universe of all books
            universe = emotion_distributions[target_emo]
            count_lower = sum(1 for x in universe if x < actual_density)
            percentile = (count_lower / len(universe)) * 100
            rank = len(universe) - count_lower
            
            # --- NEW: Find the ACTUAL dominant emotions for this book ---
            # Sort this book's emotions by density
            my_emotions = doc_densities[doc_id]
            # Filter out positive/negative to focus on the 8 core emotions for clarity
            core_emotions = {k: v for k, v in my_emotions.items() if k not in ['positive', 'negative']}
            sorted_emotions = sorted(core_emotions.items(), key=lambda x: x[1], reverse=True)
            
            # Create a string of the top 3 (e.g., "joy(1.2%), trust(1.1%)")
            top_3_str = ", ".join([f"{e}({d*100:.1f}%)" for e, d in sorted_emotions[:3]])
            
            print(f"{title[:25]:<25} | {target_emo:<12} | {percentile:4.1f}%  | #{rank:<4}/{len(universe)} | {top_3_str}")

        print("-" * 105)
        print("INTERPRETATION:")
        print(" - 90-100%: Excellent match. The book is definitively in this genre.")
        print(" - 70-90%:  Good match. The emotion is present but maybe not dominant.")
        print(" - < 50%:   Mismatch. Look at the 'ACTUAL Dominant' column to see what the system found instead.")

if __name__ == "__main__":
    validator = EmotionValidator()
    validator.calculate_percentiles()