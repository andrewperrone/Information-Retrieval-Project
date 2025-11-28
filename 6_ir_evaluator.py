"""
Information Retrieval System Evaluator

This script evaluates the performance of the IR system by testing its ability to retrieve
specific known documents from the corpus. It calculates Mean Reciprocal Rank (MRR) to
measure search result quality and identifies the ranking position of target documents.

Inputs:
- test_cases.json: JSON file containing test cases with queries and expected documents
- Pre-built IR system (via IRSystem)
- Processed corpus and search index

Outputs:
- Console output showing test results and MRR score
- Detailed ranking information for each test case
- Overall evaluation metrics

Process:
1. Loads test cases from JSON file
2. For each test case:
   a. Finds the target document ID using fuzzy matching
   b. Executes the search query using the IR system
   c. Determines the rank of the target document in results
   d. Calculates reciprocal rank for the test case
3. Computes overall MRR across all test cases
4. Prints detailed evaluation report
"""

# 1. Finds the correct doc_id in the corpus for "Moby Dick" (handling the 123_filename.txt format).
# 2. Executes the search.
# 3. Finds where "Moby Dick" ended up in the ranking
# 4. Computes the MRR (Mean Reciprocal Rank). If the book is #1, score is 1.0, if #2, score is 0.5, if #10, score is 0.1.

import json
import os
import re
import importlib.util
spec = importlib.util.spec_from_file_location("ir_system", "5_ir_system.py")
ir_system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir_system)
IRSystem = ir_system.IRSystem
# Above replaces this faulty import:
# from 5_ir_system import IRSystem

# --- Configuration ---
TEST_CASES_FILE = "test_cases.json"
# ---------------------

class IREvaluator:
    def __init__(self):
        # Load the actual system
        self.system = IRSystem()
        self.corpus_files = self.system.doc_ids
        
    def normalize_string(self, text):
        """
        Aggressively cleans a string for fuzzy matching.
        """
        text = text.lower()
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def find_target_doc_id(self, target_title_fragment):
        """
        Fuzzy search to find the actual filename.
        """
        target_clean = self.normalize_string(target_title_fragment)
        matches = []
        for doc_id in self.corpus_files:
            filename_clean = self.normalize_string(doc_id)
            if target_clean in filename_clean:
                matches.append(doc_id)
        return matches

    def evaluate(self, text_weight=1.0, emotion_weight=1.0, verbose=True):
        """
        Runs the evaluation suite with specific weights.
        Returns the Mean Reciprocal Rank (MRR) score.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATION | Text Weight: {text_weight} | Emotion Weight: {emotion_weight}")
            print(f"{'='*60}")
        
        if not os.path.exists(TEST_CASES_FILE):
            print(f"Error: {TEST_CASES_FILE} not found.")
            return 0.0

        with open(TEST_CASES_FILE, 'r') as f:
            test_cases = json.load(f)

        total_mrr = 0
        total_cases = 0
        
        for case in test_cases:
            target_title = case['target_title']
            query = case['query']
            emotion = case['emotion']
            
            # 1. Identify Target
            target_ids = self.find_target_doc_id(target_title)
            if not target_ids:
                if verbose: print(f"    [SKIPPED] '{target_title}' not found in corpus.")
                continue
            
            # 2. Run System with WEIGHTS
            text_results = self.system.text_search(query)
            
            # --- PASSING WEIGHTS HERE ---
            final_results = self.system.filter_by_emotion(
                text_results, 
                emotion, 
                text_weight=text_weight, 
                emotion_weight=emotion_weight
            )
            
            # 3. Find Rank
            rank = float('inf')
            
            for i, (doc_id, score, emo_score) in enumerate(final_results):
                if doc_id in target_ids:
                    rank = i + 1
                    break
            
            # 4. Score
            reciprocal_rank = 0.0
            if rank != float('inf'):
                reciprocal_rank = 1.0 / rank
                if verbose: print(f"  [#{rank}] {target_title:<25} (Query: {query} + {emotion})")
            else:
                if verbose: print(f"  [FAIL] {target_title:<25} (Not in top results)")
            
            total_mrr += reciprocal_rank
            total_cases += 1

        # --- Final Calculation ---
        if total_cases > 0:
            avg_mrr = total_mrr / total_cases
            if verbose:
                print(f"{'-'*60}")
                print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
                print(f"{'='*60}")
            return avg_mrr
        else:
            return 0.0

if __name__ == "__main__":
    evaluator = IREvaluator()
    evaluator.evaluate()