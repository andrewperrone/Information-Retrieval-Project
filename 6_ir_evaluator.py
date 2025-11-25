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
from ir_system import IRSystem

# --- Configuration ---
TEST_CASES_FILE = "test_cases.json"
# ---------------------

class IREvaluator:
    def __init__(self):
        # Load the actual system
        self.system = IRSystem()
        self.corpus_files = self.system.doc_ids
        
    def find_target_doc_id(self, target_title_fragment):
        """
        Fuzzy search to find the actual filename (doc_id) based on a user's
        simple book title (e.g., "Moby Dick" -> "2701_Moby_Dick_Or_The_Whale.txt").
        """
        target_title_fragment = target_title_fragment.lower()
        matches = []
        
        for doc_id in self.corpus_files:
            # Normalize filename for search (replace underscores with spaces)
            clean_name = doc_id.lower().replace("_", " ")
            if target_title_fragment in clean_name:
                matches.append(doc_id)
        
        if not matches:
            return None
        
        # If multiple matches (e.g., duplicates), return the list.
        # Any of these being high ranked counts as success.
        return matches

    def evaluate(self):
        print(f"\n{'='*60}")
        print(f"STARTING AUTOMATED EVALUATION")
        print(f"{'='*60}")
        
        if not os.path.exists(TEST_CASES_FILE):
            print(f"Error: {TEST_CASES_FILE} not found.")
            return

        with open(TEST_CASES_FILE, 'r') as f:
            test_cases = json.load(f)

        total_mrr = 0
        total_cases = 0
        
        for case in test_cases:
            target_title = case['target_title']
            query = case['query']
            emotion = case['emotion']
            
            print(f"\n--- Case: '{target_title}' ---")
            print(f"    Query: '{query}' + '{emotion}'")
            
            # 1. Identify the Target File(s)
            target_ids = self.find_target_doc_id(target_title)
            if not target_ids:
                print(f"    [SKIPPED] Target book '{target_title}' not found in corpus.")
                continue
            
            # 2. Run the System
            # We get the top 50 results to check deeper rankings
            text_results = self.system.text_search(query)
            final_results = self.system.filter_by_emotion(text_results, emotion)
            
            # 3. Find Rank
            rank = float('inf')
            found_id = None
            
            for i, (doc_id, score, emo_score) in enumerate(final_results):
                # Check if this result is one of our acceptable targets
                if doc_id in target_ids:
                    rank = i + 1 # 1-based index
                    found_id = doc_id
                    break
            
            # 4. Score (Reciprocal Rank)
            # Score = 1/Rank. 
            # Rank 1 = 1.0, Rank 2 = 0.5, Rank 5 = 0.2, Not Found = 0.0
            reciprocal_rank = 0.0
            if rank != float('inf'):
                reciprocal_rank = 1.0 / rank
                print(f"    [SUCCESS] Found at Rank #{rank} ({found_id})")
            else:
                print(f"    [FAILURE] Not found in top results.")
                
            print(f"    Score: {reciprocal_rank:.4f}")
            
            total_mrr += reciprocal_rank
            total_cases += 1

        # --- Final Report ---
        if total_cases > 0:
            avg_mrr = total_mrr / total_cases
            print(f"\n{'='*60}")
            print(f"EVALUATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total Cases Run: {total_cases}")
            print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
            print(f"  (1.0 = Perfect, >0.5 = Good, <0.1 = Poor)")
            print(f"{'='*60}")
        else:
            print("No valid test cases found.")

if __name__ == "__main__":
    evaluator = IREvaluator()
    evaluator.evaluate()