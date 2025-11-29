"""
Information Retrieval System Evaluator (Segment-Level)

This script evaluates the performance of the IR system using industry-standard metrics.
It handles the mapping between "Target Books" (what we want) and "Text Segments" (what we find).

Metrics:
1. Success@10 (Hit Rate): Percentage of queries where the correct book appears in the top 10.
2. nDCG@10: Ranking quality score (rewards relevant results appearing higher).
3. MRR: Mean Reciprocal Rank (focuses on the very first relevant result).

Inputs:
- test_cases.json: JSON file containing test cases
- Pre-built IR system (via IRSystem in 5_ir_system.py)

Outputs:
- Detailed metrics per query
- Aggregated system performance report
"""

import json
import os
import math
from ir_system import IRSystem

# --- Configuration ---
TEST_CASES_FILE = "test_cases.json"
TOP_K = 10  # Evaluate top 10 results
# ---------------------

class IREvaluator:
    def __init__(self):
        # Load the search engine
        self.system = IRSystem()
        
        # Create a reverse map: "moby dick" -> "2701"
        self.title_to_id = {}
        for book_id, title in self.system.title_map.items():
            self.title_to_id[title.lower()] = book_id
            
    def find_target_book_id(self, target_title_fragment):
        """
        Finds the Book ID (e.g., '2701') for a given title (e.g., 'Moby Dick').
        """
        target_title_fragment = target_title_fragment.lower()
        matches = []
        
        # Search our title map
        for title, book_id in self.title_to_id.items():
            if target_title_fragment in title:
                matches.append(book_id)
        
        return matches

    def calculate_ndcg(self, rank, k):
        """
        Calculates nDCG score for a single query.
        """
        if rank > k:
            return 0.0
        # DCG = 1 / log2(rank + 1)
        # IDCG is 1.0 because we assume there is at least one relevant result
        return (1.0 / math.log2(rank + 1))

    def evaluate(self):
        print(f"\n{'='*60}")
        print(f"STARTING SEGMENT-LEVEL EVALUATION (k={TOP_K})")
        print(f"{'='*60}")
        
        if not os.path.exists(TEST_CASES_FILE):
            print(f"Error: {TEST_CASES_FILE} not found.")
            return

        with open(TEST_CASES_FILE, 'r') as f:
            test_cases = json.load(f)

        metrics = {
            "mrr": 0.0,
            "ndcg": 0.0,
            "success": 0.0,
            "total": 0
        }
        
        for case in test_cases:
            target_title = case['target_title']
            query = case['query']
            emotion = case['emotion']
            
            print(f"\nQuery: '{query}' + '{emotion}' (Target: {target_title})")
            
            # 1. Find the ID for the target book
            target_ids = self.find_target_book_id(target_title)
            if not target_ids:
                print(f"    [SKIPPED] Target book '{target_title}' not found in corpus.")
                continue
            
            # 2. Run System (Fetch top results)
            # Pass empty list to text_search if query is empty
            text_results = self.system.text_search(query)
            final_results = self.system.filter_by_emotion(text_results, emotion)
            
            # 3. Check Results for a Match
            rank = float('inf')
            found_segment = None
            found_book_id = None
            
            # Look through the top 50 results to find the rank
            for i, (doc_id, _, _) in enumerate(final_results[:50]):
                # doc_id is "2701_45". We split it to get "2701"
                current_book_id = doc_id.split('_')[0]
                
                if current_book_id in target_ids:
                    rank = i + 1
                    found_segment = doc_id
                    found_book_id = current_book_id
                    break
            
            # 4. Calculate Scores
            # Success@K
            is_success = 1.0 if rank <= TOP_K else 0.0
            
            # MRR
            rr = 1.0 / rank if rank != float('inf') else 0.0
            
            # nDCG@K
            ndcg = self.calculate_ndcg(rank, TOP_K)
            
            # Log this specific case
            if is_success:
                print(f"    [SUCCESS] Found at Rank #{rank} (Segment: {found_segment})")
            elif rank != float('inf'):
                print(f"    [FOUND] But rank #{rank} is > {TOP_K}")
            else:
                print(f"    [FAILURE] Not found in top 50.")
                
            metrics["success"] += is_success
            metrics["mrr"] += rr
            metrics["ndcg"] += ndcg
            metrics["total"] += 1

        # --- Final Report ---
        if metrics["total"] > 0:
            print(f"\n{'='*60}")
            print(f"FINAL PERFORMANCE REPORT (N={metrics['total']})")
            print(f"{'='*60}")
            print(f"Success Rate @{TOP_K}:           {metrics['success'] / metrics['total'] * 100:.1f}%")
            print(f"Average nDCG@{TOP_K}:            {metrics['ndcg'] / metrics['total']:.4f}")
            print(f"Mean Reciprocal Rank (MRR):  {metrics['mrr'] / metrics['total']:.4f}")
            print(f"{'='*60}")
            
            # Interpretation Helper
            score = metrics['ndcg'] / metrics['total']
            print("\nInterpretation:")
            if score > 0.5:
                print(">> Excellent! Relevant segments are appearing near the top.")
            elif score > 0.3:
                print(">> Good. The system is working, but could be tighter.")
            else:
                print(">> Low. The emotion filter might be overpowering the text relevance.")
        else:
            print("No valid test cases run.")

if __name__ == "__main__":
    evaluator = IREvaluator()
    evaluator.evaluate()