"""
Text Retrieval Normalization Experiment

This script runs an A/B/C/D/E test to scientifically determine the optimal mathematical 
normalization strategy for the text search engine. It compares different ways of handling 
document length bias to see which yields the highest retrieval accuracy (MRR).

Action:
It overrides the standard scoring logic of the IR System to test five specific algorithms:
1. None (Raw Dot Product): Favors long documents.
2. Linear (1/Length): Favors short documents.
3. Square Root (1/sqrt(Length)): Pivoted normalization (balances short/long).
4. Logarithmic (1/log(Length)): Gentle penalty.
5. Cosine (1/L2 Norm): Standard Vector Space Model normalization.

Connection:
This is an experimental utility. It does not modify the saved index files. 
It uses the 'test_cases.json' ground truth to evaluate performance and inform 
decisions on how to write the final 'ir_system.py' logic.

Inputs:
- 'search_index.pkl' (via IRSystem)
- 'test_cases.json' (The Golden Standard queries)

Outputs:
- A console "Leaderboard" ranking the 5 methods by Mean Reciprocal Rank (MRR)
- Comparison of "#1 Hits" (how often the target book was the absolute top result)

Process:
1. Initializes the IR System and pre-calculates True Euclidean (L2) Norms 
   from the TF-IDF data (essential for testing pure Cosine Similarity).
2. Loads the test cases.
3. Iterates through each normalization method.
4. Runs all test queries using the specific mathematical formula.
5. Scores the results based on where the target book appeared in the list.
6. Prints a comparative table to identify the superior algorithm.
"""

import json
import os
import numpy as np
from collections import defaultdict
import importlib.util
spec = importlib.util.spec_from_file_location("ir_system", "5_ir_system.py")
ir_system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir_system)
IRSystem = ir_system.IRSystem
# Above replaces this faulty import:
# from 5_ir_system import IRSystem
spec1 = importlib.util.spec_from_file_location("ir_evaluator", "6_ir_evaluator.py")
ir_evaluator = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(ir_evaluator)
IREvaluator = ir_evaluator.IREvaluator
# Above replaces this faulty import:
# from 6_ir_evaluator import IREvaluator

class ExperimentalIRSystem(IRSystem):
    """
    A subclass of IRSystem that allows us to swap the scoring math
    without changing the underlying index.
    """
    def __init__(self):
        super().__init__()
        # We calculate the True L2 Norms on startup for scientific accuracy
        self.doc_l2_norms = {}
        self._precalculate_true_l2_norms()

    def _precalculate_true_l2_norms(self):
        """
        Reconstructs the exact Euclidean (L2) Norm for every document vector
        using the TF and IDF data available in the index.
        Formula: sqrt( sum( (tf * idf)^2 ) )
        """
        print("Pre-calculating True L2 Norms for Pure Cosine Similarity (this may take a moment)...")
        norm_sq_accumulator = defaultdict(float)
        
        # Iterate over the entire inverted index to sum squared weights per document
        for term, doc_dict in self.inverted_index.items():
            idf = self.idf_scores.get(term, 0)
            # Pre-calculate idf^2 to save time in the inner loop
            idf_sq = idf ** 2
            
            for doc_id, tf in doc_dict.items():
                # Add (TF * IDF)^2 to the document's running total
                term_weight_sq = (tf ** 2) * idf_sq
                norm_sq_accumulator[doc_id] += term_weight_sq
        
        # Take the square root to get the final L2 Norm
        for doc_id, val in norm_sq_accumulator.items():
            self.doc_l2_norms[doc_id] = np.sqrt(val)
            
        print(f"Calculated L2 Norms for {len(self.doc_l2_norms)} documents.")

    def text_search_variant(self, query_text, method="sqrt"):
        tokens = self.process_query(query_text)
        if not tokens: return []
            
        doc_scores = defaultdict(float)
        
        # 1. Calculate Raw Dot Product (TF * IDF)
        for token in tokens:
            if token in self.inverted_index:
                idf = self.idf_scores.get(token, 0)
                matching_docs = self.inverted_index[token]
                for doc_id, tf in matching_docs.items():
                    doc_scores[doc_id] += tf * idf
        
        # 2. Apply chosen Normalization
        final_scores = []
        for doc_id, raw_score in doc_scores.items():
            # Get document length (Total Words)
            length = self.doc_lengths.get(doc_id, 1)
            
            denominator = 1.0
            
            if method == "none":
                # Standard Dot Product (Favors Long Docs)
                denominator = 1.0 
                
            elif method == "linear":
                # Divide by Length (Favors Short Docs)
                denominator = float(length)
                
            elif method == "sqrt":
                # Pivoted Normalization (Balances Short/Long)
                # This is the approximation we used in the main system
                denominator = np.sqrt(length)
                
            elif method == "log":
                # Logarithmic (Another common standard)
                denominator = np.log(length + 1)

            elif method == "cosine":
                # Pure Cosine Similarity Normalization
                # Divides by the True Euclidean Norm (L2) calculated at startup
                denominator = self.doc_l2_norms.get(doc_id, 1.0)

            if denominator < 1: denominator = 1
            
            normalized_score = raw_score / denominator
            final_scores.append((doc_id, normalized_score))
            
        return sorted(final_scores, key=lambda x: x[1], reverse=True)

def run_normalization_test():
    print("--- Loading System ---")
    # Initialize our experimental system (now calculates L2 norms on load)
    system = ExperimentalIRSystem()
    
    # Initialize evaluator helper to find targets
    # We use the existing logic from your evaluator
    helper = IREvaluator()
    
    with open("test_cases.json", 'r') as f:
        test_cases = json.load(f)

    # We now test "cosine" using the True L2 Norms
    methods = ["none", "linear", "sqrt", "log", "cosine"]
    results = {m: {'mrr': 0.0, 'top1_hits': 0} for m in methods}
    
    print(f"\nRunning Experiment on {len(test_cases)} test cases...")
    print(f"Comparing Methods: {', '.join(methods)}\n")

    for method in methods:
        print(f"Testing Method: [{method.upper()}]...")
        total_reciprocal_rank = 0
        
        for case in test_cases:
            target_title = case['target_title']
            query = case['query']
            
            # Find valid doc IDs for this book
            target_ids = helper.find_target_doc_id(target_title)
            if not target_ids: continue

            # Run Search using the specific method
            # We assume Text Weight 1.0, Emotion Weight 0.0 (Pure Text Test)
            ranked_results = system.text_search_variant(query, method=method)
            
            # Calculate Rank
            rank = float('inf')
            for i, (doc_id, score) in enumerate(ranked_results):
                if doc_id in target_ids:
                    rank = i + 1
                    break
            
            # Score
            if rank != float('inf'):
                total_reciprocal_rank += (1.0 / rank)
                if rank == 1:
                    results[method]['top1_hits'] += 1
        
        # Calculate Average MRR for this method
        results[method]['mrr'] = total_reciprocal_rank / len(test_cases)

    # --- Final Report ---
    print("\n" + "="*50)
    print("   TEXT RETRIEVAL ALGORITHM LEADERBOARD")
    print("="*50)
    print(f"{'Method':<10} | {'MRR Score':<10} | {'#1 Hits':<10}")
    print("-" * 40)
    
    # Sort by MRR
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mrr'], reverse=True)
    
    for method, stats in sorted_results:
        print(f"{method.upper():<10} | {stats['mrr']:.4f}     | {stats['top1_hits']}")
    print("-" * 40)

if __name__ == "__main__":
    run_normalization_test()