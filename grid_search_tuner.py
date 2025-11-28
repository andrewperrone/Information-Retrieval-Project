import time
import importlib.util
spec = importlib.util.spec_from_file_location("ir_evaluator", "6_ir_evaluator.py")
ir_evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir_evaluator)
IREvaluator = ir_evaluator.IREvaluator
# Above replaces this faulty import:
# from ir_evaluator import IREvaluator


# --- Configuration ---
# Define the range of weights to test.
# We test from 0.0 (ignore feature) to 3.0 (heavy emphasis)
WEIGHT_RANGE = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

def run_grid_search():
    print("Initializing System for Grid Search...")
    # Load the system once so we don't reload pickles every loop
    evaluator = IREvaluator()
    
    results = []
    total_combinations = len(WEIGHT_RANGE) * len(WEIGHT_RANGE)
    print(f"Testing {total_combinations} combinations of weights...\n")
    
    start_time = time.time()
    count = 0
    
    print(f"{'Text W':<10} | {'Emotion W':<10} | {'MRR Score':<10}")
    print("-" * 36)
    
    # --- The Grid Loop ---
    for text_w in WEIGHT_RANGE:
        for emo_w in WEIGHT_RANGE:
            # Skip the case where both are 0 (no score)
            if text_w == 0 and emo_w == 0:
                continue
                
            # Run evaluation in silent mode (verbose=False)
            score = evaluator.evaluate(text_weight=text_w, emotion_weight=emo_w, verbose=False)
            
            # Store result
            results.append({
                'text_w': text_w,
                'emo_w': emo_w,
                'score': score
            })
            
            print(f"{text_w:<10} | {emo_w:<10} | {score:.4f}")
            count += 1

    total_time = time.time() - start_time
    
    # --- Analysis ---
    # Sort by Score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*40)
    print("   GRID SEARCH RESULTS (Top 5)")
    print("="*40)
    
    for i, res in enumerate(results[:5]):
        print(f"Rank #{i+1}: MRR {res['score']:.4f}")
        print(f"   Weights -> Text: {res['text_w']}, Emotion: {res['emo_w']}")
        print("-" * 40)
        
    best = results[0]
    print(f"\nTime Taken: {total_time:.2f}s")
    print(f"\nRECOMMENDATION: Set TEXT_WEIGHT = {best['text_w']} and EMOTION_WEIGHT = {best['emo_w']}")

if __name__ == "__main__":
    run_grid_search()