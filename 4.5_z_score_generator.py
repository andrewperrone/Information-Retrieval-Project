import pickle
import numpy as np
import os

# --- Configuration ---
EMOTION_FILE = "emotion_results.pkl"
INDEX_FILE = "search_index.pkl" # Needed for doc lengths
OUTPUT_FILE = "emotion_stats.pkl"
# ---------------------

def generate_z_scores():
    print(f"Loading data to calculate corpus statistics...")
    
    if not os.path.exists(EMOTION_FILE) or not os.path.exists(INDEX_FILE):
        print("Error: Missing data files.")
        return

    # Load Emotion Data
    with open(EMOTION_FILE, 'rb') as f:
        # Format: [(doc_id, {vector}), ...]
        raw_data = pickle.load(f)
        emotion_data = {item[0]: item[1] for item in raw_data}

    # Load Doc Lengths
    with open(INDEX_FILE, 'rb') as f:
        index_data = pickle.load(f)
        doc_lengths = index_data.get('doc_lengths', {})

    # Define Emotions
    emotions = ['joy', 'sadness', 'anger', 'fear', 'trust', 
                'disgust', 'anticipation', 'surprise']
    
    # 1. Collect all densities for every emotion
    corpus_values = {emo: [] for emo in emotions}
    
    print(f"Processing {len(emotion_data)} documents...")
    
    for doc_id, vector in emotion_data.items():
        length = doc_lengths.get(doc_id, 1)
        for emo in emotions:
            count = vector.get(emo, 0)
            density = count / length
            corpus_values[emo].append(density)

    # 2. Calculate Mean and StdDev for each emotion
    stats = {}
    print("\n--- Corpus Statistics (Density) ---")
    print(f"{'Emotion':<15} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 45)
    
    for emo in emotions:
        values = np.array(corpus_values[emo])
        mean = np.mean(values)
        std = np.std(values)
        
        stats[emo] = {'mean': mean, 'std': std}
        
        print(f"{emo:<15} | {mean:.5f}    | {std:.5f}")

    # 3. Save Stats
    try:
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(stats, f)
        print(f"\nStats saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving stats: {e}")

if __name__ == "__main__":
    generate_z_scores()