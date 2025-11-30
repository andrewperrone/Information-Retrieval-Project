"""
Plot Twist Detector - Emotion Spike Analysis

This script detects potential plot twists in books by identifying sudden spikes
in emotional intensity across text segments. It analyzes emotion volatility and
visualizes the emotional journey with flagged twist points.

Inputs:
- emotion_results.pkl: Emotion vectors per chunk from 4_chunk_level_emotion_analyzer.py
- processed_corpus.pkl: Chunk tokens for reference

Outputs:
- Console report of detected twists per book
- Matplotlib visualization: emotion timeline with spike annotations
- twist_results.pkl: Dictionary of detected twists by book_id

Process:
1. Load emotion data for all segments
2. Group segments by book
3. For each book:
   - Calculate baseline emotion statistics (mean, std dev)
   - Identify segments with unusual emotion volatility
   - Flag chunks where total emotion exceeds threshold
   - Detect emotion reversals (sentiment flips)
4. Rank and visualize twists
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration ---
EMOTION_FILE = "emotion_results.pkl"
CORPUS_FILE = "processed_corpus.pkl"
OUTPUT_FILE = "twist_results.pkl"

# Thresholds for detecting twists
VOLATILITY_THRESHOLD = 2.0  # Standard deviations above baseline
EMOTION_INTENSITY_THRESHOLD = 10  # Minimum emotion count to consider
REVERSAL_THRESHOLD = 1.0  # Correlation threshold for detecting reversal
# ---------------------

def load_emotion_data():
    """Load emotion analysis results."""
    if not os.path.exists(EMOTION_FILE):
        print(f"Error: {EMOTION_FILE} not found. Run 4_chunk_level_emotion_analyzer.py first.")
        return None
    
    with open(EMOTION_FILE, 'rb') as f:
        raw_data = pickle.load(f)
    
    # Convert list of tuples to dict: {doc_id: emotion_dict}
    emotion_data = {}
    for doc_id, emotion_dict in raw_data:
        emotion_data[doc_id] = emotion_dict
    
    return emotion_data

def load_corpus_data():
    """Load processed corpus for chunk reference."""
    if not os.path.exists(CORPUS_FILE):
        return None
    
    with open(CORPUS_FILE, 'rb') as f:
        corpus = pickle.load(f)
    
    return corpus

def group_by_book(emotion_data):
    """
    Groups emotion data by book ID.
    Returns: {book_id: [(chunk_num, emotion_dict), ...]}
    """
    books = defaultdict(list)
    
    for doc_id, emotion_dict in emotion_data.items():
        # doc_id format: "book_id_chunk_num"
        parts = doc_id.split('_')
        if len(parts) >= 2:
            book_id = parts[0]
            chunk_num = int(parts[1])
            books[book_id].append((chunk_num, emotion_dict))
    
    # Sort by chunk number
    for book_id in books:
        books[book_id].sort(key=lambda x: x[0])
    
    return books

def calculate_emotion_metrics(emotion_dict):
    """
    Convert emotion dict to a scalar metric.
    Returns: total emotion intensity
    """
    return sum(emotion_dict.values())

def detect_spikes(book_chunks):
    """
    Detects emotion spikes in a book's chunks.
    Returns: [(chunk_num, spike_score, emotion_dict), ...]
    """
    # Extract emotion intensities
    intensities = []
    chunk_nums = []
    emotion_dicts = []
    
    for chunk_num, emotion_dict in book_chunks:
        intensity = calculate_emotion_metrics(emotion_dict)
        intensities.append(intensity)
        chunk_nums.append(chunk_num)
        emotion_dicts.append(emotion_dict)
    
    intensities = np.array(intensities)
    
    # Calculate baseline statistics
    baseline_mean = np.mean(intensities)
    baseline_std = np.std(intensities)
    
    # Prevent division by zero
    if baseline_std == 0:
        baseline_std = 1
    
    # Detect spikes: z-score > threshold
    z_scores = (intensities - baseline_mean) / baseline_std
    
    spikes = []
    for i, (chunk_num, emotion_dict, z_score) in enumerate(zip(chunk_nums, emotion_dicts, z_scores)):
        if z_score > VOLATILITY_THRESHOLD:
            spikes.append((chunk_num, z_score, emotion_dict, intensities[i]))
    
    return spikes, intensities, baseline_mean, baseline_std, chunk_nums

def detect_emotion_reversals(book_chunks):
    """
    Detects sudden shifts in dominant emotion.
    Returns: [(chunk_num, reversal_score, from_emotion, to_emotion), ...]
    """
    if len(book_chunks) < 2:
        return []
    
    reversals = []
    emotions = ['joy', 'fear', 'anger', 'surprise', 'sadness', 'disgust', 'trust', 'anticipation']
    
    for i in range(1, len(book_chunks)):
        prev_chunk_num, prev_emotions = book_chunks[i-1]
        curr_chunk_num, curr_emotions = book_chunks[i]
        
        # Find dominant emotion in each
        prev_dominant = max(emotions, key=lambda e: prev_emotions.get(e, 0))
        curr_dominant = max(emotions, key=lambda e: curr_emotions.get(e, 0))
        
        prev_score = prev_emotions.get(prev_dominant, 0)
        curr_score = curr_emotions.get(curr_dominant, 0)
        
        # Flag if dominant emotion changes AND intensity is significant
        if prev_dominant != curr_dominant and (prev_score > 0 or curr_score > 0):
            reversal_strength = abs(curr_score - prev_score)
            if reversal_strength > REVERSAL_THRESHOLD:
                reversals.append((curr_chunk_num, reversal_strength, prev_dominant, curr_dominant))
    
    return reversals

def visualize_book_emotions(book_id, book_chunks, spikes, intensities, baseline_mean, baseline_std, chunk_nums, reversals=None):
    """
    Creates a visualization of emotion timeline with spike and reversal annotations.
    """
    if reversals is None:
        reversals = []
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot emotion intensity
    ax.plot(chunk_nums, intensities, 'b-', linewidth=2, label='Emotion Intensity')
    ax.axhline(baseline_mean, color='g', linestyle='--', linewidth=1, label=f'Baseline Mean ({baseline_mean:.1f})')
    ax.axhline(baseline_mean + VOLATILITY_THRESHOLD * baseline_std, color='r', linestyle='--', linewidth=1, 
               label=f'Spike Threshold')
    
    # Highlight spike points (red stars)
    spike_nums = [s[0] for s in spikes]
    spike_scores = [intensities[chunk_nums.index(s[0])] if s[0] in chunk_nums else 0 for s in spikes]
    
    ax.scatter(spike_nums, spike_scores, color='red', s=100, marker='*', zorder=5, label='Emotion Spikes')
    
    # Annotate spike points
    for spike_num, spike_z_score, emotion_dict, intensity in spikes:
        dominant_emotion = max(emotion_dict, key=emotion_dict.get) if emotion_dict else 'unknown'
        ax.annotate(f'Spike\n({dominant_emotion})', 
                   xy=(spike_num, intensity), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Highlight reversal points (blue circles)
    if reversals:
        reversal_nums = [r[0] for r in reversals]
        reversal_scores = [intensities[chunk_nums.index(r[0])] if r[0] in chunk_nums else 0 for r in reversals]
        
        ax.scatter(reversal_nums, reversal_scores, color='blue', s=80, marker='o', zorder=4, 
                  label='Emotion Reversals', edgecolors='darkblue', linewidth=1.5)
        
        # Annotate reversal points
        for reversal_num, strength, from_emo, to_emo in reversals:
            intensity_val = intensities[chunk_nums.index(reversal_num)] if reversal_num in chunk_nums else baseline_mean
            ax.annotate(f'Reversal\n({from_emo}→{to_emo})', 
                       xy=(reversal_num, intensity_val), 
                       xytext=(-50, -20), 
                       textcoords='offset points',
                       fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax.set_xlabel('Chunk Number', fontsize=12)
    ax.set_ylabel('Total Emotion Intensity', fontsize=12)
    ax.set_title(f'Plot Twist Detection - Book {book_id}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'twist_visualization_{book_id}.png', dpi=100)
    print(f"  Saved visualization: twist_visualization_{book_id}.png")
    plt.close()

def analyze_all_books():
    """
    Main analysis function.
    """
    print("Loading emotion data...")
    emotion_data = load_emotion_data()
    if not emotion_data:
        return
    
    corpus_data = load_corpus_data()
    
    print(f"Analyzing {len(emotion_data)} total segments...")
    
    books = group_by_book(emotion_data)
    print(f"Grouped into {len(books)} books.\n")
    
    all_twists = {}
    
    for book_id in sorted(books.keys()):
        book_chunks = books[book_id]
        
        print(f"\n{'='*60}")
        print(f"Book ID: {book_id}")
        print(f"Total Chunks: {len(book_chunks)}")
        print(f"{'='*60}")
        
        # Detect spikes
        spikes, intensities, baseline_mean, baseline_std, chunk_nums = detect_spikes(book_chunks)
        
        print(f"\nEmotional Baseline:")
        print(f"  Mean Intensity:  {baseline_mean:.2f}")
        print(f"  Std Dev:         {baseline_std:.2f}")
        print(f"  Spike Threshold: {baseline_mean + VOLATILITY_THRESHOLD * baseline_std:.2f}")
        
        # Detect reversals
        reversals = detect_emotion_reversals(book_chunks)
        
        # Report findings
        print(f"\n--- Detected Twists ---")
        if spikes:
            print(f"Emotion Spikes Found: {len(spikes)}")
            for chunk_num, z_score, emotion_dict, intensity in spikes:
                dominant_emotion = max(emotion_dict, key=emotion_dict.get)
                print(f"  Chunk #{chunk_num}: Z-Score={z_score:.2f}, Dominant={dominant_emotion} ({intensity:.1f})")
        else:
            print("No emotion spikes detected.")
        
        if reversals:
            print(f"\nEmotion Reversals Found: {len(reversals)}")
            for chunk_num, strength, from_emo, to_emo in reversals:
                print(f"  Chunk #{chunk_num}: {from_emo} → {to_emo} (Strength: {strength:.2f})")
        else:
            print("No emotion reversals detected.")
        
        # Visualize
        if len(book_chunks) > 5:  # Only visualize if book has enough chunks
            visualize_book_emotions(book_id, book_chunks, spikes, intensities, 
                                   baseline_mean, baseline_std, chunk_nums, reversals)
        
        # Store results
        all_twists[book_id] = {
            'spikes': spikes,
            'reversals': reversals,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'total_chunks': len(book_chunks)
        }
    
    # Save results
    print(f"\n\nSaving twist detection results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_twists, f)
    
    print("Complete!")
    

if __name__ == "__main__":
    analyze_all_books()
