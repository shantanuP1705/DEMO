from transformers import pipeline

# Load emotion detection pipeline
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def detect_emotion(text):
    results = emotion_pipeline(text)[0]
    
    # Sort emotions by score (highest first)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    return sorted_results

if __name__ == "__main__":
    print("=== Emotion Detection System ===")
    
    while True:
        user_input = input("\nEnter text (or type 'exit'): ")
        
        if user_input.lower() == "exit":
            break
        
        emotions = detect_emotion(user_input)
        
        print("\nDetected Emotions:")
        for emo in emotions[:3]:  # Top 3 emotions
            print(f"{emo['label']}: {emo['score']:.4f}")
