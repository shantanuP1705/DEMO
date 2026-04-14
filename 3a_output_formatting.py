from transformers import pipeline

# Load model
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def emotion_detector(text):
    results = emotion_pipeline(text)[0]

    # Find highest scoring emotion
    best_emotion = max(results, key=lambda x: x['score'])

    # Format output
    output = {
        "emotion": best_emotion["label"],
        "confidence": round(best_emotion["score"], 4)
    }

    return output


# Test run
if __name__ == "__main__":
    text = "I am extremely happy today!"
    result = emotion_detector(text)

    print("Formatted Output:")
    print(result)
