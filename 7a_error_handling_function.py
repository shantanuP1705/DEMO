import requests
import json

def emotion_detector(text_to_analyze):
    url = 'https://ibm.com'
    header = {"Letters-Id": "whisk-234234234"} # Example header
    myobj = { "raw_document": { "text": text_to_analyze } }
    response = requests.post(url, json = myobj, headers=header)
    
    # Parsing the JSON response from the API
    formatted_response = json.loads(response.text)

    # Error handling for status code 400
    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # Extracting emotion scores from the response
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']
    
    # Finding the dominant emotion
    emotion_list = [anger_score, disgust_score, fear_score, joy_score, sadness_score]
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }
