from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import cv2
import numpy as np
import json
import random
import re

app = Flask(__name__)

# Load intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

def classify_intent(user_text):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Check if the pattern is found in the user's text (case insensitive)
            if re.search(r'\b' + re.escape(pattern) + r'\b', user_text, re.IGNORECASE):
                return random.choice(intent['responses'])
    return "I'm here for you. How can I assist you?"

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_text = request.form['msg']
    return str(classify_intent(user_text))

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    try:
        # Analyze the image using DeepFace to detect emotion
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion
        dominant_emotion = analysis['dominant_emotion']
        return jsonify({"mood": dominant_emotion})

    except ValueError as e:
        # Handle the case where no face is detected
        return jsonify({"mood": "no_face", "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
