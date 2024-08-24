## Overview (Version 2 - Still under development)

This Mental Health ChatBot is a web-based application designed to assist users by providing supportive and empathetic responses related to mental health. The chatbot uses Natural Language Processing (NLP) techniques to understand and respond to user inputs, aiming to offer guidance, comfort, and general information about mental well-being. I am gathering large data sets, and currently fine-tuning the data using tensorflow, I am considering fine tuning as in this process ill learn all these concepts, otherwise if anyone is planning to build a similar chat app i would recommend using pre-trained model like GPT 2 using open AI api key.

## Features

- Real-time chat with NLP-based responses
- Facial expression analysis for better emotion detection
- Image upload and session history

--------------------------------------------------------------------------------------------------

## Technologies Used

- **Languages**: Python, HTML/CSS
- **Frameworks**: Flask, TensorFlow, Keras
- **Libraries**: NLTK, OpenCV

--------------------------------------------------------------------------------------------------

### Change Logs- August

1. Trained the bot on a large Dataset, it replies to a lot of new prompts now.
2. Added a option to create new chats, old chats gets saved!
4. Added Session history system, chats are stored in terms of session in sidebar.
5. Added a feature to delete pre existing Sessions.
6. Added a feature to edit a sent text.

https://github.com/user-attachments/assets/6b8da763-715c-443e-9cac-178995852429

--------------------------------------------------------------------------------------------------

### old video

https://github.com/user-attachments/assets/048d2391-13d1-47a9-8ea0-241420356ece

--------------------------------------------------------------------------------------------------

**STEP BY STEP GUIDE**

### Prerequisites

- Python 3.7 or higher
- [VS Code](https://code.visualstudio.com/)

--------------------------------------------------------------------------------------------------

### Steps to Git Clone

1. Clone the repo: `git clone https://github.com/pranavvss/Mental-Health-ChatBot.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

--------------------------------------------------------------------------------------------------

**Install Required Libraries**

You need several Python libraries to build the chatbot. Let’s install them

- Install Flask [Read Documentation](https://flask.palletsprojects.com/_/downloads/en/1.1.x/pdf/) : Flask is a lightweight web framework used to build web applications.
- Install TensorFlow and Keras [Read Documentation](https://www.tensorflow.org/guide/basics) : TensorFlow is a machine learning framework. Keras is a high-level API for building and training neural networks, which comes as part of TensorFlow.
- Install NLTK (Natural Language Toolkit) [Read Documentation](https://www.nltk.org/) : NLTK is used for processing textual data.
- Install Other Dependencies: You will also need Numpy, Pickle, and other essential libraries.
  
```
pip install Flask tensorflow nltk numpy pickle-mixin
```

--------------------------------------------------------------------------------------------------

### Explanation of the Code

--------------------------------------------------------------------------------------------------

# Step 5: Create Project Files

Now you will create the necessary files for your project.
1. Create intents.json (Data File):

This file contains the training data for your chatbot, including possible user inputs and corresponding bot responses. There is nothing much to explain, I feel this part is self explanatory, As you wish you can keep on adding more data, remember everytime you add new set of data make sure to save the indent file them train the data (This step is discussed below)

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "How are you?"],
      "responses": ["Hello!", "Hi there!", "Hey!"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": ["Goodbye!", "See you later!", "Bye! Have a great day!"]
    },
    {
      "tag": "mental_health",
      "patterns": [
        "I feel sad today",
        "How can someone feel better or fix his depression?",
        "I am depressed",
        "I need help with my mental health"
      ],
      "responses": [
        "I'm sorry you're feeling this way. How can I assist you?",
        "Practicing mindfulness, engaging in regular exercise, and maintaining a balanced diet can improve your mental health.",
        "It's important to seek help from a professional if you're struggling."
      ]
    },
    {
      "tag": "thanks",
      "patterns": ["Thanks", "Thank you", "That's helpful"],
      "responses": ["You're welcome!", "No problem!", "Happy to help!"]
    }
  ]
}
```

>[!NOTE]
> According to your wish you can add more data, But everytime to add another set of data youll have to retrain the whole data (In upcoming steps ill mention how to train data).

**Just a glimpse (After adding another set of data in intents.json youll have to run train_chatbot.py to train the new model. **

![Screenshot 2024-08-19 020429](https://github.com/user-attachments/assets/fae76fdd-a1d7-459e-8601-ad075b8da632)


--------------------------------------------------------------------------------------------------

**2. Create train_chatbot.py (Training Script):** This script will train the chatbot model using the data in intents.json.
```python
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import pickle

lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_words = ['?', '!']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # add documents in the corpus
        documents.append((word_list, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # bag of words
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # output is '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert training data to np.array
random.shuffle(training)
training = np.array(training)

# Create training and testing lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create model - 3 layers
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model and data
model.save('chatbot_model.h5', hist)
print("model created")
```

--------------------------------------------------------------------------------------------------

**3. Create app.py (Flask Web Application):** This script sets up the Flask web server and handles incoming requests.
```python
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import nltk
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model):
    # filter below  threshold predictions
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    msg = request.form['msg']
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)
```
--------------------------------------------------------------------------------------------------

> [!NOTE]
> At this point your chatbot is ready to reply to all your questions. But if you want to host this chatbot to a webapp page follow the below mentioned process.

--------------------------------------------------------------------------------------------------

**Step 6: Create Frontend Files**

**1. Create chat.html (Chat Interface)**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
    <title>Mental Health ChatBot</title>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h2>Mental Health ChatBot</h2>
            <p>Share with me! Whatever you want. I am here for you.</p>
        </div>
        <div id="chatbox" class="chatbox"></div>
        <div class="chat-input">
            <input id="userInput" type="text" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
```

--------------------------------------------------------------------------------------------------

**2. Create style.css (Stylesheet): This file contains the CSS to style your chat interface.**
```
body, html {
    height: 100%;
    margin: 0;
    background: linear-gradient(to right, rgb(38, 51, 61), rgb(50, 55, 65), rgb(33, 33, 78));
    font-family: Arial, Helvetica, sans-serif;
}

.chat-container {
    width: 400px;
    margin: auto;
    position: relative;
    top: 50%;
    transform: translateY(-50%);
    border-radius: 10px;
    overflow: hidden;
}

.chat-header {
    background-color: #444;
    padding: 10px;
    color: white;
    text-align: center;
}

.chatbox {
    height: 300px;
    background-color: #222;
    padding: 10px;
    overflow-y: scroll;
    color: white;
}

.chat-input {
    display: flex;
}

.chat-input input {
    width: 80%;
    padding: 10px;
    border: none;
    border-top-left-radius: 10px;
    border-bottom-left-radius: 10px;
}

.chat-input button {
    width: 20%;
    background-color: #444;
    border: none;
    color: white;
    cursor: pointer;
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
}
```
--------------------------------------------------------------------------------------------------

**3. Create script.js (JavaScript File): This file contains the JavaScript to handle sending and receiving messages.**
```js
function sendMessage() {
    const userInput = document.getElementById("userInput").value;

    if (userInput) {
        const chatbox = document.getElementById("chatbox");
        const userMessage = `<div class="chat-message"><span class="user">${userInput}</span></div>`;
        chatbox.innerHTML += userMessage;

        fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'msg=' + encodeURIComponent(userInput)
        })
        .then(response => response.json())
        .then(data => {
            const botMessage = `<div class="chat-message"><span class="bot">${data}</span></div>`;
            chatbox.innerHTML += botMessage;
            chatbox.scrollTop = chatbox.scrollHeight;
        });

        document.getElementById("userInput").value = '';
    }
}
```

--------------------------------------------------------------------------------------------------

**Step 7: Train the Model**

- Open your terminal and run the training script
```
python train_chatbot.py
```
--------------------------------------------------------------------------------------------------

**Step 8: Run the Flask Application**

- In your terminal, run, This will start the Flask development server on http://127.0.0.1:5000.
```
python app.py
```
--------------------------------------------------------------------------------------------------

You can chat with the bot now
--------------------------------------------------------------------------------------------------
Thankyou !! 
