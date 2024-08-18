from flask import Flask, render_template, request, jsonify
from chatbot import predict_class, get_response, model, intents

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    msg = request.form['msg']
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res

if __name__ == "__main__":
    app.run(debug=True)
