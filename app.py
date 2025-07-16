from flask import Flask, render_template, request, jsonify
from web_bot import get_name, mood_matching, handle_confirmation
import datetime
import csv
import nltk

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    mood = mood_matching(user_input)
    
    if mood == "happy":
        response = "I'm glad to hear you're feeling good today!"
    elif mood == "sad":
        response = "I'm sorry to hear that. Would you like to talk about it?"
    elif mood == "angry":
        response = "That sounds frustrating. I'm here for you."
    else:
        response = "I'm not quite sure how you're feeling. Can you say that another way?"

    log_interaction(user_input, mood)
    return jsonify({"response": response})

def log_interaction(user_input, mood):
    with open("interaction_logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now(), user_input, mood])

if __name__ == "__main__":
    app.run(debug=True)
