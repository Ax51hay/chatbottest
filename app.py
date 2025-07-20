# app.py
from flask import Flask, render_template, request, session
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.spatial.distance import cosine
from nltk.stem.snowball import PorterStemmer
import numpy as np
import re
import warnings

app = Flask(__name__)
app.secret_key = "super-secret-key"

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Mood dataset
moods = {
    "happy": [
        "I am feeling good today",
        "I am well thank you",
        "I'm good",
        "I'm feeling happy",
        "I'm so happy",
        "I'm so excited",
        "Today I am feeling happy",
        "I'm feeling joyful",
        "Things are going really well",
        "I'm feeling optimistic today",
        "I'm in a great mood!",
        "Life is good right now",
        "I'm quite cheerful",
        "I feel content and peaceful",
        "I'm full of energy today",
        "I'm feeling positive",
        "Everything feels right today"
    ],

    "sad": [
        "I'm feeling down",
        "Today hasn't been great",
        "I'm not doing so well",
        "I'm a bit sad",
        "I feel really low today",
        "I've been feeling blue",
        "I'm upset",
        "I'm having a hard day",
        "I just feel empty",
        "Things have been rough",
        "I feel like crying",
        "I'm emotionally drained",
        "I feel hopeless",
        "I'm feeling a little lost",
        "My mood is really low",
        "I'm feeling lonely",
        "I'm not in the best headspace right now"
    ],

    "angry": [
        "I'm really annoyed",
        "I'm frustrated today",
        "Things are making me angry",
        "I'm feeling irritated",
        "I'm mad right now",
        "I'm seriously pissed off",
        "I can't deal with this!",
        "Everything is getting on my nerves",
        "I'm so fed up",
        "I'm losing my patience",
        "People keep pushing my buttons",
        "I'm just in a bad mood",
        "I'm raging inside",
        "I feel so tense and angry",
        "This day has been infuriating",
        "I'm seething right now",
        "My temper is running thin"
    ]
}

# Preprocessing
def preprocess(user_input):
    p_stemmer = PorterStemmer()
    tokens = re.findall(r'\b\w+\b', user_input.lower())
    stemmed_tokens = [p_stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# Prepare training data
processed_moods = {intent: [preprocess(p) for p in phrases] for intent, phrases in moods.items()}
all_phrases = [phrase for phrases in processed_moods.values() for phrase in phrases]
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
count_matrix = count_vectorizer.fit_transform(all_phrases)
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
tfidf_vectors = tfidf_matrix.toarray()

# Match user mood
def mood_matching(user_input):
    input_vector = tfidf_transformer.transform(count_vectorizer.transform([preprocess(user_input)])).toarray()[0]
    similarities = [1 - cosine(input_vector, vec) for vec in tfidf_vectors]

    if np.isnan(max(similarities)) or max(similarities) < 0.4:
        return None
    best_match_index = np.argmax(similarities)

    intent_lengths = [len(phrases) for phrases in processed_moods.values()]
    cumulative_lengths = np.cumsum(intent_lengths)
    for i, length in enumerate(cumulative_lengths):
        if best_match_index < length:
            return list(processed_moods.keys())[i]

# Web routes
@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        name = request.form["name"].strip().capitalize()
        message = request.form["message"]
        session["name"] = name

        mood = mood_matching(message)
        if mood == "happy":
            response = f"I'm so glad to hear you're feeling happy today, {name}!"
        elif mood == "sad":
            response = f"I'm sorry to hear you're feeling sad, {name}. Maybe I can help."
        elif mood == "angry":
            response = f"It's okay to feel angry, {name}. Let's talk about it."
        else:
            response = "I'm not sure I understand — could you try expressing how you feel again?"

        return render_template("index.html", name=name, message=message, response=response)

    return render_template("index.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
