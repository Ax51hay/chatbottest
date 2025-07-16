#!/usr/bin/env python3

import nltk
import re
import numpy as np
import warnings
from nltk.stem.snowball import PorterStemmer
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.spatial.distance import cosine

# Downloads
nltk.download('punkt')
nltk.download('words')

# Suppress warnings for cosine divide-by-zero
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

english_words = set(words.words())

# Mood Training Data
moods = {
    "happy": [
        "I am feeling good today", "I am well thank you", "I'm good", "I'm feeling happy",
        "I'm so happy", "I'm so excited", "Today I am feeling happy", "I'm feeling joyful",
        "Things are going really well", "I'm feeling optimistic today", "I'm in a great mood!",
        "Life is good right now", "I'm quite cheerful", "I feel content and peaceful",
        "I'm full of energy today", "I'm feeling positive", "Everything feels right today"
    ],
    "sad": [
        "I'm feeling down", "Today hasn't been great", "I'm not doing so well", "I'm a bit sad",
        "I feel really low today", "I've been feeling blue", "I'm upset", "I'm having a hard day",
        "I just feel empty", "Things have been rough", "I feel like crying", "I'm emotionally drained",
        "I feel hopeless", "I'm feeling a little lost", "My mood is really low",
        "I'm feeling lonely", "I'm not in the best headspace right now"
    ],
    "angry": [
        "I'm really annoyed", "I'm frustrated today", "Things are making me angry",
        "I'm feeling irritated", "I'm mad right now", "I'm seriously pissed off",
        "I can't deal with this!", "Everything is getting on my nerves", "I'm so fed up",
        "I'm losing my patience", "People keep pushing my buttons", "I'm just in a bad mood",
        "I'm raging inside", "I feel so tense and angry", "This day has been infuriating",
        "I'm seething right now", "My temper is running thin"
    ]
}

name = ""
matched_mood = ""

# --- Preprocessing ---
def preprocess(user_input):
    p_stemmer = PorterStemmer()

    # Remove apostrophes and lowercase
    cleaned = re.sub(r"[’']", "", user_input.lower())

    # Optional: remove all punctuation (comment this line out if you want to keep it)
    # cleaned = re.sub(r"[^\w\s]", "", cleaned)

    tokens = nltk.word_tokenize(cleaned)
    stemmed_tokens = [p_stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# --- Mood Vector Preprocessing ---
processed_moods = {intent: [preprocess(phrase) for phrase in phrases] for intent, phrases in moods.items()}

# --- Helpers ---
def is_english_word(word):
    return word.lower() in english_words

def handle_confirmation(user_input):
    while True:
        if "yes" in user_input.lower() and "no" in user_input.lower():
            print("Bot: I detected both 'yes' and 'no'. Could you confirm with just 'Yes' or 'No'?")
            user_input = input("You: ")
        elif "yes" in user_input.lower():
            return True
        elif "no" in user_input.lower():
            return False
        else:
            print("Bot: Sorry, I didn't get that. Please type 'Yes' or 'No'.")
            user_input = input("You: ")

def get_name():
    global name
    while True:
        if name == "":
            print("Bot: May I ask what your name is?")
            user_input = input("You: ")
            for word in user_input.split():
                if not is_english_word(word):
                    name = word.capitalize()
                    print(f"Bot: I understand your name is {name}. Is that correct?")
                    user_input = input("You: ")
                    if handle_confirmation(user_input):
                        return f"Bot: Nice to meet you, {name}! How are you feeling today?"
                    else:
                        print("Bot: No problem, let's try again.")
                        name = ""

# --- Intent Matching ---
def mood_matching(user_input):
    all_phrases = [phrase for phrases in processed_moods.values() for phrase in phrases]

    count_vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    count_matrix = count_vectorizer.fit_transform(all_phrases)
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

    input_count_vector = count_vectorizer.transform([preprocess(user_input)])
    input_tfidf_vector = tfidf_transformer.transform(input_count_vector)

    input_vector = input_tfidf_vector.toarray()[0]
    tfidf_vectors = tfidf_matrix.toarray()

    similarities = [1 - cosine(input_vector, phrase_vector) for phrase_vector in tfidf_vectors]

    if np.isnan(max(similarities)) or max(similarities) < 0.4:
        return None

    best_match_index = np.argmax(similarities)
    intent_lengths = [len(phrases) for phrases in processed_moods.values()]
    cumulative_lengths = np.cumsum(intent_lengths)

    for i, length in enumerate(cumulative_lengths):
        if best_match_index < length:
            return list(processed_moods.keys())[i]

# --- Conversation Flow ---
def emotion_q():
    global matched_mood
    user_input = input("You: ")
    matched_mood = mood_matching(user_input)
    if matched_mood is None:
        print("Bot: I'm sorry, I didn't quite understand that. Try saying something like 'I'm feeling happy'.")
    elif matched_mood == "happy":
        print(f"Bot: I'm so glad to hear that you're happy, {name}!")
    elif matched_mood == "sad":
        print(f"Bot: I'm sorry you're feeling sad, {name}. I'm here for you.")
    elif matched_mood == "angry":
        print(f"Bot: It's okay to feel angry, {name}. Let's talk through it together.")

def activity_q():
    global matched_mood
    if matched_mood == "happy":
        print("Bot: Would you like to share what's made your day so good?")
    elif matched_mood == "sad":
        print("Bot: Would you like to talk about what's been bothering you?")
    elif matched_mood == "angry":
        print("Bot: Want to talk about what made you angry today?")
    input("You: ")  # Just consumes the response

def suggestion_q():
    if matched_mood == "happy":
        print("Bot: That's amazing!")
    elif matched_mood == "sad":
        print("Bot: That sounds tough. Thanks for opening up.")
    elif matched_mood == "angry":
        print("Bot: That must be really frustrating. I appreciate you sharing that.")

# --- Main Loop ---
def main():
    print("------------------------------------------------------------CHATBOT STARTED------------------------------------------------------------")
    print(get_name())
    emotion_q()
    activity_q()
    suggestion_q()
    print("------------------------------------------------------------CHATBOT ENDED--------------------------------------------------------------")

if __name__ == "__main__":
    main()
