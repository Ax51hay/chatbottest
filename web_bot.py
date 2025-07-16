import nltk
import re
import numpy as np
from nltk.stem.snowball import PorterStemmer
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.spatial.distance import cosine
import warnings

nltk.download('punkt')
nltk.download('words')
nltk.data.path.append('./nltk_data')  # if you bundle nltk_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

english_words = set(words.words())

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

def preprocess(user_input):
    p_stemmer = PorterStemmer()
    cleaned = re.sub(r"[’']", "", user_input.lower())
    tokens = nltk.word_tokenize(cleaned)
    stemmed_tokens = [p_stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

processed_moods = {intent: [preprocess(phrase) for phrase in phrases] for intent, phrases in moods.items()}

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
