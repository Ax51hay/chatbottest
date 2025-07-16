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
    "happy": [...],
    "sad": [...],
    "angry": [...]
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
