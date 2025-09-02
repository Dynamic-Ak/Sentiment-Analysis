import os
import re
import nltk
import string
import pickle
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Download NLTK data if needed
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
 

stopwords_list = set(stopwords.words('english'))
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    """
    Uses regex to find anything inside <...> and remove it
    """
    return TAG_RE.sub('', text)

# PreprocessData class
class PreprocessData:
    def __init__(self):
        pass
    
    def preprocess_text(self, sentence):
        sentence = sentence.lower()

        # Remove html tags
        sentence = remove_tags(sentence)

        # Remove Punctuations & Numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Remove Single Characters
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Remove Multiple Spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        # Remove Stopwords
        pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
        sentence = pattern.sub('', sentence)

        return sentence

# Load trained LSTM model and tokenizer if present; otherwise use VADER fallback
MODEL_DIR = os.path.join(BASE_DIR, "models")
POSSIBLE_MODEL_FILES = [
    os.path.join(MODEL_DIR, "lstm_sentiment_classifier.h5"),
    os.path.join(MODEL_DIR, "lstm_model.h5"),
    os.path.join(BASE_DIR, "lstm_model.h5"),
]
POSSIBLE_TOKENIZER_FILES = [
    os.path.join(MODEL_DIR, "tokenizer.pkl"),
    os.path.join(MODEL_DIR, "tokenizer.pickle"),
    os.path.join(BASE_DIR, "tokenizer.pickle"),
    os.path.join(BASE_DIR, "tokenizer.pkl"),
]

MODEL_PATH = next((p for p in POSSIBLE_MODEL_FILES if os.path.exists(p)), None)
TOKENIZER_PATH = next((p for p in POSSIBLE_TOKENIZER_FILES if os.path.exists(p)), None)

USE_DL = False
preprocess = PreprocessData()
MAXLEN = 250
analyzer = None

if MODEL_PATH and TOKENIZER_PATH:
    try:
        lstm_model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        USE_DL = True
        print(f"Loaded DL model from: {MODEL_PATH}\nLoaded tokenizer from: {TOKENIZER_PATH}")
    except Exception as e:
        print(f"Failed to load DL model/tokenizer, falling back to VADER. Error: {e}")
        analyzer = SentimentIntensityAnalyzer()
else:
    print("Model/tokenizer files not found. Using NLTK VADER as a fallback.")
    analyzer = SentimentIntensityAnalyzer()

# Prediction function
def predict_sentiment(review_text):
    if not review_text.strip():
        return None, None

    if USE_DL:
        # DL pipeline
        review = preprocess.preprocess_text(review_text)
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAXLEN)
        pred_prob = float(lstm_model.predict(padded, verbose=0)[0][0])
        sentiment = 'Positive' if pred_prob >= 0.5 else 'Negative'
        return sentiment, pred_prob
    else:
        # VADER fallback uses raw text for better handling of negations/emphasis
        scores = analyzer.polarity_scores(review_text)
        # Map compound [-1,1] to [0,1]
        pred_prob = (scores['compound'] + 1.0) / 2.0
        sentiment = 'Positive' if pred_prob >= 0.5 else 'Negative'
        return sentiment, pred_prob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form.get('review', '')
    reviews = [r.strip() for r in review_text.split('\n') if r.strip()]
    
    if not reviews:
        return jsonify({'error': 'No reviews provided'}), 400
    
    results = []
    positive_count = 0
    negative_count = 0
    
    for rev in reviews:
        sentiment, prob = predict_sentiment(rev)
        if sentiment == 'Positive':
            positive_count += 1
        elif sentiment == 'Negative':
            negative_count += 1
        results.append({
            'review': rev,
            'sentiment': sentiment,
            'probability': f"{prob:.2f}"
        })
    
    is_batch = len(reviews) > 1
    distribution = {
        'positive': positive_count,
        'negative': negative_count
    } if is_batch else None
    
    return jsonify({
        'results': results,
        'distribution': distribution
    })

if __name__ == '__main__':
    app.run(debug = True)