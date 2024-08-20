from flask import Flask, request, render_template
import pickle
import numpy as np
from textstat.textstat import textstatistics
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, KeyedVectors
import nltk
#   nltk.download('punkt')
from textstat.textstat import textstatistics
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
# Import or define your text processing functions here

app = Flask(__name__)

# Load your trained models
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
pca_model = pickle.load(open('mlp_pca.pkl', 'rb'))
mlp = pickle.load(open('mlp 2.pkl', 'rb'))



def lexical_diversity(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0
    else:
        return len(set(tokens)) / len(tokens)
    
def readability_score(text):
    return textstatistics().flesch_reading_ease(text)

def generate_word2vec_features(text, word2vec_model):
    tokens = text.split()
    embeddings = np.zeros((300,))
    valid_tokens = 0
    for token in tokens:
        if token in word2vec_model:
            embeddings += word2vec_model[token]
            valid_tokens += 1
    if valid_tokens > 0:
        embeddings /= valid_tokens
    return embeddings

def text_to_features(text, word2vec_model, pca_model):
    lex_div = lexical_diversity(text)
    read_score = readability_score(text)
    word2vec_features = generate_word2vec_features(text, word2vec_model)
    word2vec_features_reshaped = word2vec_features.reshape(1, -1)
    word2vec_features_reduced = pca_model.transform(word2vec_features_reshaped)
    features = np.concatenate(([lex_div, read_score], word2vec_features_reduced.flatten()))
    return features

def predict_text_classification(text, word2vec_model, pca_model, mlp):
    features = text_to_features(text, word2vec_model, pca_model).reshape(1, -1)
    prediction = mlp.predict(features)
    return "AI-generated" if prediction == 1 else "Human-written"



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')  # Safely get text, defaulting to empty string if not found
    
    if text:  # Only predict if text is not empty
        prediction = predict_text_classification(text,word2vec_model, pca_model, mlp)
        prediction_text = f'Prediction: {prediction}'
    else:
        prediction_text = 'No text provided.'
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
