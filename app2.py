import streamlit as st
import numpy as np
import pickle
import nltk
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Base path from the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative paths
lg_model_path = os.path.join(BASE_DIR, "lg.pkl")
label_encoder_path = os.path.join(BASE_DIR, "lb.pkl")
tfidf_path = os.path.join(BASE_DIR, "tfidf_vec.pkl")
dl_model_path = os.path.join(BASE_DIR, "model.keras")

# Load models
ml_model = pickle.load(open(lg_model_path, "rb"))
label_encoder = pickle.load(open(label_encoder_path, "rb"))
tfidf = pickle.load(open(tfidf_path, "rb"))
dl_model = load_model(dl_model_path)


# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# Emotion index mapping for DL
emotion_index_map = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise",
}


# ------------------------------------------
#  Text Cleaning Functions
def clean_text_ml(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)


def clean_text_dl(text, vocab_size=11000, max_len=300):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = " ".join(text)
    encoded = [one_hot(text, vocab_size)]
    return pad_sequences(encoded, maxlen=max_len, padding="pre")


# ------------------------------------------
#  Prediction Functions
def predict_emotion_ml(text):
    cleaned = clean_text_ml(text)
    vectorized = tfidf.transform([cleaned])
    prediction = ml_model.predict(vectorized)[0]
    return prediction


def predict_emotion_dl(text):
    cleaned_seq = clean_text_dl(text)
    pred_probs = dl_model.predict(cleaned_seq)[0]
    pred_idx = np.argmax(pred_probs, axis=-1)
    return emotion_index_map[pred_idx], float(np.max(pred_probs) * 100)


# ------------------------------------------
# 🎨 Streamlit UI
st.set_page_config(page_title="Emotion Detector By JEET", page_icon="🧠")
st.title("🧠 Emotion Detection App")
st.markdown("This app predicts **Six Human Emotions** using NLP : Made By JEET")
st.success("`Joy`, `Love`, `Fear`, `Anger`, `Surprise`, `Sadness`")

input_text = st.text_area("✍️ Enter your sentence below:")

if st.button("🔍 Predict with ML Model"):
     if input_text.strip() == "":
         st.warning("Please enter some text.")
     else:
         emotion = predict_emotion_ml(input_text)
         st.info(f"**Predicted Emotion :** {emotion.upper()}")
         emotion, confidence = predict_emotion_dl(input_text)
         st.caption(f"Confidence: {confidence:.2f}%")



st.markdown("---")
st.caption("Built with ❤️ using Streamlit, TensorFlow, and Scikit-learn")
