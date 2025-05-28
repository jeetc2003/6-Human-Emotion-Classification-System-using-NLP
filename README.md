# 6-Human-Emotion-Classification-System-using-NLP

Welcome to the Emotion Detector web app! This project predicts six basic human emotions from text using both Machine Learning and Deep Learning techniques.

Built with ❤️ by Jeet Choudhury using Streamlit, TensorFlow, Scikit-learn, and Natural Language Processing (NLP).


##🔍 Features
- Predicts 6 emotions from any English sentence.

- Dual prediction using:

- ✅ Machine Learning (Logistic Regression + TF-IDF)

- ✅ Deep Learning (Keras Sequential Model)

- NLP text cleaning using NLTK (stopwords, stemming).

- Clean and minimal Streamlit UI.

- Shows emotion confidence percentage from DL model.

- Made for easy web deployment on Streamlit Cloud.

##🚀 Live Demo
👉 https://human-emotion-nlp-4jeet.streamlit.app/

##🧠 Emotions Predicted
😠 Anger

😨 Fear

😄 Joy

❤️ Love

😢 Sadness

😲 Surprise

##⚙️ How It Works
🔤 Text Preprocessing
Remove special characters and lowercase

Remove stopwords

Stemming using PorterStemmer

##🔍 ML Prediction
Text is vectorized using TF-IDF

Passed to a Logistic Regression model

Output label is decoded via LabelEncoder

##🤖 DL Prediction
Text is tokenized using one-hot encoding

Sequence is padded

Passed to a Keras deep learning model

Outputs emotion class with confidence score
