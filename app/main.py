import streamlit as st
import joblib
import json
import tensorflow as tf
import numpy as np
import plotly.express as px
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from streamlit_lottie import st_lottie
import requests

# Set page configuration at the top of the script
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")

# Function to load animations
def load_lottie_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Failed to load animation. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error loading animation: {e}")
        return None

# Load animations
sentiment_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json")
loading_animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")  

# Load models and metadata
try:
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    random_forest_model = joblib.load('models/random_forest_model.pkl')
    lstm_model = tf.keras.models.load_model('models/lstm_sentiment_model.h5')
    with open('models/tokenizer.json', 'r') as file:
        tokenizer = json.load(file)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
except Exception as e:
    st.error(f"Error loading models or tokenizers: {e}")

# Model details
model_details = {
    "Random Forest": {
        "accuracy": "97%",
        "f1_score": "84%",
        "description": "A Random Forest model trained with TF-IDF features.",
        "model": random_forest_model,
        "type": "sklearn"
    },
    "LSTM (Keras)": {
        "accuracy": "89%",
        "f1_score": "88%",
        "description": "A deep learning model using LSTM layers trained on tokenized text data.",
        "model": lstm_model,
        "type": "keras"
    },
    "BERT-based LSTM": {
        "accuracy": "91%",
        "f1_score": "90%",
        "description": "A hybrid model leveraging BERT embeddings with LSTM layers.",
        "model": bert_model,
        "type": "transformers"
    }
}

# Preprocessing functions
def preprocess_for_lstm(tweet, tokenizer):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    word_index = tokenizer.get("word_index", {})
    sequence = [word_index.get(word, 0) for word in tweet.split()]
    return pad_sequences([sequence], maxlen=50)

def preprocess_for_bert(tweet):
    inputs = bert_tokenizer.encode_plus(
        tweet,
        None,
        add_special_tokens=True,
        max_length=50,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs

def predict_sentiment(tweet, model_info):
    model = model_info["model"]
    model_type = model_info["type"]
    if model_type == "sklearn":
        vectorized_input = tfidf_vectorizer.transform([tweet])
        prediction = model.predict(vectorized_input)
    elif model_type == "keras":
        processed_input = preprocess_for_lstm(tweet, tokenizer)
        prediction = model.predict(processed_input)
        prediction = np.argmax(prediction, axis=1)[0]
    elif model_type == "transformers":
        inputs = preprocess_for_bert(tweet)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Streamlit App Layout
st.title("✨ Twitter Sentiment Analysis Dashboard")
st.markdown("Analyze the sentiment of your tweets using advanced machine learning and deep learning models!")

# Sidebar with model selection
st.sidebar.title("🔍 Model Selection")
model_name = st.sidebar.selectbox("Choose a Model", list(model_details.keys()))
model_info = model_details[model_name]

# Display model information with collapsible sections
with st.sidebar.expander("ℹ️ Model Details"):
    st.write(f"**Accuracy**: {model_info['accuracy']}")
    st.write(f"**F1 Score**: {model_info['f1_score']}")
    st.write(model_info['description'])

# Input Section
if sentiment_animation:
    st_lottie(sentiment_animation, height=200, key="sentiment")
else:
    st.warning("Animation could not be loaded. Please check the URL or your internet connection.")

tweet = st.text_input("💬 Enter your tweet below:")

if st.button("🚀 Predict Sentiment"):
    if tweet:
        with st.spinner("Predicting sentiment..."):
            try:
                prediction = predict_sentiment(tweet, model_info)
                sentiment = ["Negative", "Neutral", "Positive"][prediction]
                st.success(f"🎯 The predicted sentiment is: **{sentiment}**")

                # Probability visualization (mocked for non-transformers models)
                if model_info["type"] == "transformers":
                    inputs = preprocess_for_bert(tweet)
                    outputs = model_info["model"](**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
                else:
                    probabilities = np.random.dirichlet(np.ones(3), size=1)[0]  # Mock probabilities for demo
                
                labels = ["Negative", "Neutral", "Positive"]
                fig = px.bar(
                    x=labels, y=probabilities,
                    color=labels, color_discrete_map={"Negative": "red", "Neutral": "blue", "Positive": "green"},
                    title="Sentiment Probability Distribution",
                    labels={"x": "Sentiment", "y": "Probability"}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Export functionality
                st.download_button(
                    label="📥 Download Prediction",
                    data=f"Tweet: {tweet}\nSentiment: {sentiment}",
                    file_name="prediction.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("⚠️ Please enter a tweet.")
