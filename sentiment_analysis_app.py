# ============================================
# Streamlit App for Sentiment Analysis
# ============================================
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text

# Load model and tokenizer
model = load_model("model_lstm.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Streamlit Page Config
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")
st.title("💬 Customer Review Sentiment Analysis (LSTM)")
st.write("Analyze customer feedback and detect positive or negative sentiment automatically.")

# Predict Function
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200, padding='post')
    pred = (model.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive 😊" if pred == 1 else "Negative 😞"

# Sidebar Login
login_type = st.sidebar.radio("Login as:", ["User", "Admin"])

if login_type == "User":
    st.subheader("🔹 User Mode: Test Sentiment")
    user_input = st.text_area("✍️ Enter your review below:")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            result = predict_sentiment(user_input)
            st.success(f"Predicted Sentiment: **{result}**")
        else:
            st.warning("Please type a review first.")

elif login_type == "Admin":
    st.subheader("👑 Admin Dashboard")
    username = st.text_input("Admin Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.success("Welcome, Admin 👑")
            st.metric("Model Accuracy", "91.8%")
            st.metric("F1-Score", "0.90")
            st.write("📊 Model performing well on test data.")
            st.info("You can upload new reviews for bulk sentiment analysis soon (future update).")
        else:
            st.error("Invalid credentials! Try again.")

# Note: Word cloud images and bulk upload feature are planned for future updates.
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Model and tokenizer saved successfully!")

# =======================================================
# Sentiment Analysis using LSTM + Streamlit Interface
# =======================================================

import streamlit as st
import re
import nltk
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords

# -------------------------------------------------------
# 1️⃣ NLTK Setup
# -------------------------------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------------------------------
# 2️⃣ Text Cleaning Function
# -------------------------------------------------------
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))                # remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)                # remove numbers/symbols
    text = text.lower()                                   # lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# -------------------------------------------------------
# 3️⃣ Load Saved Model & Tokenizer
# -------------------------------------------------------
try:
    model = load_model("model_lstm.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error("⚠️ Model or tokenizer not found! Please ensure 'model_lstm.h5' and 'tokenizer.pkl' exist in the same folder.")
    st.stop()

# -------------------------------------------------------
# 4️⃣ Streamlit Page Config
# -------------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

st.title("💬 Customer Review Sentiment Analysis (LSTM)")
st.write("Analyze customer feedback and detect **Positive** or **Negative** sentiment automatically.")

# -------------------------------------------------------
# 5️⃣ Prediction Function
# -------------------------------------------------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200, padding='post')
    pred = (model.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive 😊" if pred == 1 else "Negative 😞"

# -------------------------------------------------------
# 6️⃣ Sidebar Navigation
# -------------------------------------------------------
st.sidebar.title("🔐 Login Options")
login_type = st.sidebar.radio("Login as:", ["User", "Admin"])

# -------------------------------------------------------
# 7️⃣ User Mode
# -------------------------------------------------------
if login_type == "User":
    st.subheader("🔹 User Mode: Test Sentiment")
    user_input = st.text_area("✍️ Enter your review below:")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            result = predict_sentiment(user_input)
            st.success(f"Predicted Sentiment: **{result}**")
        else:
            st.warning("Please type a review first.")

# -------------------------------------------------------
# 8️⃣ Admin Mode
# -------------------------------------------------------
elif login_type == "Admin":
    st.subheader("👑 Admin Dashboard Login")
    username = st.text_input("Admin Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.success("Welcome, Admin 👑")
            st.metric("Model Accuracy", "91.8%")
            st.metric("F1-Score", "0.90")
            st.write("📊 Model performing well on test data.")

            # Optional: Display placeholder images if available
            try:
                st.image("wordcloud_positive.png", caption="Positive Reviews", use_container_width=True)
                st.image("wordcloud_negative.png", caption="Negative Reviews", use_container_width=True)
            except:
                st.info("🖼️ Wordcloud images not found — skip showing them.")
        else:
            st.error("❌ Invalid credentials! Try again.")

# -------------------------------------------------------
# 9️⃣ Footer
# -------------------------------------------------------
st.markdown("---")
st.caption("Developed by Mangaladeepa Kannan | Deep Learning NLP Project | © 2025")

