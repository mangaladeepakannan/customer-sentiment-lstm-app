# app.py
# ============================================
# Sentiment & Emotion Analyzer (Streamlit)
# ============================================

import streamlit as st
import numpy as np
import pickle
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import re

from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# optional imports (wrapped)
try:
    import speech_recognition as sr
except Exception:
    sr = None

# utils.py must provide these functions
# from utils import clean_text, get_highlighted_html
# If you don't have utils.py, a minimal clean_text is provided below.
try:
    from utils import clean_text, get_highlighted_html
except Exception:
    # minimal fallback clean_text and highlighter
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))

    def clean_text(text: str) -> str:
        text = str(text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = text.lower().strip()
        tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
        return " ".join(tokens)

    def get_highlighted_html(text: str) -> str:
        # simple fallback highlighter
        pos = {"good","great","excellent","love","best","nice","amazing","awesome"}
        neg = {"bad","worst","poor","hate","terrible","awful","disappointing"}
        parts = []
        for w in text.split():
            wc = re.sub(r"[^a-zA-Z]", "", w).lower()
            if wc in pos:
                parts.append(f"<span style='color:green;font-weight:600'>{w}</span>")
            elif wc in neg:
                parts.append(f"<span style='color:red;font-weight:600'>{w}</span>")
            else:
                parts.append(w)
        return " ".join(parts)

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Sentiment+Emotion Analyzer", page_icon="💬", layout="wide")
st.title("💬 Customer Review Sentiment & Emotion Analyzer")
st.markdown("**Bidirectional LSTM (Negative / Neutral / Positive)** + emotion detection + many features.")

# -------------------------
# Load model & tokenizer
# -------------------------
MODEL_PATH = "model_multi.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200

try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
except Exception as err:
    st.error("Model/tokenizer not found. Run the training script first to create 'model_multi.h5' and 'tokenizer.pkl'.")
    st.stop()

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
emoji_map = {0: "☹️ Negative", 1: "😐 Neutral", 2: "😊 Positive"}

def predict_probs(text: str):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(padded)[0]
    cls = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return cls, confidence, probs

# -------------------------
# Sidebar - Navigation
# -------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Single Text", "Bulk Upload", "Audio Input", "Admin Dashboard", "About"])

# -------------------------
# Page: Single Text
# -------------------------
if page == "Single Text":
    st.header("🧪 Single Review Analysis")
    with st.form("text_form"):
        input_text = st.text_area("Enter review text here:", height=160)
        spell_correction = st.checkbox("Auto spell-correction (SpellChecker)")
        show_preprocessing = st.checkbox("Show preprocessing steps")
        run_btn = st.form_submit_button("Analyze")

    if run_btn:
        if not input_text or not input_text.strip():
            st.warning("Please enter text to analyze.")
        else:
            # optional spell correction
            if spell_correction:
                try:
                    from spellchecker import SpellChecker
                    spell = SpellChecker()
                    corrected = []
                    for w in input_text.split():
                        cw = spell.correction(w)
                        corrected.append(cw if cw else w)
                    corrected_text = " ".join(corrected)
                except Exception:
                    corrected_text = input_text
                st.info("Corrected Text:")
                st.write(corrected_text)
                text_to_use = corrected_text
            else:
                text_to_use = input_text

            # show preprocessing
            if show_preprocessing:
                st.write("**Preprocessing steps:**")
                st.write("- Original:", input_text)
                lower = input_text.lower()
                st.write("- Lowercased:", lower)
                no_html = re.sub(r"<.*?>", "", input_text)
                st.write("- Remove HTML:", no_html)
                letters_only = re.sub(r"[^a-zA-Z]", " ", input_text)
                st.write("- Letters only:", letters_only)
                cleaned = clean_text(input_text)
                st.write("- Remove stopwords & cleaned:", cleaned)

            # predict
            cls, conf, probs = predict_probs(text_to_use)
            st.markdown(f"### Predicted sentiment: **{sentiment_map[cls]}** ({emoji_map[cls]})")
            st.markdown(f"**Confidence:** {conf*100:.1f}%")

            # probability bar chart
            prob_df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Sentiment"))

            # show highlighted keywords
            st.markdown("**Keyword Highlighting:**")
            st.markdown(get_highlighted_html(input_text), unsafe_allow_html=True)

            # emotion detection (text2emotion)
            try:
                import text2emotion as te
                emotions = te.get_emotion(input_text)
                st.write("**Emotions detected:**", emotions)
            except Exception:
                st.info("Install `text2emotion` for emotion detection (optional).")

            # small wordcloud for this text
            wc = WordCloud(width=800, height=300, background_color="white").generate(clean_text(text_to_use))
            st.image(wc.to_array(), use_column_width=True)

# -------------------------
# Page: Bulk Upload
# -------------------------
elif page == "Bulk Upload":
    st.header("📁 Bulk Sentiment Analysis (CSV)")
    st.write("Upload a CSV with a column named `review` (or `reviewText`). We'll analyze each row and return a CSV with results.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "review" not in df.columns and "reviewText" in df.columns:
            df = df.rename(columns={"reviewText": "review"})
        if "review" not in df.columns:
            st.error("CSV must contain a 'review' or 'reviewText' column.")
            st.stop()

        st.write("Sample rows:")
        st.dataframe(df.head())

        if st.button("Run Bulk Analysis"):
            texts = df["review"].astype(str).tolist()
            results = []
            for t in texts:
                cls, conf, probs = predict_probs(t)
                results.append({
                    "review": t,
                    "predicted_class": sentiment_map[cls],
                    "confidence": conf,
                    "p_negative": float(probs[0]),
                    "p_neutral": float(probs[1]),
                    "p_positive": float(probs[2])
                })
            res_df = pd.DataFrame(results)
            out = pd.concat([df.reset_index(drop=True), res_df.drop(columns=["review"])], axis=1)
            st.success("Bulk analysis complete. Preview:")
            st.dataframe(out.head())

            # class distribution
            st.subheader("Class distribution")
            st.bar_chart(out["predicted_class"].value_counts())

            # download result CSV
            csv = out.to_csv(index=False).encode("utf-8")
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_bulk_results.csv">Download results as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

# -------------------------
# Page: Audio Input
# -------------------------
elif page == "Audio Input":
    st.header("🎙️ Audio → Text → Sentiment")
    st.write("Upload a WAV/MP3 audio file (spoken review). The app will transcribe and analyze sentiment.")
    if sr is None:
        st.info("SpeechRecognition not installed — audio transcription disabled. Install 'speechrecognition' and 'pydub' (with ffmpeg) to enable.")
    else:
        audio_file = st.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3", "m4a"])
        if audio_file:
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_file.getbuffer())
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(temp_path) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                st.write("Transcribed text:", text)
                cls, conf, probs = predict_probs(text)
                st.markdown(f"### Predicted sentiment: **{sentiment_map[cls]}** ({conf*100:.1f}% confidence)")
            except Exception as e:
                st.error("Audio transcription failed. Ensure file is clear and dependencies are installed.")
                st.write(e)

# -------------------------
# Page: Admin Dashboard
# -------------------------
elif page == "Admin Dashboard":
    st.header("👑 Admin Dashboard")
    username = st.text_input("Admin Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.success("Welcome Admin")
            st.metric("Model Type", "Bidirectional LSTM (3-class)")
            st.metric("Classes", "Negative / Neutral / Positive")

            # try to show wordclouds if available
            for fname, caption in [
                ("wordcloud_positive.png", "Positive Wordcloud"),
                ("wordcloud_neutral.png", "Neutral Wordcloud"),
                ("wordcloud_negative.png", "Negative Wordcloud"),
            ]:
                try:
                    st.image(fname, caption=caption, use_column_width=True)
                except Exception:
                    st.info(f"Wordcloud image '{fname}' not found — run training script or upload images.")
            st.write("Tip: Use **Bulk Upload** to analyze many reviews and download results.")
        else:
            st.error("Invalid admin credentials.")

# -------------------------
# Page: About
# -------------------------
else:
    st.header("About this project")
    st.markdown(
        """
        - **Model:** Bidirectional LSTM (3-class sentiment)
        - **Features:** Single / Bulk / Audio inputs, spell correction, preprocessing steps, emotion detection, wordclouds, admin dashboard.
        - **Author:** Mangaladeepa Kannan
        """
    )

