# app.py
import streamlit as st
import numpy as np
import pickle
import base64
import pandas as pd
import re
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# optional audio
try:
    import speech_recognition as sr
except Exception:
    sr = None

# helpers (utils.py preferred)
try:
    from utils import clean_text, get_highlighted_html
except Exception:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
    def clean_text(text):
        text = str(text)
        text = re.sub(r'<.*?>',' ', text)
        text = re.sub(r'[^a-zA-Z]',' ', text)
        text = text.lower().strip()
        return " ".join([w for w in text.split() if w not in STOP_WORDS and len(w)>1])
    def get_highlighted_html(text):
        pos = {"good","great","excellent","love","best","nice","amazing","awesome"}
        neg = {"bad","worst","poor","hate","terrible","awful","disappointing"}
        parts=[]
        for w in text.split():
            wc = re.sub(r'[^a-zA-Z]','',w).lower()
            if wc in pos: parts.append(f"<span style='color:green;font-weight:600'>{w}</span>")
            elif wc in neg: parts.append(f"<span style='color:red;font-weight:600'>{w}</span>")
            else: parts.append(w)
        return " ".join(parts)

st.set_page_config(page_title="Sentiment+Emotion Analyzer", page_icon="💬", layout="wide")
st.title("💬 Customer Review Sentiment & Emotion Analyzer")

MODEL_PATH = "model_multi.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200

try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH,'rb') as f:
        tokenizer = pickle.load(f)
except Exception:
    st.error("Model/tokenizer not found. Run train_model_multi.py to create 'model_multi.h5' and 'tokenizer.pkl'.")
    st.stop()

sentiment_map = {0:"Negative",1:"Neutral",2:"Positive"}
emoji_map = {0:"☹️ Negative",1:"😐 Neutral",2:"😊 Positive"}

def predict_probs(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    probs = model.predict(padded)[0]
    cls = int(np.argmax(probs))
    conf = float(np.max(probs))
    return cls, conf, probs

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Single Text","Bulk Upload","Audio Input","Admin Dashboard","About"])

if page == "Single Text":
    st.header("🧪 Single Review Analysis")
    with st.form("text_form"):
        input_text = st.text_area("Enter review text here:", height=160)
        spell = st.checkbox("Auto spell-correction (SpellChecker)")
        show_pre = st.checkbox("Show preprocessing steps")
        submit = st.form_submit_button("Analyze")
    if submit:
        if not input_text or not input_text.strip():
            st.warning("Please enter text.")
        else:
            if spell:
                try:
                    from spellchecker import SpellChecker
                    sp = SpellChecker()
                    corrected = [sp.correction(w) or w for w in input_text.split()]
                    input_used = " ".join(corrected)
                    st.info("Corrected Text:")
                    st.write(input_used)
                except Exception:
                    input_used = input_text
            else:
                input_used = input_text
            if show_pre:
                st.write("**Preprocessing:**")
                st.write("- Original:", input_text)
                st.write("- Cleaned:", clean_text(input_text))
            cls, conf, probs = predict_probs(input_used)
            st.markdown(f"### Predicted: **{sentiment_map[cls]}** ({emoji_map[cls]})")
            st.markdown(f"**Confidence:** {conf*100:.1f}%")
            prob_df = pd.DataFrame({"Sentiment":["Negative","Neutral","Positive"], "Probability":probs})
            st.bar_chart(prob_df.set_index("Sentiment"))
            st.markdown("**Keyword Highlighting:**")
            st.markdown(get_highlighted_html(input_text), unsafe_allow_html=True)
            try:
                import text2emotion as te
                st.write("**Emotions:**", te.get_emotion(input_text))
            except Exception:
                st.info("Install text2emotion for emotions.")
            wc = WordCloud(width=800, height=300, background_color="white").generate(clean_text(input_used))
            st.image(wc.to_array(), use_column_width=True)

elif page == "Bulk Upload":
    st.header("📁 Bulk Sentiment Analysis (CSV)")
    st.write("Upload CSV with 'review' or 'reviewText' column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "reviewText" in df.columns and "review" not in df.columns:
            df = df.rename(columns={"reviewText":"review"})
        if "review" not in df.columns:
            st.error("CSV must have 'review' or 'reviewText' column.")
            st.stop()
        st.dataframe(df.head())
        if st.button("Run Bulk Analysis"):
            texts = df['review'].astype(str).tolist()
            results=[]
            for t in texts:
                cls, conf, probs = predict_probs(t)
                results.append({"predicted_class":sentiment_map[cls],"confidence":conf,
                                "p_negative":float(probs[0]),"p_neutral":float(probs[1]),"p_positive":float(probs[2])})
            out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            st.success("Done")
            st.dataframe(out.head())
            st.subheader("Class distribution")
            st.bar_chart(out['predicted_class'].value_counts())
            csv = out.to_csv(index=False).encode('utf-8')
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_bulk_results.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

elif page == "Audio Input":
    st.header("🎙️ Audio → Text → Sentiment")
    st.write("Upload WAV/MP3; transcription uses Google API via SpeechRecognition (online).")
    if sr is None:
        st.info("Install speechrecognition & pydub to enable audio.")
    else:
        audio = st.file_uploader("Upload audio", type=["wav","mp3","m4a"])
        if audio:
            temp = "temp_audio.wav"
            with open(temp,"wb") as f: f.write(audio.getbuffer())
            r = sr.Recognizer()
            try:
                with sr.AudioFile(temp) as source:
                    aud = r.record(source)
                text = r.recognize_google(aud)
                st.write("Transcribed:", text)
                cls, conf, probs = predict_probs(text)
                st.markdown(f"### Predicted: **{sentiment_map[cls]}** ({conf*100:.1f}% confidence)")
            except Exception as e:
                st.error("Transcription failed. Check dependencies or internet.")
                st.write(e)

elif page == "Admin Dashboard":
    st.header("👑 Admin Dashboard")
    u = st.text_input("Admin Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u=="admin" and p=="1234":
            st.success("Welcome Admin")
            st.metric("Model","Bidirectional LSTM (3-class)")
            for fname,cap in [("wordcloud_positive.png","Positive"),("wordcloud_neutral.png","Neutral"),("wordcloud_negative.png","Negative")]:
                try:
                    st.image(fname, caption=cap, use_column_width=True)
                except Exception:
                    st.info(f"{fname} not found.")
            st.write("Use Bulk Upload to analyze many reviews.")
        else:
            st.error("Invalid credentials.")

else:
    st.header("About")
    st.markdown("""
    - Model: Bidirectional LSTM (3-class)
    - Features: Single/Bulk/Audio inputs, preprocessing, wordclouds, admin dashboard
    - Author: Mangaladeepa Kannan
    """)
