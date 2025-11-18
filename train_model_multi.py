
# train_model_multi.py
# Train a 3-class sentiment LSTM model and save tokenizer/model/wordclouds

import pandas as pd
import numpy as np
import re
import nltk
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# -------------
# NLTK downloads
# -------------
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# -------------
# Load & prepare dataset
# -------------
# Update the path to your CSV (amazon_reviews.csv)
df = pd.read_csv(r"C:\Users\Admin\Downloads\amazon_reviews.csv", encoding='latin-1')

# Columns used: 'reviewText' or 'review', 'overall' (rating)
if 'reviewText' in df.columns:
    df = df.rename(columns={'reviewText': 'review'})
if 'review' not in df.columns:
    raise ValueError("Dataset must contain 'review' or 'reviewText' column.")

# Map ratings -> sentiment classes (3-class)
# Negative: overall <= 2, Neutral: overall == 3, Positive: overall >= 4
df = df.dropna(subset=['review', 'overall'])
df['overall'] = df['overall'].astype(int)
def map_label(x):
    if x <= 2:
        return 0  # Negative
    elif x == 3:
        return 1  # Neutral
    else:
        return 2  # Positive
df['sentiment'] = df['overall'].apply(map_label)

# -------------
# Cleaning
# -------------
import string
def clean_text(s):
    s = str(s)
    s = re.sub(r'<.*?>', ' ', s)
    s = re.sub(r'[^a-zA-Z]', ' ', s)
    s = s.lower().strip()
    tokens = [w for w in s.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

df['clean_review'] = df['review'].apply(clean_text)

# Filter very short reviews (optional)
df = df[df['clean_review'].str.split().str.len() >= 2]

# -------------
# Tokenize & pad
# -------------
MAX_WORDS = 15000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_review'])
sequences = tokenizer.texts_to_sequences(df['clean_review'])
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

# -------------
# Prepare labels
# -------------
y = to_categorical(df['sentiment'], num_classes=3)

# -------------
# Train/test split
# -------------
X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42, stratify=df['sentiment'])

# -------------
# Build model (Bidirectional LSTM)
# -------------
EMBED_DIM = 128

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])
model.summary()

# -------------
# Train (you can increase epochs if you have time)
# -------------
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# -------------
# Save model & tokenizer
# -------------
model.save("model_multi.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Saved model_multi.h5 and tokenizer.pkl")
# -------------
# Save model with gzip compression
# -------------
import h5py
from tensorflow.keras.models import load_model

model = load_model("model_multi.h5")
with h5py.File("model_multi_gzip.h5", "w") as f:
    model.save(f, compression="gzip")

print("Model saved with gzip compression: model_multi_gzip.h5")
# -------------
# Generate WordClouds (positive/neutral/negative)
# -------------
pos_text = " ".join(df[df['sentiment']==2]['clean_review'])
neu_text = " ".join(df[df['sentiment']==1]['clean_review'])
neg_text = " ".join(df[df['sentiment']==0]['clean_review'])

def save_wc(text, fname, bg='white'):
    if not text.strip():
        return
    wc = WordCloud(width=1200, height=600, background_color=bg).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

save_wc(pos_text, "wordcloud_positive.png", bg='white')
save_wc(neu_text, "wordcloud_neutral.png", bg='white')
save_wc(neg_text, "wordcloud_negative.png", bg='black')

print("Saved wordcloud images.")
