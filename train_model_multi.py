# train_model_multi.py
# Train a 3-class Bidirectional LSTM and save model/tokenizer + wordclouds

import os, re, pickle
import pandas as pd
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOP = set(stopwords.words('english'))

# CONFIG
DATA_PATH = os.path.join("data", "your_dataset.csv")
MODEL_OUT = "model_multi.h5"
TOKENIZER_OUT = "tokenizer.pkl"
MAX_WORDS = 15000
MAX_LEN = 200
EMBED_DIM = 128
EPOCHS = 5
BATCH = 128

def clean_text(s):
    s = str(s)
    s = re.sub(r'<.*?>', ' ', s)
    s = re.sub(r'[^a-zA-Z]', ' ', s)
    s = s.lower().strip()
    toks = [w for w in s.split() if w not in STOP and len(w) > 1]
    return " ".join(toks)

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH, encoding='utf-8', low_memory=False)

# detect text column
if 'review' in df.columns:
    text_col = 'review'
elif 'reviewText' in df.columns:
    text_col = 'reviewText'
    df = df.rename(columns={'reviewText':'review'})
else:
    raise KeyError("Dataset must contain 'review' or 'reviewText' column.")

# detect label column or map numeric rating
if 'sentiment' in df.columns:
    label_col = 'sentiment'
else:
    if 'overall' in df.columns:
        def map_rating(r):
            try:
                r = float(r)
                if r <= 2: return 'negative'
                if r == 3: return 'neutral'
                return 'positive'
            except:
                return 'neutral'
        df['sentiment'] = df['overall'].apply(map_rating)
        label_col = 'sentiment'
    else:
        raise KeyError("Dataset must contain 'sentiment' or 'overall' column.")

df = df[[text_col, label_col]].dropna()
df = df.rename(columns={text_col:'review', label_col:'sentiment'})
df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
df = df[df['sentiment'].isin(['negative','neutral','positive'])]

print("Cleaning text...")
df['clean_review'] = df['review'].apply(clean_text)
df = df[df['clean_review'].str.split().str.len() >= 2]

label_map = {'negative':0,'neutral':1,'positive':2}
df['label'] = df['sentiment'].map(label_map)

print("Tokenizing...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_review'])
sequences = tokenizer.texts_to_sequences(df['clean_review'])
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

X_train, X_test, y_train, y_test = train_test_split(padded, df['label'].values,
                                                    test_size=0.2, random_state=42, stratify=df['label'])

print("Building model...")
model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])
model.summary()

print("Training...")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1)

print("Saving model and tokenizer...")
model.save(MODEL_OUT, include_optimizer=False)
with open(TOKENIZER_OUT, 'wb') as f:
    pickle.dump(tokenizer, f)

print("Generating wordclouds...")
for label, name, bg in [(2,'positive','white'),(1,'neutral','white'),(0,'negative','black')]:
    text = " ".join(df[df['label']==label]['clean_review'])
    if not text.strip(): continue
    wc = WordCloud(width=1200, height=600, background_color=bg).generate(text)
    fname = f"wordcloud_{name}.png"
    wc.to_file(fname)
    print("Saved:", fname)

print("Done. Artifacts:", MODEL_OUT, TOKENIZER_OUT)
