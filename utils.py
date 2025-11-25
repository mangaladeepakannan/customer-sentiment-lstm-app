# utils.py
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOP = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>',' ', text)
    text = re.sub(r'[^a-zA-Z]',' ', text)
    text = text.lower().strip()
    return " ".join([w for w in text.split() if w not in STOP and len(w)>1])

def get_highlighted_html(text):
    pos = {"good","great","excellent","love","best","nice","amazing","awesome","satisfied"}
    neg = {"bad","worst","poor","hate","terrible","awful","disappointing","slow","late"}
    parts=[]
    for w in text.split():
        wc = re.sub(r'[^a-zA-Z]','', w).lower()
        if wc in pos:
            parts.append(f"<span style='color:green;font-weight:600'>{w}</span>")
        elif wc in neg:
            parts.append(f"<span style='color:red;font-weight:600'>{w}</span>")
        else:
            parts.append(w)
    return " ".join(parts)
