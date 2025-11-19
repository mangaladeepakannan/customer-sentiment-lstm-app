q='temp_audio.wav'
p='predicted_class'
o='Sentiment'
n='Admin Dashboard'
m='Audio Input'
l='Bulk Upload'
k='Single Text'
j='Positive'
i='Neutral'
h='Negative'
T='reviewText'
L=float
K=Exception
G='review'
D=True
import streamlit as A,numpy as U,pickle as r,io,base64 as s,pandas as I,matplotlib.pyplot as AN
from wordcloud import WordCloud as t
from tensorflow.keras.preprocessing.sequence import pad_sequences as u
from tensorflow.keras.models import load_model as v
import speech_recognition as V
from utils import clean_text as M,get_highlighted_html as w
A.set_page_config(page_title='Sentiment+Emotion Analyzer',page_icon='💬',layout='wide')
A.title('💬 Customer Review Sentiment & Emotion Analyzer')
A.markdown('**LSTM multi-class sentiment (Negative / Neutral / Positive)** + emotion detection + many features.')
try:
	x=v('model_multi.h5')
	with open('tokenizer.pkl','rb')as N:y=r.load(N)
except K as W:A.error('Model/tokenizer not found. Run `train_model_multi.py` first to create `model_multi.h5` and `tokenizer.pkl`.');A.stop()
z=200
O={0:h,1:i,2:j}
A0={0:'☹️ Negative',1:'😐 Neutral',2:'😊 Positive'}
def P(text):B=M(text);C=y.texts_to_sequences([B]);D=u(C,maxlen=z,padding='post');A=x.predict(D)[0];E=int(U.argmax(A));F=L(U.max(A));return E,F,A
A.sidebar.header('Navigation')
J=A.sidebar.radio('Go to',[k,l,m,n,'About'])
if J==k:
	A.header('🧪 Single Review Analysis')
	with A.form('text_form'):B=A.text_area('Enter review text here:',height=160);A1=A.checkbox('Auto spell-correction (SpellChecker)');A2=A.checkbox('Show preprocessing steps');A3=A.form_submit_button('Analyze')
	if A3:
		if not B or not B.strip():A.warning('Please enter text to analyze.')
		else:
			if A1:
				try:
					from spellchecker import SpellChecker as A4;A5=A4();X=[]
					for Y in B.split():Z=A5.correction(Y);X.append(Z if Z else Y)
					Q=' '.join(X)
				except K:Q=B
				A.info('Corrected Text:');A.write(Q);R=Q
			else:R=B
			if A2:A.write('**Preprocessing steps:**');A.write('- Original:',B);A6=B.lower();A.write('- Lowercased:',A6);import re;A7=re.sub('<.*?>','',B);A.write('- Remove HTML:',A7);A8=re.sub('[^a-zA-Z]',' ',B);A.write('- Letters only:',A8);A9=M(B);A.write('- Remove stopwords & cleaned:',A9)
			E,H,F=P(R);A.markdown(f"### Predicted sentiment: **{O[E]}** ({A0[E]})");A.markdown(f"**Confidence:** {H*100:.1f}%");AA=I.DataFrame({o:[h,i,j],'Probability':F});A.bar_chart(AA.set_index(o));A.markdown('**Keyword Highlighting:**');A.markdown(w(B),unsafe_allow_html=D)
			try:import text2emotion as AB;AC=AB.get_emotion(B);A.write('**Emotions detected:**',AC)
			except K:A.info('Install `text2emotion` for emotion detection (optional).')
			AD=t(width=800,height=300,background_color='white').generate(M(R));A.image(AD.to_array(),use_column_width=D)
elif J==l:
	A.header('📁 Bulk Sentiment Analysis (CSV)');A.write("Upload a CSV with a column named `review` (or `reviewText`). We'll analyze each row and return a CSV with results.");a=A.file_uploader('Upload CSV file',type=['csv'])
	if a:
		C=I.read_csv(a);b=None
		if G in C.columns:b=G
		elif T in C.columns:b=T;C=C.rename(columns={T:G})
		else:A.error("CSV must contain a 'review' or 'reviewText' column.");A.stop()
		A.write('Sample rows:');A.dataframe(C.head())
		if A.button('Run Bulk Analysis'):
			AE=C[G].astype(str).tolist();c=[]
			for d in AE:E,H,F=P(d);c.append({G:d,p:O[E],'confidence':H,'p_negative':L(F[0]),'p_neutral':L(F[1]),'p_positive':L(F[2])})
			AF=I.DataFrame(c);S=I.concat([C.reset_index(drop=D),AF.drop(columns=[G])],axis=1);A.success('Bulk analysis complete. Preview:');A.dataframe(S.head());A.subheader('Class distribution');A.bar_chart(S[p].value_counts());AG=S.to_csv(index=False).encode('utf-8');AH=s.b64encode(AG).decode();AI=f'<a href="data:file/csv;base64,{AH}" download="sentiment_bulk_results.csv">Download results as CSV</a>';A.markdown(AI,unsafe_allow_html=D)
elif J==m:
	A.header('🎙️ Audio → Text → Sentiment');A.write('Upload a WAV/MP3 audio file (spoken review). The app will transcribe and analyze sentiment.');e=A.file_uploader('Upload audio (wav/mp3)',type=['wav','mp3','m4a'])
	if e:
		with open(q,'wb')as N:N.write(e.getbuffer())
		f=V.Recognizer()
		try:
			with V.AudioFile(q)as AJ:AK=f.record(AJ)
			g=f.recognize_google(AK);A.write('Transcribed text:',g);E,H,F=P(g);A.markdown(f"### Predicted sentiment: **{O[E]}** ({H*100:.1f}% confidence)")
		except K as W:A.error('Audio transcription failed. Make sure the file is clear and system supports SpeechRecognition dependencies.');A.write(W)
elif J==n:
	A.header('👑 Admin Dashboard');AL=A.text_input('Admin Username');AM=A.text_input('Password',type='password')
	if A.button('Login'):
		if AL=='admin'and AM=='1234':
			A.success('Welcome Admin');A.metric('Model Type','Bidirectional LSTM (3-class)');A.metric('Classes','Negative / Neutral / Positive')
			try:A.image('wordcloud_positive.png',caption='Positive Wordcloud',use_column_width=D);A.image('wordcloud_neutral.png',caption='Neutral Wordcloud',use_column_width=D);A.image('wordcloud_negative.png',caption='Negative Wordcloud',use_column_width=D)
			except:A.info('Wordcloud images not found — run training script or upload images.')
			A.write('Tip: Use **Bulk Upload** to analyze many reviews and download results.')
		else:A.error('Invalid admin credentials.')
else:A.header('About this project');A.markdown('\n    - **Model:** Bidirectional LSTM (3-class sentiment)\n    - **Features:** Single / Bulk / Audio inputs, spell correction, preprocessing steps, emotion detection, wordclouds, admin dashboard.\n    - **Author:** Mangaladeepa Kannan\n    ')
