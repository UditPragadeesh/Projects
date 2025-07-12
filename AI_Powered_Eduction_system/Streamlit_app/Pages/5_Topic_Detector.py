import streamlit as st
import torch
from torch import nn
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import re
import json

st.title('Topic Detector')

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = stopwords.words('english')
punctuations = string.punctuation
lemmatizer = WordNetLemmatizer()

def preprocess(txt):
    txt=txt.lower()
    txt = re.sub(r'[^A-Za-z0-9\s]','',txt)
    clean_text = nltk.word_tokenize(txt)
    clean_text = [word for word in clean_text if word not in stop_words and word not in punctuations]
    clean_text = [lemmatizer.lemmatize(word) for word in clean_text]
    return ' '.join(clean_text)

class Net(nn.Module):
    def __init__(self,num_classes,input_size=1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10000,embedding_dim=50)
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=32,num_layers=2,batch_first=True)
        self.lin = nn.Linear(32,num_classes)
    def forward(self,x):
        x = self.embedding(x)
        h0 = torch.zeros(2,x.size(0),32)
        c0 = torch.zeros(2,x.size(0),32)
        out,_ = self.lstm(x,(h0,c0))
        out = self.lin(out[:,-1,:])
        return out

tokenizer = pickle.load(open(r'Models\Topic_Detector\tokenizer.sav','rb'))
encoder = pickle.load(open(r'Models\Topic_Detector\encoder.sav','rb'))
classifier = Net(num_classes=76,input_size=50)
classifier.load_state_dict(torch.load(r'Models\Topic_Detector\topic_model_weights.pth'))

input = st.text_area('Input text here:')
length = len(input.split())
if length<=50:
    color = 'green'
else:
    color = 'red'
st.markdown(f"<h3 style='color: {color}; font-size:16px;'>{length} words </h3>",unsafe_allow_html=True)

if 'find_bt' not in st.session_state:
    st.session_state.find_bt = False
if st.button('find'):
    st.session_state.find_bt = True

if st.session_state.find_bt:
    if len(input.split())<2:
        st.error('Input a complete sentence')
    elif len(input.split())>50:
        st.error('Input too long.(Input less than 50 words)')
    else:
        clean_text = preprocess(input)
        seq = tokenizer.texts_to_sequences([clean_text])
        seq = pad_sequences(seq,maxlen=50)
        seq = torch.tensor(seq)
        classifier.eval()
        op = classifier(seq)
        _,preds =torch.max(op,1)
        result = encoder.inverse_transform(preds)
        st.write('Topic is:')
        st.markdown(f"<h3 style='font-size:32px;'>{result[0]}</h3>",unsafe_allow_html=True)
        if st.button('OK'):
            st.session_state.find_bt=False


