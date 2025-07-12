import streamlit as st
import pickle

st.title(r'AI-Based Topic Summarizer')

summarizer = pickle.load(open(r'Models\Topic_summarizer.sav','rb'))
input = st.text_area('Input text here')
length = len(input.split())
st.markdown(f"<h3 style='color: 'grey'; font-size:1px;'>{length} words</h3>",unsafe_allow_html=True)
col1,col2,col3 = st.columns(3)
max_length = col2.number_input('Input maximum length of summary:',min_value=50,max_value=1000,value=100)
min_length = col1.number_input('Input minimum length of summary:',min_value=15,max_value=100,value=30)

if 'sum_bt' not in st.session_state:
    st.session_state.sum_bt = False
if st.button('Summarize'):
    st.session_state.sum_bt = True

if st.session_state.sum_bt:
    summary = summarizer(input,max_length=max_length,min_length=min_length,do_sample=False)
    st.write('Summary:')
    st.markdown(f"<h3 style='font-size:16px;'>{summary[0]['summary_text']}</h3>",unsafe_allow_html=True)
    if st.button('OK'):
        st.session_state.sum_bt = False
