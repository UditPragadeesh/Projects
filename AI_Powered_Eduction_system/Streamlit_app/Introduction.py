import streamlit as st

st.set_page_config(
    page_title='Introduction'
)
st.title('AI Powered Education system')

st.write('''**This app features 7 models** \n
**1.Pass/Fail Predictor:** Predicts student performance in a test.\n
**2.Score Predictor:** Predicts future scores. \n
**3.Learning style Predictor :** Groups studens based on learning styles. \n
**4.Dropout Risk analysing:** Analyses risk of a student dropout.\n
**5.Topic Detector:** Detects the topic of given essay.\n
**6.Digit Recognition:** Detects digit from provided image.\n
**7.Topic summarizer:** Gives a summary for essays.''')


