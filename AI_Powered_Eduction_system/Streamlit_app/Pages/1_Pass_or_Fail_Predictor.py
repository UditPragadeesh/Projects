import streamlit as st
import pickle
import sqlite3 as sql
import pandas as pd
import numpy as np

st.title('PASS OR FAIL PREDICTOR')

con=sql.connect('data\Train_datasets\Pass_Fail\AI_Edu_system.db')
cur=con.cursor() 
cur.execute('''SELECT student_ID FROM student_log''')

student_df=pd.DataFrame(cur.fetchall(),columns=['student_ID'])
student_df.drop_duplicates(inplace=True)
student_df.reset_index(inplace=True)
student_df = student_df.drop(columns='index',axis=1)
student_ids = list(student_df['student_ID'])

col1,col2 = st.columns(2)
selected_student = col1.selectbox('Select student:',options=student_ids)
pass_percent=col2.number_input('Minimum percentage to pass:',min_value=10,max_value=100,step=1,value=50)
file = st.file_uploader('Upload quiz data here(.csv):',type='.csv')
if st.button('Predict'):
     if file:
          quiz_data = pd.read_csv(file)
          skills = pd.read_sql('SELECT * FROM skills',con)
          problem_types = pd.read_sql('SELECT * FROM problemtype',con)
          quiz_data = quiz_data.merge(skills,on='skill',how='left')
          quiz_data = quiz_data.merge(problem_types,on='problemType',how='left')
          quiz_data = quiz_data.drop(columns=['skill','problemType'],axis=1)
          quiz_data['student_ID']=selected_student
          feature_cols = ['student_ID','timeTaken','hintCount','attemptCount','year1','year2','skill_ID','Type_ID']
          features=quiz_data[feature_cols]
          predictor = pickle.load(open(r'Models\correct_predictor.sav','rb'))
          correct=predictor.predict(features)
          score=np.sum(correct==1)
          total_score=len(correct)
          percent_scored=((score/total_score)*100).round(1)
          st.write ('Result is predicted as:')
          if percent_scored>=pass_percent:
               st.markdown(f"<h1 style='font-size:32px;'> PASS </h1>",unsafe_allow_html=True)
          elif percent_scored<pass_percent:
               st.markdown(f"<h1 style='font-size:32px;'> FAIL </h1>",unsafe_allow_html=True)
          st.write(f"**SCORE:** {percent_scored}%")
     else:
         st.error('Upload a file')
         
    
