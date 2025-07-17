import streamlit as st
import pandas as pd 
import pickle
import sqlite3 as sql

st.title('Score Predictor')

type = st.selectbox('Select type of prediction:',options=['Single Prediction','Multiple Prediction'])
con = sql.connect(r'data\Train_datasets\Pass_Fail\AI_Edu_system.db')
cur = con.cursor()
skills = pd.read_sql('SELECT * FROM skills',con)
Type = pd.read_sql('SELECT * FROM problemtype',con)
Type = Type.drop(index=[0,1],axis=0).reset_index(drop=True)
student_data = pd.read_csv(r'data\Working_Data\score_data_table.csv')
skill_opts = list(skills['skill'])
test_opts = list(Type['problemType'])
pipe = pickle.load(open(r'Models\Score_predictor.sav','rb'))

def predict(df):
   df1 = df
   result_df = df.merge(skills,on='skill',how = 'left')
   result_df = result_df[['Type','Days_prepared','TimeTaken','skill_ID']]
   df1['score'] = pipe.predict(result_df).round(2)
   return df1
i = 0
if type == 'Single Prediction':
    student_data['skill'] = st.selectbox("What is the student's skill?:",options=skill_opts,key=f'skill{i}')
    student_data['Days_prepared'] = st.number_input('How many days did the student prepare?:',value = 5,min_value=0,step = 1,key=f'dp{i}')
    student_data['TimeTaken'] = st.number_input('What is the duration of test in minutes?:',min_value=10,max_value=240,value = 60,key=f'TT{i}')
    student_data['Type'] = st.selectbox("What is the test type?:",options=test_opts,key=f'typ{i}')
    student_data=student_data[student_data.index==0]
    if st.button('Predict Score'):
        score_df = predict(student_data)
        st.markdown(f"Predicted score is :\n <h1 style='font-size:32px;'>{score_df['score'].iloc[0]}</h1>",unsafe_allow_html=True)
if type == 'Multiple Prediction':
    file = st.file_uploader('Upload data file here:',type='.csv')
    st.markdown("**Note:** key columns for prediction are ('Type','Days_prepared','TimeTaken','skill')")
    if 'predict_button' not in st.session_state:
        st.session_state.predict_button=False
    if st.button('Predict Score'):
        st.session_state.predict_button=True
    if st.session_state.predict_button:
        try:
            data = pd.read_csv(file)
            result_data= data
            result_data = predict(result_data)
            st.write(result_data)
            csv = result_data.to_csv(index=False).encode('utf-8')
            st.download_button('Download table as csv',data = csv, file_name='Predicted_scores.csv',key='download')
        except ValueError:
           st.error('Upload file')
           if st.button('OK',key='ok'):
                 st.session_state.predict_button=False
            
        
    


        

    
