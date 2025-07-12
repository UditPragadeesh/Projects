import streamlit as st
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title('Dropout Risk')
# Data Loading
file = st.file_uploader('Upload file',type='.csv')
# st.markdown("**Note:** Required columns for prediction are \n **'Displaced', 'Debtor', 'Tuition fees up to date', 'Scholarship holder'," \
# " 'Age at enrollment', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)'," \
# " 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)'," \
# " 'Curricular units 2nd sem (grade)', 'Application mode','Course', 'Mother's qualification', 'Father's qualification', 'Gender'**")
application_mode = pd.read_csv(r'data\Working_Data\Application_mode.csv')
course_name = pd.read_csv(r'data\Working_Data\course_names.csv')
mother_qual = pd.read_csv(r'data\Working_Data\mother_qual.csv')
father_qual = pd.read_csv(r'data\Working_Data\father_qual.csv')
# Loading trained model
model = nn.Sequential(nn.Linear(17,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1),nn.Sigmoid())
model.load_state_dict(torch.load(r'Models\dropout_model_weights.pth'))

# Find button
if 'pred_bt' not in st.session_state:
    st.session_state.pred_bt = False
if st.button('Find'):
    st.session_state.pred_bt = True

if st.session_state.pred_bt:
    if file:
        df = pd.read_csv(file)
        cols = ['Displaced', 'Debtor', 'Tuition fees up to date', 'Scholarship holder','Age at enrollment', 'Curricular units 1st sem (enrolled)',
                 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)','Curricular units 2nd sem (enrolled)', 
                 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)',
                'Application mode','Course', "Mother's qualification", "Father's qualification", 'Gender']
        for col in cols:
             if col not in list(df.columns):
                st.error(f'{col} not found in provided table \n Make sure {cols} are available in the table')
                st.stop()
        pred_df = pd.merge(df,application_mode,on='Application mode',how='left')
        pred_df = pd.merge(pred_df,course_name,on='Course',how='left')
        pred_df = pd.merge(pred_df,mother_qual,on="Mother's qualification",how='left')
        pred_df = pd.merge(pred_df,father_qual,on="Father's qualification",how='left')
        pred_df = pd.get_dummies(pred_df,columns=['Gender'],drop_first=True,dtype='int')
        feature_cols = ['Displaced', 'Debtor', 'Tuition fees up to date', 'Scholarship holder',
       'Age at enrollment', 'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)', 'Application mode_ID',
       'Course_ID', "Mother's qualification_ID", "Father's qualification_ID",
       'Gender_Male']
        X = pred_df[feature_cols].to_numpy()
        X = torch.Tensor(X)
        outputs = model(X)
        df['Dropout'] = outputs.detach().numpy()
        df['Dropout'] = [True if x>0.5 else False for x in df['Dropout']]
        dropout_df = df[df['Dropout']==True]
        dropout_df = dropout_df.reset_index(drop=True)
        st.write('Dropout Table',dropout_df)
        col1,col2= st.columns(2)
        csv = dropout_df.to_csv(index=False).encode('utf-8')
        col1.download_button('Download dropout table as csv',data = csv, file_name='Dropout_Table.csv',key='download1')
        csv1 = df.to_csv(index=False).encode('utf-8')
        col2.download_button('Download original file with dropout as csv',data = csv1, file_name='Dropout_risk.csv',key='download')
        if st.button('OK',key='ok'):
                 st.session_state.pred_bt=False
    else:
        st.error('Upload file')
        if st.button('OK',key='ok'):
                 st.session_state.pred_bt=False
