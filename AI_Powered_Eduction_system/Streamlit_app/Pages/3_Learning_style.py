import streamlit as st
import pandas as pd
import pickle

st.title('Learning Style')

file = st.file_uploader('Upload data file here:',type='.csv')
pipe = pickle.load(open(r'Models\Learning_style.sav','rb'))
cluster_names = pd.read_csv(r'data\Working_Data\Cluster_names.csv') 
def predict(df):
    df1 = df[['attempt_count','num_of_videos','num_of_books','time_on_topic','avg_score','avg_timetaken']]
    df['cluster'] = pipe.predict(df1)
    df2 = pd.merge(df,cluster_names,on='cluster',how='left')
    df2= df2[['Student_ID','Learning_style']]
    return df2
def filter(data,*args):
    df = data
    for col in args:
        ops = list(df[col].unique())
        selection = st.multiselect(label=f'{col}:',options=ops)
        if len(selection) != 0:
            df = df[df[col].isin(selection)]
    return df
if 'predict_b' not in st.session_state:
    st.session_state.predict_b=False
if st.button('Find Learning style'):
    st.session_state.predict_b=True

if st.session_state.predict_b:
    try:
        x_df = pd.read_csv(file)
        learning_style_df = predict(x_df)
        result_df = filter(learning_style_df,'Student_ID')
        st.write(result_df)
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download table as csv',data = csv, file_name='Learning_style.csv',key='download')
        if st.button('OK'):
            st.session_state.predict_b=False
    except ValueError:
        st.error('Upload File')
        if st.button('OK'):
            st.session_state.predict_b=False


