import streamlit as st

st.set_page_config(layout='wide')

models = ['Scalar', 'Cosine', 'Jaccard', 'BM25', 'DataMining'] # 'Bool'
to_display = st.multiselect(
    'Choose the models to display',
    models, models)