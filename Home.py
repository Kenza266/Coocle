from Index import Index 
import streamlit as st
import json
import pandas as pd

st.set_page_config(layout='wide')

with open('index.json') as f:
    index = json.load(f)
    
with open('inverted.json') as f:
    inverted = json.load(f)

with open('queries.json') as f:
    queries = json.load(f) 

ground_truth = pd.read_csv('ground_truth.csv', sep=',')

index = Index((index, inverted, queries, ground_truth), preprocessed=True)

col1, col2 = st.columns(2)

with col1:
    st.title('Index')
    #st.write(index.index)
    doc = st.number_input('Choose a document', min_value=0, max_value=len(index.index)-1)
    out1 = pd.DataFrame(index.index[str(doc)])
    out1.columns = ['token', 'Frequency', 'Weight']
    st.dataframe(out1)

with col2:
    st.title('Inverted')
    #st.write(index.inverted)
    token = st.text_input('Enter a token')
    try:
        out2 = pd.DataFrame(index.get_docs(token))
        #out2 = out2.transpose()
        out2.columns = ['Document', 'Frequency', 'Weight']
        st.dataframe(out2)
    except:
        pass 
        #st.write('Token not found')

    query = st.text_input('Enter a query')
    
    try:
        details, all = index.get_docs_query(query)
        output1 = pd.DataFrame(details)
        output2 = pd.DataFrame(all)
        output2 = output2.transpose()
        output2.columns = ['Frequency', 'Weight']

        output = pd.concat([output1, output2], axis=1)
        #st.dataframe(output1)
        #st.dataframe(output2)
        st.dataframe(output)
    
    except:
        pass
