import streamlit as st
from Index import Index

st.set_page_config(layout='wide')

index = Index(('index.json', 'inverted.json', 'queries.json', 'ground_truth.csv'), preprocessed=True)

st.title('Fonctions d\'apparaiment')

doc = st.number_input('Choose a document', min_value=0, max_value=len(index.index)-1)
q = st.number_input('Choose a query', min_value=0, max_value=len(list(index.queries.values()))-1)
query = index.queries[str(q)] 

output = {'Scalar product': index.scalar_prod(str(doc), query), 
          'Cosine measure': index.cosine_measure(str(doc), query),
          'Jaccard measure': index.jaccard_measure(str(doc), query)}
st.write(output)
