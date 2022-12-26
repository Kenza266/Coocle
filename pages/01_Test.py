import streamlit as st
from Index import Index

st.set_page_config(layout='wide')

index = Index(('DS\\index.json', 'DS\\inverted.json', 'DS\\queries.json', 'DS\\ground_truth.csv', 'DS\\raw_queries.json', 'DS\\raw_docs.json'), preprocessed=True)
st.title('Metrics')

doc = st.number_input('Choose a document', min_value=0, max_value=len(index.index)-1)
st.write(index.raw_docs[str(doc)])
q = st.number_input('Choose a query', min_value=0, max_value=len(list(index.queries.values()))-1)
query = index.queries[str(q)] 
st.write(index.raw_queries[str(q)])

output = {'Scalar product': index.scalar_prod(str(doc), query), 
          'Cosine measure': index.cosine_measure(str(doc), query),
          'Jaccard measure': index.jaccard_measure(str(doc), query)}
st.write(output)

st.title('Boolean')
query = st.text_input('Enter a boolean query')
docs = index.parse_boolean_query(query)
if docs is not None:
    st.write('Number of relevent documents: ', len(docs))
else:
    st.write('Please rewrite the query')