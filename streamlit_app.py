# streamlit_app.py - Simple web UI for the retrieval chatbot
import streamlit as st
import pandas as pd
from preprocess import Preprocessor
from pathlib import Path

from chatbot import build_index, answer_query

st.set_page_config(page_title='NLP Chatbot', layout='centered')
st.title('Task-3: NLP Chatbot (Retrieval-based)')

st.markdown('Enter your question and the chatbot will return the best matching answer from the knowledge base.')

kb_path = 'knowledge_base.csv'
state = build_index(kb_path)

with st.form('ask_form'):
    q = st.text_input('Your question', value='What is this chatbot?')
    submitted = st.form_submit_button('Ask')

if submitted and q.strip():
    matches = answer_query(state, q, top_k=3)
    if matches:
        top_score, top_answer, top_question = matches[0]
        st.success(f'Answer (score={top_score:.3f}): {top_answer}')
        if len(matches) > 1:
            st.write('Other close matches:')
            for s, ans, ques in matches[1:]:
                st.write(f'- (score={s:.3f}) Q: {ques} -> A: {ans}')
    else:
        st.error('No good match found. Consider adding this question to the knowledge base.')

st.sidebar.header('Knowledge base')
if Path(kb_path).exists():
    df = pd.read_csv(kb_path)
    st.sidebar.dataframe(df)
else:
    st.sidebar.write('No knowledge base found.')
