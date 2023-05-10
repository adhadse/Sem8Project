import streamlit as st
import pandas as pd

from bert_classifier import BERTclassifier

BASE_DIR = "/mnt/Data/pycharm_repo/Kavida/tasks/dummy_project"

st.title("Article Filtration Pipeline")

st.markdown("""
This pipeline can reject pipeline based on a news Article.
""")

news_title = st.text_input("News Title")
client_name = st.selectbox(
    'Client Name', 
    ('Client2-Deloitte', 'Client44-Amazon')
)

def execute():
    classify = BERTclassifier(base_dir=BASE_DIR)
    with st.spinner("Processing"):
        result = classify.predict(news_title)

    print(result)
    if result[0][0] >= 0.5:
        st.error("This title should be Accepted")
    else:
        st.success("This title should be Rejected")

st.button('Execute', on_click=execute)
