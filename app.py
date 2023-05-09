import streamlit as st
import pandas as pd
from keyword_article_rejection_pipeline import RejectionPipeline

BASE_DIR = "/mnt/Data/pycharm_repo/Kavida/tasks/dummy_project"

st.title("Keyword based Article Filtration Pipeline")

st.markdown("""
This pipeline can reject pipeline based on identified keywords as rejectable in the title
of a news Article. Provide a 
""")

news_title = st.text_input("News Title")
client_name = st.selectbox(
    'Client Name', 
    ('Client2-Deloitte', 'Client44-Amazon')
)

def execute():
    rejection_pipeline = RejectionPipeline(base_dir=BASE_DIR)
    df = pd.DataFrame({
        "news_title": [news_title],
        "client_name": [client_name]
    })
    accepted_rejected_series = rejection_pipeline.reject_article_based_on_keywords_mvp(df)
    if accepted_rejected_series[0] == 0:
        st.error("This title should be rejected")
    else:
        st.success("This title should be Accepted")

st.button('Execute', on_click=execute)
