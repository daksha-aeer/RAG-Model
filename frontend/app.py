import streamlit as st
from backend.qa_system import process_query
import os
import sys 
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

st.title("Document-based Q&A System")

uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)
query = st.text_input("Enter your query")
# st.button("Get Answer")

if st.button("Get Answer"):
    if uploaded_files and query:
        # Process the query and uploaded documents
        answer, context = process_query(query, uploaded_files)
        st.write(f"**Answer**: {answer}")
        st.write(f"**Context**: {context}")
    else:
        st.warning("Please upload documents and enter a query.")
