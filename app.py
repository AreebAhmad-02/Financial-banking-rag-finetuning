# app.py
import streamlit as st
from rag_pipeline import get_rag_response, ingest_documents  # Assuming you have these

st.set_page_config(page_title="Banking Support Assistant", page_icon="💼", layout="centered")

st.title("💼 Banking Support Assistant")
st.subheader("Ask your banking-related questions below:")

# Sidebar upload
with st.sidebar:
    st.header("📄 Upload Additional Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])
    if uploaded_files:
        for file in uploaded_files:
            file_path = f"uploads/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.read())
        st.success("Files uploaded successfully!")
        if st.button("Ingest Documents"):
            ingest_documents(uploaded_files)
            st.success("Documents ingested into RAG pipeline.")

# Main Query Input
user_query = st.text_input("Enter your query:", placeholder="E.g., What are the requirements for a personal loan?")

if st.button("Submit Query") and user_query.strip():
    with st.spinner("Fetching response..."):
        response = get_rag_response(user_query)
        st.markdown("### 🧠 Response")
        st.write(response)
else:
    st.info("Please enter a query to receive support.")

# Optional Footer
st.markdown("---")
st.markdown("🔐 This tool is secure & confidential. Your data stays with you.")
