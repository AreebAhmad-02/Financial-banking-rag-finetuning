import json
import streamlit as st
from rag_pipeline import create_rag_pipeline
from utils import extract_text_from_pdf, extract_text_from_docx, chunk_text_by_character
from guardrail.guards import input_guard, output_guard, validate_with_guard

st.set_page_config(page_title="NUST Banking Support Assistant", page_icon="💼", layout="centered")

st.title("💼 Banking Support Assistant")
st.subheader("Ask your banking-related questions below:")

# --- Session State for Pipeline ---
if "pipeline" not in st.session_state:
    with st.spinner("Ingesting initial documents..."):
        pipeline = create_rag_pipeline()
        pipeline.ingest_documents(json_file="chunks/header/all_chunks.json")
    st.session_state.pipeline = pipeline
    st.success("Initial documents ingested and system is ready!")

pipeline = st.session_state.pipeline

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
            # Read and chunk each file
            new_chunks = {}
            for file in uploaded_files:
                file_path = f"uploads/{file.name}"
                st.write(f"Processing file: {file.name}")
                if file.name.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                    st.write(f"Extracted text from PDF ({file.name}):", text[:500])  # Show first 500 chars
                    print("Extracted text from PDF", text[:500])
                elif file.name.lower().endswith(".docx"):
                    text = extract_text_from_docx(file_path)
                    st.write(f"Extracted text from DOCX ({file.name}):", text[:500])
                    print("Extracted text from DOCX", text[:500])
                elif file.name.lower().endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    st.write(f"Extracted text from TXT ({file.name}):", text[:500])
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    continue  # Skip unsupported file types

                chunks = chunk_text_by_character(text)
                print(f"Number of chunks for {file.name}: {len(chunks)}")
                st.write(f"Number of chunks for {file.name}: {len(chunks)}")
                if chunks:
                    st.write(f"First chunk for {file.name}:", chunks[0][:500])
                    print(f"First chunk for {file.name}:", chunks[0][:500])
                new_chunks[file.name] = chunks

            # Load existing JSON
            json_path = "chunks/header/all_chunks.json"
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    all_chunks = json.load(f)
                st.write("Loaded existing chunks from JSON.")
                print("Loaded existing chunks from JSON.")
                print("Existing all_chunks keys:", list(all_chunks.keys()))
            except (FileNotFoundError, json.JSONDecodeError):
                all_chunks = {}
                st.write("No existing JSON found, starting fresh.")

            # Append new chunks
            all_chunks.update(new_chunks)
            st.write("Updated all_chunks keys:", list(all_chunks.keys()))

            # Save back to JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            st.write(f"Saved updated chunks to {json_path}")

            # Ingest into pipeline
            pipeline.ingest_documents(json_file=json_path)
            st.success("Documents ingested into RAG pipeline.")


# --- Main Query Input ---
user_query = st.text_input("Enter your query:", placeholder="E.g., What are the requirements for a personal loan?")

if st.button("Submit Query") and user_query.strip():
    with st.spinner("Validating your query..."):
        validated_query = validate_with_guard(input_guard, user_query)
    if not validated_query:
        st.error("❌ Your query did not pass our safety checks. Please rephrase and try again.")
    else:
        with st.spinner("Fetching response..."):
            response = pipeline.get_response(user_query)
            validated_response = validate_with_guard(output_guard, response)
        if not validated_response:
            st.error("❌ The generated response did not pass our safety checks. Please try a different query.")
        else:
            st.markdown("### 🧠 Response")
            st.write(response)
else:
    st.info("Please enter a query to receive support.")

# --- Optional Footer ---
st.markdown("---")
st.markdown("🔐 This tool is secure & confidential. Your data stays with you.")