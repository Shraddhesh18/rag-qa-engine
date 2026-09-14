import streamlit as st
import os
from rag_pipeline import process_pdf, get_qa_chain

st.set_page_config(page_title="RAG PDF Q&A Engine", layout="wide")

st.title("📄 RAG Document Q&A Engine")
st.write("Upload a PDF document to retrieve exact, context-backed answers.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    temp_file_path = f"temp_{uploaded_file.name}"

    # Check if a new file was uploaded or if it's already indexed in session state
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success("File uploaded successfully.")

        # Process PDF and initialize QA Chain
        with st.spinner("Chunking text & indexing into vector database..."):
            try:
                vectorstore = process_pdf(temp_file_path)
                qa_chain = get_qa_chain(vectorstore)
                
                # Store in session state to prevent reprocessing on every user interaction
                st.session_state["qa_chain"] = qa_chain
                st.session_state["current_file"] = uploaded_file.name
                st.sidebar.info("Indexing complete.")
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.stop()
            finally:
                # Safely attempt temporary file removal with Windows file-lock protection
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except PermissionError:
                        pass
    else:
        st.sidebar.info("Indexing complete.")

    # Interface for user queries
    user_query = st.text_input("Ask a question about the document:")

    if user_query and "qa_chain" in st.session_state:
        with st.spinner("Searching document context & generating response..."):
            response = st.session_state["qa_chain"].invoke({"input": user_query})
            
            st.markdown("### Answer:")
            st.write(response["answer"])

            # Render exact vector chunks retrieved from ChromaDB
            with st.expander("View Retrieved Context Chunks"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                    st.write(f"_{doc.page_content}_")
                    st.divider()

else:
    # Clear session state if file is removed
    st.session_state.pop("qa_chain", None)
    st.session_state.pop("current_file", None)
    st.info("Upload a PDF file using the sidebar to start asking questions.")