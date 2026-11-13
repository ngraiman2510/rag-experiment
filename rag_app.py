import tempfile

import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()  # Load environment variables from .env file

import streamlit as st


def main():
    st.title("RAG App")

    # Upload PDF file
    uploaded_files = st.file_uploader(
        "Upload a PDF file", type="pdf", accept_multiple_files=True
    )

    if "store" not in st.session_state:
        st.session_state.store = None

    if uploaded_files:
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

        # Create embeddings and store them in FAISS
        embeddings = OpenAIEmbeddings()

        if st.button("Process PDFs"):
            chunks = []
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    # Create a temporary file to save the uploaded PDF
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    # Load the PDF file
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()

                    docs = text_splitter.split_documents(documents)
                    chunks.extend(docs)
                    st.success(f"Indexed {len(chunks)} chunks")

            if len(chunks):
                st.session_state.store = FAISS.from_documents(chunks, embeddings)
                st.success("Documents processed and indexed.")

    # User query input
    query = st.text_input(
        "Enter your Question:", disabled=st.session_state.store is None
    )
    if query and st.session_state.store is not None:
        # Retrieve relevant documents from vector store
        results = st.session_state.store.as_retriever().invoke(query)

        # Display the results
        st.write("Results:")
        for i, result in enumerate(results):
            st.write(f"Result {i + 1}:")
            st.write(result.page_content)
            st.caption(f"{result.metadata['source']} - page {result.metadata['page']}")
            st.write("---")


if __name__ == "__main__":
    main()
