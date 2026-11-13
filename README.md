# PDF RAG App

A Streamlit app that lets you upload one or more PDFs, index their contents with OpenAI embeddings, and ask questions about them. Relevant text excerpts are displayed with their source file and page number.

## Requirements

- Python 3.10 or later
- An OpenAI API key

## Setup

From this folder, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install streamlit langchain-community langchain-openai langchain-text-splitters python-dotenv faiss-cpu pypdf
```

Create a `.env` file in this folder:

```env
OPENAI_API_KEY=your_openai_api_key
```

Keep your API key private; `.env` is excluded by `.gitignore`. OpenAI API usage may incur charges.

## Run

```bash
streamlit run rag_app.py
```

Upload PDF files, select **Process PDFs**, then enter a question. The vector index is kept in the current Streamlit session and is not saved between sessions.
