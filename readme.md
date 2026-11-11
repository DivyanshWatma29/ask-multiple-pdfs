# Multi-PDF RAG Chatbot (Ollama)

A simple personal learning project for asking questions across multiple PDF files. It uses **Python, Streamlit, LangChain, FAISS, and Ollama**. Chat generation and embeddings run through an Ollama server rather than an OpenAI API key.

## Features

- Upload and process multiple text-based PDFs.
- Split extracted PDF text into overlapping chunks.
- Create local embeddings with `nomic-embed-text` and index them with FAISS.
- Retrieve relevant text and generate answers using `llama3.2:3b`.
- Follow up within the same chat session and inspect retrieved document/page references.

## How it works

PDFs → text extraction → chunking → Ollama embeddings → FAISS index → relevant chunks → Ollama chat model → answer.

## Run locally

1. Install [Python](https://www.python.org/downloads/) (3.11 recommended) and [Ollama](https://ollama.com/download).
2. Start Ollama (the desktop application or `ollama serve`).
3. Download the two models:

   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

6. Upload PDFs, select **Process PDFs**, and ask a question.

By default, the app connects to `http://localhost:11434`. You can configure `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, and `OLLAMA_EMBED_MODEL` in your environment, or edit the Ollama server URL in the sidebar. Do not expose an unauthenticated Ollama server to the public internet.

## Live demo / hosting

**This repository is ready for a local demo. It is not automatically a public live demo.** Streamlit Community Cloud hosts the Streamlit UI but does not provide a local Ollama model server. For an online deployment, you need a separate always-on Ollama host reachable by the Streamlit app, or a supported remote model API. Avoid pointing a cloud app at `localhost:11434` unless Ollama is running on that exact same host.

## Notes

- Scanned PDFs without selectable text require OCR (not included).
- Responses may be incomplete or incorrect; verify important facts against the original PDF.
- Documents, text chunks, and the FAISS index are kept in app session memory; this version does not persist uploads to a database.
- This is an adaptation of [Alejandro AO's ask-multiple-pdfs project](https://github.com/alejandro-ao/ask-multiple-pdfs), built as a learning exercise. The original project is MIT licensed; see its repository for license information and the [original tutorial](https://youtu.be/dXxQ0LR-3Hg).
