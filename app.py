"""Multi-PDF RAG chatbot using local Ollama models."""
import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_pdf_text(pdf_docs):
    """Read text from uploaded PDF files, preserving each source name."""
    documents = []
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page_number, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    documents.append({
                        "text": extracted,
                        "source": pdf.name,
                        "page": page_number,
                    })
        except Exception as exc:
            st.warning(f"Could not read {pdf.name}: {exc}")
    return documents


def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = []
    for document in documents:
        for chunk in splitter.split_text(document["text"]):
            chunks.append((chunk, {"source": document["source"], "page": document["page"]}))
    return chunks


def get_vectorstore(chunks, base_url):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=base_url)
    return FAISS.from_texts(
        texts=[chunk for chunk, _ in chunks],
        embedding=embeddings,
        metadatas=[metadata for _, metadata in chunks],
    )


def get_conversation_chain(base_url):
    return ChatOllama(
        model=CHAT_MODEL,
        temperature=0.1,
        base_url=base_url,
    )


def handle_userinput(question):
    if "vectorstore" not in st.session_state:
        st.info("Upload and process some PDFs first.")
        return

    # Include recent conversation turns when searching for follow-up questions.
    recent_history = st.session_state.messages[-6:]
    search_question = question
    if recent_history:
        history = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent_history
        )
        search_question = f"{history}\nCurrent question: {question}"

    with st.spinner("Searching your PDFs..."):
        docs = st.session_state.vectorstore.similarity_search(search_question, k=4)
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}, "
            f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )
        history_messages = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent_history
        )
        prompt = (
            "You answer questions based on the supplied PDF excerpts only. "
            "If the answer is not in the excerpts, say you could not find it "
            "in the uploaded documents. Do not invent facts. "
            "Keep the answer clear and concise.\n\n"
            f"Conversation history:\n{history_messages}\n\n"
            f"PDF excerpts:\n{context}\n\n"
            f"Question: {question}"
        )
        answer = st.session_state.llm.invoke(prompt).content

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.sources = [
        (doc.metadata.get("source", "unknown"), doc.metadata.get("page", "?"))
        for doc in docs
    ]


def main():
    st.set_page_config(page_title="Multi-PDF RAG Chatbot", page_icon="📚")
    st.title("📚 Multi-PDF RAG Chatbot")
    st.caption("Ask questions about multiple PDFs using local Ollama models.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []

    with st.sidebar:
        st.subheader("Documents")
        base_url = st.text_input(
            "Ollama server URL", value=DEFAULT_BASE_URL,
            help="The Ollama server must be reachable from the computer hosting Streamlit.",
        )
        st.caption(f"Chat model: {CHAT_MODEL} · Embeddings: {EMBED_MODEL}")
        pdf_docs = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Process PDFs", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Extracting text and creating embeddings..."):
                    documents = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(documents)
                    if not chunks:
                        st.error("No readable text found. Scanned PDFs may need OCR.")
                    else:
                        try:
                            st.session_state.vectorstore = get_vectorstore(chunks, base_url)
                            st.session_state.llm = get_conversation_chain(base_url)
                            st.session_state.messages = []
                            st.session_state.sources = []
                            st.success(f"Processed {len(pdf_docs)} PDF(s) into {len(chunks)} chunks.")
                        except Exception as exc:
                            st.error(
                                "Could not connect to Ollama or build the index. "
                                "Make sure Ollama is running and both models are downloaded. "
                                f"Details: {exc}"
                            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about your PDFs")
    if question:
        try:
            handle_userinput(question)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not generate an answer: {exc}")
    if st.session_state.sources:
        with st.expander("Retrieved sources (may include unrelated passages)"):
            for source, page in dict.fromkeys(st.session_state.sources):
                st.write(f"{source} — page {page}")


if __name__ == "__main__":
    main()
