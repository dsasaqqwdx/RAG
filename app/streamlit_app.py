"""
Simple Streamlit frontend for the arXiv/document RAG bot.
Talks to the FastAPI backend running separately (uvicorn app.main:app).

Run with: streamlit run streamlit_app.py
(Keep uvicorn running in a separate terminal at the same time.)
"""
import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Q&A Bot", page_icon="📄", layout="wide")

st.title("📄 Document Q&A Bot")
st.caption("Upload PDFs, then ask questions grounded in their content.")

# ---- Sidebar: upload new documents ----
with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader("Choose PDF(s)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if st.button(f"Ingest {len(uploaded_files)} file(s)"):
            with st.spinner(f"Ingesting {len(uploaded_files)} file(s)..."):
                files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                try:
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files_payload, timeout=600)
                    if response.status_code == 200:
                        for r in response.json():
                            if "Skipped" in r["message"]:
                                st.warning(f"⚠️ {r['filename']}: {r['message']}")
                            else:
                                st.success(f"✅ {r['filename']}: {r['chunks_stored']} chunks stored")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Is uvicorn running?")

    st.divider()
    st.header("Search scope")
    try:
        docs_response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        ingested_docs = docs_response.json() if docs_response.status_code == 200 else []
    except requests.exceptions.ConnectionError:
        ingested_docs = []

    doc_options = ["All documents"] + [d["filename"] for d in ingested_docs]
    selected_doc = st.selectbox("Ask about:", doc_options)

    st.divider()
    st.caption("Backend: FastAPI + Pinecone + Hybrid Search + Reranking")
# ---- Main area: chat-style Q&A ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past exchanges
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            with st.expander(f"📚 Sources ({len(entry['sources'])})"):
                for src in entry["sources"]:
                    page_info = f", page {src['page']}" if src.get("page") else ""
                    st.markdown(f"**{src['source']}**{page_info}")
                    st.caption(src["snippet"])

# Chat input
question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                f"{API_BASE_URL}/ask",
                json={
                    "question": question,
                    "source": None if selected_doc == "All documents" else selected_doc,
                },
                timeout=120,
            )
                if response.status_code == 200:
                    data = response.json()
                    st.write(data["answer"])

                    if data["sources"]:
                        with st.expander(f"📚 Sources ({len(data['sources'])})"):
                            for src in data["sources"]:
                                page_info = f", page {src['page']}" if src.get("page") else ""
                                st.markdown(f"**{src['source']}**{page_info}")
                                st.caption(src["snippet"])

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": data["answer"],
                        "sources": data["sources"],
                    })
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is uvicorn running at http://127.0.0.1:8000?")