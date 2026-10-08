"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the singleton VectorStoreManager."""
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """Initialise all st.session_state keys on first run."""
    defaults = {
        "chat_history": [],  # list of dicts with role, content, sources, no_context_found
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """Render the document ingestion panel in the sidebar."""
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Ingest Documents", disabled=not uploaded_files):
        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = Path(temp_dir) / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(temp_path)

                try:
                    chunks = chunker.chunk_files(temp_paths)
                    result = store.ingest(chunks)
                    st.session_state["last_ingestion_result"] = result
                    
                    # Safely fetch result attributes with fallbacks
                    ingested = getattr(result, "ingested", len(chunks))
                    skipped = getattr(result, "skipped", 0)
                    errored = getattr(result, "errored", 0)
                    
                    success_msg = f"✅ Added {ingested} chunks. Skipped {skipped} duplicates."
                    if errored > 0:
                        success_msg += f" ({errored} errors)"
                        
                    st.sidebar.success(success_msg)
                except Exception as e:
                    st.sidebar.error(f"Ingestion failed: {e}")

    # Render ingested documents list
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Ingested Documents")
    docs = store.list_documents()
    st.session_state["ingested_documents"] = docs

    if not docs:
        st.sidebar.info("No documents ingested yet.")
    else:
        for doc in docs:
            source = doc["source"]
            topic = doc["topic"]
            count = doc["chunk_count"]
            col_info, col_btn = st.columns([4, 1])
            with col_info:
                st.sidebar.text(f"{source}\n{topic} ({count} chunks)")
            with col_btn:
                if st.sidebar.button("🗑", key=f"del_{source}", help=f"Remove {source}"):
                    store.delete_document(source)
                    st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """Render a compact corpus health summary in the sidebar."""
    stats = store.get_collection_stats()
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    if stats["topics"]:
        st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.info("💡 Add bonus topics (SOM, GAN, etc.) for extra depth.")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """Render the document viewer in the main centre column."""
    st.subheader("📄 Document Viewer")

    docs = st.session_state.get("ingested_documents", [])
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [doc["source"] for doc in docs]
    selected_source = st.selectbox(
        "Select document to inspect",
        options=sources,
        index=0 if sources else None,
    )

    if selected_source:
        st.session_state["selected_document"] = selected_source
        chunks = store.get_document_chunks(selected_source)

        st.caption(f"Showing {len(chunks)} chunks for **{selected_source}**")

        with st.container(height=500):
            for chunk in chunks:
                meta = chunk.metadata
                with st.expander(
                    f"Chunk {meta.chunk_index} | Topic: {meta.topic} | Difficulty: {meta.difficulty}"
                ):
                    st.markdown(chunk.chunk_text)


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """Render the chat interface in the right column."""
    st.subheader("💬 Interview Prep Chat")

    # Filters
    stats = get_vector_store().get_collection_stats()
    available_topics = ["All"] + stats.get("topics", [])

    col_topic, col_diff = st.columns(2)
    with col_topic:
        chosen_topic = st.selectbox("Topic Filter", options=available_topics)
        st.session_state["topic_filter"] = None if chosen_topic == "All" else chosen_topic
    with col_diff:
        chosen_diff = st.selectbox(
            "Difficulty Filter", options=["All", "beginner", "intermediate", "advanced"]
        )
        st.session_state["difficulty_filter"] = None if chosen_diff == "All" else chosen_diff

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    if query := st.chat_input("Ask about a deep learning topic..."):
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Invoke LangGraph agent
        inputs = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        try:
            result = graph.invoke(inputs, config=config)
            final_response: AgentResponse = result.get("final_response")

            if final_response:
                answer = final_response.answer
                sources = final_response.sources
                no_context = final_response.no_context_found
            else:
                answer = "I'm sorry, I couldn't generate a response."
                sources = []
                no_context = True

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "no_context_found": no_context,
                }
            )
        except Exception as e:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Error communicating with agent: {e}",
                    "sources": [],
                    "no_context_found": True,
                }
            )

        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()