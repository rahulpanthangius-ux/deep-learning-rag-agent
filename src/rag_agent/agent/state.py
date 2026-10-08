"""
state.py
========
Data models and agent state definitions for the LangGraph agent.

All inter-component data structures are defined here to ensure
consistent interfaces across the entire codebase.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


# ---------------------------------------------------------------------------
# Corpus / Ingestion Models
# ---------------------------------------------------------------------------


@dataclass
class ChunkMetadata:
    """Metadata associated with a document chunk."""
    
    topic: str
    difficulty: str
    type: str
    source: str
    page_or_header: str | None = None  # <--- Add this line here
    related_topics: list[str] = field(default_factory=list)
    is_bonus: bool = False
    chunk_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict for ChromaDB metadata storage."""
        return {
            "topic": self.topic,
            "difficulty": self.difficulty,
            "type": self.type,
            "source": self.source,
            "page_or_header": self.page_or_header or "",  # <--- Include in dict
            "related_topics": ",".join(self.related_topics),
            "is_bonus": self.is_bonus,
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkMetadata:
        """Deserialise from ChromaDB metadata dict."""
        related = data.get("related_topics", "")
        related_list = [r.strip() for r in related.split(",") if r.strip()] if related else []
        return cls(
            topic=data.get("topic", "General"),
            difficulty=data.get("difficulty", "intermediate"),
            type=data.get("type", "concept_explanation"),
            source=data.get("source", "unknown"),
            page_or_header=data.get("page_or_header"),  # <--- Read from dict
            related_topics=related_list,
            is_bonus=bool(data.get("is_bonus", False)),
            chunk_index=int(data.get("chunk_index", 0)),
        )


@dataclass
class DocumentChunk:
    """
    A single unit of content ready for embedding and storage.

    Attributes
    ----------
    chunk_id : str
        Deterministic ID derived from a hash of (source + chunk_text).
        Identical content always produces the same ID — this is the
        foundation of duplicate detection.
    chunk_text : str
        The actual content to embed. Between 100 and 300 words.
    metadata : ChunkMetadata
        Structured metadata for filtering and attribution.
    """

    chunk_id: str
    chunk_text: str
    metadata: ChunkMetadata


@dataclass
class IngestionResult:
    """
    Summary of a single ingestion operation.

    Returned by VectorStoreManager.ingest() so the UI can display
    a meaningful status message to the user.

    Attributes
    ----------
    ingested : int
        Number of new chunks successfully added to the vector store.
    skipped : int
        Number of chunks skipped because they already existed
        (duplicate detection fired).
    errors : list[str]
        Human-readable error messages for any chunks that failed.
    document_ids : list[str]
        Unique document-level IDs for all successfully ingested files.
    """

    ingested: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total chunks processed (ingested + skipped + errored)."""
        return self.ingested + self.skipped + len(self.errors)

    @property
    def success(self) -> bool:
        """True if at least one chunk was ingested without error."""
        return self.ingested > 0 and len(self.errors) == 0


# ---------------------------------------------------------------------------
# Retrieval Models
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """
    A single chunk returned from a vector store query.

    Attributes
    ----------
    chunk_id : str
        The ID of the retrieved chunk.
    chunk_text : str
        The content of the retrieved chunk.
    metadata : ChunkMetadata
        Structured metadata for citation and display.
    score : float
        Similarity score between query and chunk (0.0 to 1.0).
        Higher is more similar.
    """

    chunk_id: str
    chunk_text: str
    metadata: ChunkMetadata
    score: float

    def to_citation(self) -> str:
        """
        Format this chunk as a source citation string for display
        in the UI and in LLM responses.

        Example output: '[LSTM | intermediate | hochreiter1997.pdf]'
        """
        return f"[{self.metadata.topic} | {self.metadata.difficulty} | {self.metadata.source}]"


# ---------------------------------------------------------------------------
# Agent Response Models
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """
    The structured response returned by the LangGraph agent.

    Attributes
    ----------
    answer : str
        The generated answer or interview question.
    sources : list[str]
        Citation strings for each chunk used to generate the answer.
        Empty list indicates the hallucination guard fired.
    confidence : float
        Average similarity score of retrieved chunks (0.0 to 1.0).
        Low confidence should be surfaced visibly in the UI.
    no_context_found : bool
        True when no chunks above the similarity threshold were found.
        The UI must display a clear "no relevant context" message
        rather than showing a low-confidence hallucinated answer.
    rewritten_query : str
        The query after rewriting by the query_rewrite_node.
        Useful for debugging retrieval quality.
    """

    answer: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    no_context_found: bool = False
    rewritten_query: str = ""


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class AgentState(MessagesState):
    """
    The state object passed between all nodes in the LangGraph agent graph.

    Inherits from MessagesState which provides the `messages` field
    (list[BaseMessage]) with built-in message accumulation behaviour.

    Additional fields track the intermediate results of each graph node
    so subsequent nodes have access to prior outputs.

    Attributes
    ----------
    messages : list[BaseMessage]
        Full conversation history. Inherited from MessagesState.
        Trimmed by the generation node when approaching max_context_tokens.
    original_query : str
        The raw query as submitted by the user.
    rewritten_query : str
        The query after processing by query_rewrite_node.
    retrieved_chunks : list[RetrievedChunk]
        Chunks returned by retrieval_node.
    no_context_found : bool
        Set to True by retrieval_node when similarity threshold
        is not met by any retrieved chunk.
    final_response : AgentResponse | None
        Populated by generation_node. None until generation completes.
    topic_filter : str | None
        Optional topic to restrict retrieval scope (e.g. "LSTM").
    difficulty_filter : str | None
        Optional difficulty to restrict retrieval scope.
    """

    original_query: str = ""
    rewritten_query: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    no_context_found: bool = False
    final_response: AgentResponse | None = None
    topic_filter: str | None = None
    difficulty_filter: str | None = None