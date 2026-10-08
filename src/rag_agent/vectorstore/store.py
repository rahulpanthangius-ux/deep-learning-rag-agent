"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        try:
            Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._settings.chroma_db_path)
            
            # Appended '_v2' here to instantly bypass old cache and create a fresh collection
            collection_name = f"{self._settings.chroma_collection_name}_v2"
            
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            count = self._collection.count()
            logger.info(
                f"ChromaDB initialised successfully at '{self._settings.chroma_db_path}' "
                f"with collection '{collection_name}' (items: {count})"
            )
        except Exception as e:
            logger.exception("Failed to initialise ChromaDB vector store.")
            raise RuntimeError(f"Could not initialise ChromaDB: {e}") from e

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        try:
            existing = self._collection.get(ids=[chunk_id])
            return bool(existing and existing["ids"])
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            valid_chunks = []
            
            for chunk in batch:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue
                valid_chunks.append(chunk)

            if not valid_chunks:
                continue

            try:
                texts = [c.chunk_text for c in valid_chunks]
                ids = [c.chunk_id for c in valid_chunks]
                embeddings = self._embeddings.embed_documents(texts)
                metadatas = [c.metadata.to_dict() for c in valid_chunks]

                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                result.ingested += len(valid_chunks)
            except Exception as e:
                logger.exception(f"Error ingesting chunk batch: {e}")
                if hasattr(result, "errored"):
                    result.errored += len(valid_chunks)
                else:
                    setattr(result, "errored", len(valid_chunks))

        errored_count = getattr(result, "errored", 0)
        logger.info(
            f"Ingestion complete. Ingested: {result.ingested}, "
            f"Skipped: {result.skipped}, Errored: {errored_count}"
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        k = k or self._settings.retrieval_k
        
        where_filter = {}
        if topic_filter:
            where_filter["topic"] = topic_filter
        if difficulty_filter:
            where_filter["difficulty"] = difficulty_filter
        
        query_filter = where_filter if where_filter else None

        try:
            query_embedding = self._embeddings.embed_query(query_text)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=query_filter,
                include=["documents", "metadatas", "distances"]
            )

            retrieved = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                score = 1.0 - dist  # Cosine distance to similarity score conversion
                if score >= self._settings.similarity_threshold:
                    metadata = ChunkMetadata.from_dict(meta)
                    retrieved.append(
                        RetrievedChunk(
                            chunk_id="",
                            chunk_text=doc,
                            metadata=metadata,
                            score=score,
                        )
                    )

            retrieved.sort(key=lambda x: x.score, reverse=True)
            return retrieved
        except Exception as e:
            logger.exception(f"Error querying vector store: {e}")
            return []

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        try:
            data = self._collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            
            doc_map = {}
            for meta in metadatas:
                if not meta:
                    continue
                source = meta.get("source", "unknown")
                topic = meta.get("topic", "General")
                if source not in doc_map:
                    doc_map[source] = {"source": source, "topic": topic, "chunk_count": 0}
                doc_map[source]["chunk_count"] += 1

            return sorted(doc_map.values(), key=lambda x: x["source"])
        except Exception as e:
            logger.exception(f"Error listing documents: {e}")
            return []

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        try:
            data = self._collection.get(
                where={"source": source},
                include=["documents", "metadatas", "ids"]
            )
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            chunks = []
            for cid, doc, meta in zip(ids, documents, metadatas):
                metadata = ChunkMetadata.from_dict(meta)
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        chunk_text=doc,
                        metadata=metadata,
                    )
                )
            # Order chunks by chunk_index if present
            chunks.sort(key=lambda c: c.metadata.chunk_index)
            return chunks
        except Exception as e:
            logger.exception(f"Error getting document chunks for {source}: {e}")
            return []

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        try:
            total_chunks = self._collection.count()
            data = self._collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])

            topics = set()
            sources = set()
            for meta in metadatas:
                if meta:
                    if meta.get("topic"):
                        topics.add(meta.get("topic"))
                    if meta.get("source"):
                        sources.add(meta.get("source"))

            return {
                "total_chunks": total_chunks,
                "topics": sorted(list(topics)),
                "sources": sorted(list(sources)),
                "bonus_topics_present": len(topics) > 1,
            }
        except Exception as e:
            logger.exception(f"Error getting collection stats: {e}")
            return {"total_chunks": 0, "topics": [], "sources": [], "bonus_topics_present": False}

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        try:
            data = self._collection.get(where={"source": source}, include=[])
            ids = data.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunks for source: {source}")
                return len(ids)
            return 0
        except Exception as e:
            logger.exception(f"Error deleting document {source}: {e}")
            return 0