"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager
from rag_agent.config import Settings, get_settings


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(path, chunk_size, chunk_overlap)
        elif suffix in [".md", ".markdown"]:
            raw_chunks = self._chunk_markdown(path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Supported: .pdf, .md")

        metadata = self._infer_metadata(path, metadata_overrides)

        documents = []
        for idx, item in enumerate(raw_chunks):
            text = item.get("text", "")
            if not text.strip():
                continue

            # Update chunk-specific metadata copy
            chunk_meta = ChunkMetadata(
                source=metadata.source,
                topic=metadata.topic,
                difficulty=metadata.difficulty,
                is_bonus=metadata.is_bonus,
                chunk_index=idx,
                page_or_header=item.get("page") or item.get("header"),
            )

            chunk_id = VectorStoreManager.generate_chunk_id(metadata.source, text)
            documents.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=chunk_meta,
                )
            )

        logger.info(f"Successfully chunked '{path.name}' into {len(documents)} chunks.")
        return documents

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        all_chunks = []
        for path in file_paths:
            try:
                chunks = self.chunk_file(path, metadata_overrides=metadata_overrides)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.exception(f"Failed to chunk file {path}: {e}")
        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        results = []
        for page_num, page_doc in enumerate(pages, start=1):
            splits = text_splitter.split_text(page_doc.page_content)
            for split in splits:
                results.append({"text": split, "page": page_num})
        return results

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file using robust recursive character splitting.
        Guarantees all text is captured accurately without header parsing edge-cases.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        splits = text_splitter.split_text(markdown_content)
        
        results = []
        for split in splits:
            if split.strip():
                results.append({"text": split, "header": "General"})
                
        return results

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        overrides = overrides or {}
        stem = file_path.stem  # filename without extension
        parts = stem.split("_")

        inferred_topic = parts[0].capitalize() if parts else "General"
        inferred_difficulty = parts[1].lower() if len(parts) > 1 else "intermediate"

        # Check for valid difficulties
        valid_difficulties = {"beginner", "intermediate", "advanced"}
        if inferred_difficulty not in valid_difficulties:
            inferred_difficulty = "intermediate"

        # Bonus topics check
        bonus_topics = {"som", "boltzmannmachine", "gan", "autoencoder"}
        is_bonus = inferred_topic.lower() in bonus_topics or overrides.get("is_bonus", False)

        source = file_path.name
        topic = overrides.get("topic", inferred_topic)
        difficulty = overrides.get("difficulty", inferred_difficulty)

        return ChunkMetadata(
            source=source,
            topic=topic,
            difficulty=difficulty,
            is_bonus=is_bonus,
            chunk_index=0,
            
        )