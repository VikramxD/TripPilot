"""
Vector store for TripPilot RAG pipeline.

Stores and retrieves travel knowledge using ChromaDB or LanceDB.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
import uuid

import structlog

from trippilot.core.config import settings
from trippilot.rag.embeddings import EmbeddingService, get_embedding_service

logger = structlog.get_logger()


@dataclass
class Document:
    """A document in the vector store."""

    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A search result from the vector store."""

    document: Document
    score: float


class VectorStore:
    """Vector store for travel knowledge using ChromaDB."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_path: str | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the collection
            persist_path: Path to persist the database
            embedding_service: Optional custom embedding service
        """
        self.collection_name = collection_name or settings.collection_name
        self.persist_path = persist_path or settings.chromadb_path
        self.embedding_service = embedding_service or get_embedding_service()
        self._client = None
        self._collection = None

    def _ensure_initialized(self):
        """Ensure ChromaDB client and collection are initialized."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                # Create persist directory if needed
                Path(self.persist_path).mkdir(parents=True, exist_ok=True)

                self._client = chromadb.PersistentClient(
                    path=self.persist_path,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )

                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

                logger.info(
                    "Vector store initialized",
                    collection=self.collection_name,
                    path=self.persist_path,
                )
            except ImportError:
                logger.error("ChromaDB not installed. Install with: pip install chromadb")
                raise

    @property
    def collection(self):
        """Get the ChromaDB collection."""
        self._ensure_initialized()
        return self._collection

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """
        Add a document to the vector store.

        Args:
            content: Document content
            metadata: Optional metadata
            doc_id: Optional document ID

        Returns:
            Document ID
        """
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}

        # Generate embedding
        embedding = self.embedding_service.embed_text(content)

        # Add to collection
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )

        logger.debug("Added document", doc_id=doc_id)
        return doc_id

    def add_documents(
        self,
        documents: List[Document] | List[dict[str, Any]],
        batch_size: int = 100,
    ) -> List[str]:
        """
        Add multiple documents to the vector store.

        Args:
            documents: List of Document objects or dicts with content/metadata
            batch_size: Batch size for adding documents

        Returns:
            List of document IDs
        """
        doc_ids = []
        contents = []
        metadatas = []

        for doc in documents:
            if isinstance(doc, Document):
                doc_ids.append(doc.id)
                contents.append(doc.content)
                metadatas.append(doc.metadata)
            else:
                doc_id = doc.get("id") or str(uuid.uuid4())
                doc_ids.append(doc_id)
                contents.append(doc.get("content", ""))
                metadatas.append(doc.get("metadata", {}))

        # Generate embeddings in batch
        embeddings = self.embedding_service.embed_texts(contents, batch_size=batch_size)

        # Add in batches
        for i in range(0, len(doc_ids), batch_size):
            end = min(i + batch_size, len(doc_ids))
            self.collection.add(
                ids=doc_ids[i:end],
                documents=contents[i:end],
                embeddings=embeddings[i:end].tolist(),
                metadatas=metadatas[i:end],
            )

        logger.info("Added documents", count=len(doc_ids))
        return doc_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Build query kwargs
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            kwargs["where"] = filter_metadata

        # Execute search
        results = self.collection.query(**kwargs)

        # Parse results
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                document = Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                # Convert distance to similarity score (ChromaDB uses L2 distance for cosine)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - (distance / 2)  # Convert cosine distance to similarity
                search_results.append(SearchResult(document=document, score=score))

        logger.debug("Search completed", query=query[:50], num_results=len(search_results))
        return search_results

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug("Deleted document", doc_id=doc_id)
            return True
        except Exception as e:
            logger.error("Failed to delete document", doc_id=doc_id, error=str(e))
            return False

    def clear(self):
        """Clear all documents from the collection."""
        self._ensure_initialized()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared collection", collection=self.collection_name)

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def get_document(self, doc_id: str) -> Document | None:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        result = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )

        if result["ids"]:
            return Document(
                id=result["ids"][0],
                content=result["documents"][0] if result["documents"] else "",
                metadata=result["metadatas"][0] if result["metadatas"] else {},
            )
        return None
