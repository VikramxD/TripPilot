"""
Knowledge retriever for TripPilot RAG pipeline.

Retrieves relevant travel knowledge to augment LLM responses.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, List

import structlog

from trippilot.rag.vector_store import VectorStore, SearchResult

logger = structlog.get_logger()


@dataclass
class RetrievalResult:
    """Result from knowledge retrieval."""

    query: str
    documents: List[dict[str, Any]]
    context: str
    sources: List[str]


class TravelKnowledgeRetriever:
    """Retriever for travel knowledge from the vector store."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        default_top_k: int = 5,
        min_score_threshold: float = 0.5,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store instance
            default_top_k: Default number of results to retrieve
            min_score_threshold: Minimum similarity score to include results
        """
        self.vector_store = vector_store or VectorStore()
        self.default_top_k = default_top_k
        self.min_score_threshold = min_score_threshold

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        destination: str | None = None,
        category: str | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant travel knowledge.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            destination: Optional filter by destination
            category: Optional filter by category

        Returns:
            RetrievalResult with documents and formatted context
        """
        top_k = top_k or self.default_top_k

        # Build metadata filter
        filter_metadata = {}
        if destination:
            filter_metadata["destination"] = destination
        if category:
            filter_metadata["category"] = category

        # Run search in thread pool (vector store operations are sync)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.vector_store.search(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata if filter_metadata else None,
            ),
        )

        # Filter by score threshold
        filtered_results = [
            r for r in results if r.score >= self.min_score_threshold
        ]

        # Format results
        documents = []
        sources = []
        context_parts = []

        for i, result in enumerate(filtered_results, 1):
            doc = result.document
            documents.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": result.score,
            })

            # Extract source if available
            source = doc.metadata.get("source", doc.metadata.get("url", f"Document {doc.id}"))
            sources.append(source)

            # Build context
            context_parts.append(
                f"[{i}] {doc.content}\n"
                f"   (Score: {result.score:.2f}, Source: {source})"
            )

        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        logger.info(
            "Retrieved knowledge",
            query=query[:50],
            num_results=len(filtered_results),
        )

        return RetrievalResult(
            query=query,
            documents=documents,
            context=context,
            sources=list(set(sources)),
        )

    async def retrieve_for_destination(
        self,
        destination: str,
        aspects: List[str] | None = None,
    ) -> dict[str, RetrievalResult]:
        """
        Retrieve knowledge about a destination across multiple aspects.

        Args:
            destination: The destination to search for
            aspects: Aspects to search (default: attractions, restaurants, tips, etc.)

        Returns:
            Dict mapping aspect to retrieval result
        """
        aspects = aspects or [
            "attractions",
            "restaurants",
            "hotels",
            "local tips",
            "transportation",
            "culture",
        ]

        results = {}
        for aspect in aspects:
            query = f"{destination} {aspect}"
            result = await self.retrieve(
                query=query,
                destination=destination.lower(),
            )
            results[aspect] = result

        return results

    async def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve with additional reranking step for better relevance.

        Args:
            query: Search query
            top_k: Initial retrieval count
            rerank_top_k: Final count after reranking

        Returns:
            RetrievalResult with reranked documents
        """
        # Initial retrieval
        initial_result = await self.retrieve(query=query, top_k=top_k)

        # Simple reranking by combining semantic score with keyword matching
        def calculate_rerank_score(doc: dict) -> float:
            base_score = doc["score"]
            content = doc["content"].lower()
            query_terms = query.lower().split()

            # Boost for exact term matches
            term_matches = sum(1 for term in query_terms if term in content)
            term_boost = term_matches / len(query_terms) * 0.2

            return base_score + term_boost

        # Rerank documents
        for doc in initial_result.documents:
            doc["rerank_score"] = calculate_rerank_score(doc)

        initial_result.documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        initial_result.documents = initial_result.documents[:rerank_top_k]

        # Rebuild context with reranked docs
        context_parts = []
        for i, doc in enumerate(initial_result.documents, 1):
            source = doc["metadata"].get("source", f"Document {doc['id']}")
            context_parts.append(
                f"[{i}] {doc['content']}\n"
                f"   (Score: {doc['rerank_score']:.2f}, Source: {source})"
            )

        initial_result.context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        return initial_result

    def add_travel_knowledge(
        self,
        content: str,
        destination: str | None = None,
        category: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add travel knowledge to the vector store.

        Args:
            content: The knowledge content
            destination: Related destination
            category: Knowledge category
            source: Source of the information
            metadata: Additional metadata

        Returns:
            Document ID
        """
        full_metadata = metadata or {}
        if destination:
            full_metadata["destination"] = destination.lower()
        if category:
            full_metadata["category"] = category
        if source:
            full_metadata["source"] = source

        return self.vector_store.add_document(
            content=content,
            metadata=full_metadata,
        )

    def bulk_add_knowledge(
        self,
        documents: List[dict[str, Any]],
    ) -> List[str]:
        """
        Bulk add travel knowledge documents.

        Args:
            documents: List of dicts with content, destination, category, source

        Returns:
            List of document IDs
        """
        formatted_docs = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            if "destination" in doc:
                metadata["destination"] = doc["destination"].lower()
            if "category" in doc:
                metadata["category"] = doc["category"]
            if "source" in doc:
                metadata["source"] = doc["source"]

            formatted_docs.append({
                "content": doc["content"],
                "metadata": metadata,
            })

        return self.vector_store.add_documents(formatted_docs)
