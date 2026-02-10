"""Retriever - finds relevant document chunks for a given question.

THE RETRIEVAL STEP IN RAG:
1. User asks: "What is backpropagation?"
2. The question gets embedded into a vector
3. We search the vector database for the most similar chunk vectors
4. Return the top K most relevant chunks

These chunks become the "context" that gets injected into the LLM prompt.
The LLM then answers based on YOUR documents, not just its training data.
"""

from typing import List, Tuple

from .embeddings import EmbeddingEngine


class Retriever:
    """Retrieves relevant document chunks for a given query."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        top_k: int = 4,
        score_threshold: float = 0.3,
    ):
        """Initialize the retriever.

        Args:
            embedding_engine: The embedding engine with indexed documents.
            top_k: Number of chunks to retrieve.
            score_threshold: Minimum similarity score to include a result.
        """
        self.embedding_engine = embedding_engine
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> Tuple[str, List[dict]]:
        """Retrieve relevant context for a query.

        Args:
            query: The user's question.

        Returns:
            Tuple of (formatted_context_string, raw_results_list).
        """
        results = self.embedding_engine.search(query, top_k=self.top_k)

        # Filter by score threshold
        filtered = [r for r in results if r["score"] >= self.score_threshold]

        if not filtered:
            return "", []

        # Format context for the LLM
        context_parts = []
        for i, result in enumerate(filtered, 1):
            source = result["metadata"].get("source", "unknown")
            context_parts.append(
                f"[Source: {source} | Relevance: {result['score']:.0%}]\n"
                f"{result['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)
        return context, filtered
