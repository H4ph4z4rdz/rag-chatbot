"""Embedding engine for converting text to vectors.

WHAT ARE EMBEDDINGS?
Embeddings convert text into numerical vectors (lists of numbers).
Similar texts produce similar vectors. This lets us do "semantic search" -
finding text that means the same thing, not just matches keywords.

Example:
  "The cat sat on the mat"  → [0.12, -0.34, 0.56, ...]
  "A kitten rested on a rug" → [0.11, -0.33, 0.55, ...]  (very similar!)
  "Stock prices rose today"  → [0.87, 0.22, -0.91, ...]  (very different!)

HOW IT WORKS IN RAG:
1. Document chunks → embedded → stored in vector database
2. User question → embedded → compared against all stored vectors
3. Most similar chunks retrieved → fed to LLM as context
"""

import os
from typing import List

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Handles text embedding and vector storage using ChromaDB."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "vectorstore",
        collection_name: str = "documents",
        device: str = "cuda",
    ):
        """Initialize the embedding engine.

        Args:
            model_name: Sentence-transformer model for embeddings.
            persist_directory: Directory to persist the vector store.
            collection_name: Name of the ChromaDB collection.
            device: Device for the embedding model ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Load the embedding model
        print(f"Loading embedding model: {model_name} (on {device})...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.embedding_dim}")

        # Initialize ChromaDB
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def embed_documents(self, documents: List[Document]) -> int:
        """Embed and store documents in the vector database.

        Args:
            documents: List of Document objects to embed.

        Returns:
            Number of documents embedded.
        """
        if not documents:
            print("No documents to embed.")
            return 0

        print(f"Embedding {len(documents)} chunks...")

        # Extract text and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
        ).tolist()

        # Clear existing data and add new
        # (simple approach - in production you'd do incremental updates)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"Stored {len(documents)} chunks in vector database.")
        return len(documents)

    def search(self, query: str, top_k: int = 4) -> List[dict]:
        """Search for relevant document chunks.

        Args:
            query: The search query (user's question).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'text', 'metadata', and 'score' keys.
        """
        # Embed the query
        query_embedding = self.model.encode([query]).tolist()

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                })

        return formatted

    @property
    def document_count(self) -> int:
        """Get the number of documents in the vector store."""
        return self.collection.count()
