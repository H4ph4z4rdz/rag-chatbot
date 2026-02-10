"""Document loading and chunking.

This module handles:
1. Loading documents from various formats (PDF, TXT, DOCX)
2. Splitting them into overlapping chunks for embedding

WHY CHUNKING?
LLMs have limited context windows. Instead of feeding an entire 100-page PDF,
we split it into small, overlapping pieces. When a user asks a question, we
only retrieve the most relevant chunks - this is the "Retrieval" in RAG.

WHY OVERLAP?
If a sentence spans two chunks, overlap ensures it appears in both.
Without overlap, you'd lose context at chunk boundaries.
"""

import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentLoader:
    """Loads and chunks documents from a directory."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document loader.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of characters that overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file and return its content as Documents.

        Args:
            file_path: Path to the file.

        Returns:
            List of Document objects with content and metadata.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".txt" or ext == ".md":
            return self._load_text(file_path)
        elif ext == ".docx":
            return self._load_docx(file_path)
        else:
            print(f"  Skipping unsupported file: {file_path}")
            return []

    def load_directory(self, directory: str) -> List[Document]:
        """Load all supported files from a directory.

        Args:
            directory: Path to the directory containing documents.

        Returns:
            List of chunked Document objects.
        """
        all_documents = []
        supported_extensions = {".pdf", ".txt", ".md", ".docx"}

        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []

        files = [
            f for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]

        if not files:
            print(f"No supported documents found in {directory}")
            return []

        print(f"Loading {len(files)} document(s)...")

        for filename in sorted(files):
            file_path = os.path.join(directory, filename)
            docs = self.load_file(file_path)
            all_documents.extend(docs)
            print(f"  Loaded: {filename} ({len(docs)} chunks)")

        print(f"Total: {len(all_documents)} chunks from {len(files)} files")
        return all_documents

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load and chunk a PDF file."""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

        if not text.strip():
            print(f"  Warning: No text extracted from {file_path}")
            return []

        documents = self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": os.path.basename(file_path), "type": "pdf"}],
        )
        return documents

    def _load_text(self, file_path: str) -> List[Document]:
        """Load and chunk a text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if not text.strip():
            return []

        documents = self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": os.path.basename(file_path), "type": "text"}],
        )
        return documents

    def _load_docx(self, file_path: str) -> List[Document]:
        """Load and chunk a DOCX file."""
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)
        text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        if not text.strip():
            return []

        documents = self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": os.path.basename(file_path), "type": "docx"}],
        )
        return documents
