# RAG Chatbot

A local AI chatbot that answers questions about your documents using Retrieval-Augmented Generation (RAG). Everything runs on your machine — no data leaves your PC.

## How It Works

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
│  Upload Docs │ →  │ Chunk & Embed │ →  │ Store Vectors│ →  │   ChromaDB   │
│  (PDF, TXT)  │    │ (split text)  │    │ (numerical)  │    │ (vector DB)  │
└──────────────┘    └───────────────┘    └──────────────┘    └──────┬───────┘
                                                                    │
┌──────────────┐    ┌───────────────┐    ┌──────────────┐          │
│  Ask Question│ →  │ Find Relevant │ →  │ Build Prompt │ ←────────┘
│  (chat UI)   │    │   Chunks      │    │ with Context │
└──────────────┘    └───────────────┘    └──────┬───────┘
                                                │
                    ┌───────────────┐    ┌───────▼──────┐
                    │   Response    │ ←  │ Ollama LLM   │
                    │ with Sources  │    │ (local GPU)  │
                    └───────────────┘    └──────────────┘
```

## Features

- **100% Local** — Runs on your machine using Ollama. No API keys, no cloud, no data leaks
- **Multi-format** — Supports PDF, TXT, Markdown, and DOCX files
- **GPU Accelerated** — Embeddings and LLM inference use your GPU
- **Web UI** — Clean Gradio chat interface with file upload
- **Streaming** — Responses stream token-by-token for instant feedback
- **Source Citations** — Every answer shows which documents it used
- **Conversation Memory** — Maintains chat context across messages

## Prerequisites

- **Ollama** — Install from [ollama.com](https://ollama.com)
- **Python 3.10+**
- **NVIDIA GPU** recommended (works on CPU too, just slower)

## Setup

```bash
# Clone the repo
git clone git@github.com:H4ph4z4rdz/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model (one-time, ~2GB download)
ollama pull llama3.2
```

## Usage

### 1. Add Your Documents

Drop PDF, TXT, MD, or DOCX files into the `documents/` folder, or upload them through the web UI.

### 2. Launch the Chatbot

```bash
python src/app.py
```

Open **http://localhost:7861** in your browser.

### 3. Ask Questions

Upload documents through the UI or place them in the `documents/` folder, then ask anything about their content!

## Project Structure

```
rag-chatbot/
├── configs/
│   └── default.yaml          # All configuration (model, chunking, retrieval)
├── documents/                 # Drop your documents here
├── vectorstore/               # Auto-generated embeddings (ChromaDB)
└── src/
    ├── app.py                 # Gradio web UI + main entry point
    └── core/
        ├── document_loader.py # Load & chunk PDFs, TXT, DOCX
        ├── embeddings.py      # Text → vectors (sentence-transformers)
        ├── retriever.py       # Semantic search over vectors
        └── chat.py            # LLM orchestration (Ollama)
```

## Configuration

Edit `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm.model` | llama3.2 | Ollama model (try: mistral, codellama, phi3) |
| `llm.temperature` | 0.7 | Creativity (0=factual, 1=creative) |
| `documents.chunk_size` | 1000 | Characters per chunk |
| `documents.chunk_overlap` | 200 | Overlap between chunks |
| `retrieval.top_k` | 4 | Number of chunks to retrieve |
| `embeddings.model` | all-MiniLM-L6-v2 | Sentence-transformer model |

## Tech Stack

- **Ollama** — Local LLM inference
- **LangChain** — LLM orchestration framework
- **ChromaDB** — Vector database for semantic search
- **Sentence-Transformers** — Text embedding models
- **Gradio** — Web UI framework
- **PyPDF / python-docx** — Document parsing

## License

MIT
