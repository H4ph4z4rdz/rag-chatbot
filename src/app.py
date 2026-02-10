"""RAG Chatbot - Gradio Web UI.

Launch with:
    python src/app.py

Then open http://localhost:7861 in your browser.
Upload documents and chat with your AI assistant!
"""

import os
import sys

import yaml
import gradio as gr

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from core import DocumentLoader, EmbeddingEngine, Retriever, ChatEngine


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def initialize_pipeline(config: dict):
    """Initialize the full RAG pipeline."""
    # Document loader
    doc_loader = DocumentLoader(
        chunk_size=config["documents"]["chunk_size"],
        chunk_overlap=config["documents"]["chunk_overlap"],
    )

    # Embedding engine
    embedding_engine = EmbeddingEngine(
        model_name=config["embeddings"]["model"],
        persist_directory=config["vectorstore"]["directory"],
        collection_name=config["vectorstore"]["collection"],
        device=config["embeddings"]["device"],
    )

    # Load and index documents if they exist
    doc_dir = config["documents"]["directory"]
    if os.path.exists(doc_dir) and os.listdir(doc_dir):
        documents = doc_loader.load_directory(doc_dir)
        if documents:
            embedding_engine.embed_documents(documents)

    # Retriever
    retriever = Retriever(
        embedding_engine=embedding_engine,
        top_k=config["retrieval"]["top_k"],
        score_threshold=config["retrieval"]["score_threshold"],
    )

    # Chat engine
    chat_engine = ChatEngine(
        retriever=retriever,
        model_name=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        num_ctx=config["llm"]["num_ctx"],
    )

    return doc_loader, embedding_engine, retriever, chat_engine


# Load config and initialize
config = load_config()
doc_loader, embedding_engine, retriever, chat_engine = initialize_pipeline(config)


def respond(message: str, chat_history: list):
    """Handle a chat message with streaming."""
    chat_history = chat_history or []

    # Add user message
    chat_history.append({"role": "user", "content": message})
    # Add empty assistant message to stream into
    chat_history.append({"role": "assistant", "content": ""})

    # Stream the response
    for partial_response, sources in chat_engine.stream_chat(message):
        chat_history[-1]["content"] = partial_response
        yield "", chat_history

    # Append source info
    if sources:
        source_names = set(s["metadata"].get("source", "?") for s in sources)
        source_text = f"\n\n📚 *Sources: {', '.join(source_names)}*"
        chat_history[-1]["content"] += source_text
        yield "", chat_history


def upload_and_index(files):
    """Handle file upload and re-index."""
    if not files:
        return "No files uploaded."

    doc_dir = config["documents"]["directory"]
    os.makedirs(doc_dir, exist_ok=True)

    uploaded = []
    for file in files:
        filename = os.path.basename(file.name)
        dest = os.path.join(doc_dir, filename)

        # Copy file to documents directory
        with open(file.name, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        uploaded.append(filename)

    # Re-index all documents
    documents = doc_loader.load_directory(doc_dir)
    count = embedding_engine.embed_documents(documents)

    chat_engine.clear_history()

    return (
        f"✅ Uploaded {len(uploaded)} file(s): {', '.join(uploaded)}\n"
        f"📊 Indexed {count} chunks total\n"
        f"💬 Chat history cleared. You can now ask questions!"
    )


def clear_chat():
    """Clear the chat history."""
    chat_engine.clear_history()
    return [], ""


def get_status():
    """Get current system status."""
    doc_dir = config["documents"]["directory"]
    doc_count = len([
        f for f in os.listdir(doc_dir)
        if not f.startswith(".")
    ]) if os.path.exists(doc_dir) else 0

    chunk_count = embedding_engine.document_count

    return (
        f"📄 Documents: {doc_count}\n"
        f"🧩 Chunks indexed: {chunk_count}\n"
        f"🤖 LLM: {config['llm']['model']}\n"
        f"🧠 Embeddings: {config['embeddings']['model']}\n"
        f"💾 Vector store: {config['vectorstore']['directory']}"
    )


# Build the Gradio UI
with gr.Blocks(
    title="RAG Chatbot",
) as demo:

    gr.Markdown(
        """
        # 🤖 RAG Chatbot
        ### Chat with your documents using local AI (Ollama + LLama)

        Upload PDFs, text files, or Word docs — then ask questions about them!
        Everything runs locally on your machine. No data leaves your PC.
        """
    )

    with gr.Row():
        # Left sidebar
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Document Management")

            file_upload = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md", ".docx"],
                height=120,
            )
            upload_btn = gr.Button("📥 Upload & Index", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=4,
            )

            gr.Markdown("### ℹ️ System Info")
            status_display = gr.Textbox(
                label="Status",
                value=get_status(),
                interactive=False,
                lines=5,
            )
            refresh_btn = gr.Button("🔄 Refresh Status")

        # Main chat area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about your documents...",
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Chat")

    gr.Markdown(
        """
        ---
        **Tips:**
        - Upload documents first, then ask questions about them
        - The AI only answers based on your uploaded documents
        - Supports PDF, TXT, Markdown, and DOCX files
        - All processing happens locally — your data never leaves your machine

        *Built with LangChain, ChromaDB, Ollama & Gradio*
        """
    )

    # Wire up events
    upload_btn.click(upload_and_index, [file_upload], [upload_status]).then(
        get_status, [], [status_display]
    )

    msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
    send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])
    clear_btn.click(clear_chat, [], [chatbot, msg_input])
    refresh_btn.click(get_status, [], [status_display])


if __name__ == "__main__":
    print("\n🚀 Launching RAG Chatbot...")
    print(f"   Model: {config['llm']['model']}")
    print(f"   Open http://localhost:{config['ui']['port']} in your browser\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=config["ui"]["port"],
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo"),
    )
