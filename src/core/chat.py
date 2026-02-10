"""Chat engine - orchestrates the RAG pipeline.

THE FULL RAG FLOW:
1. User sends a message
2. Retriever finds relevant document chunks
3. Chunks are injected into a prompt template as "context"
4. The LLM (via Ollama) generates a response grounded in your documents
5. Response is returned with source citations

This is how enterprise AI assistants work - same pattern, just scaled up.
"""

from typing import List, Tuple, Generator

from langchain_ollama import OllamaLLM

from .retriever import Retriever


# System prompt that tells the LLM how to behave
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context from the user's documents.

RULES:
- Answer the question using ONLY the information from the context below.
- If the context doesn't contain enough information, say so honestly.
- Cite which source document(s) you used in your answer.
- Be concise but thorough.
- If the user's question is a greeting or general chat, respond naturally.

CONTEXT FROM DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

ANSWER:"""

NO_CONTEXT_PROMPT = """You are a helpful AI assistant. The user has asked a question but no relevant documents were found in the knowledge base.

Let the user know that you couldn't find relevant information in their documents, and suggest they:
1. Upload relevant documents to the 'documents' folder
2. Re-index the documents
3. Try rephrasing their question

If the user is just chatting or greeting you, respond naturally.

USER: {question}

ANSWER:"""


class ChatEngine:
    """Orchestrates the full RAG chat pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        num_ctx: int = 4096,
    ):
        """Initialize the chat engine.

        Args:
            retriever: Retriever instance for finding relevant chunks.
            model_name: Ollama model name.
            temperature: LLM temperature (creativity).
            num_ctx: Context window size.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.conversation_history: List[Tuple[str, str]] = []

        print(f"Connecting to Ollama model: {model_name}...")
        self.llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        print(f"  LLM ready: {model_name}")

    def chat(self, question: str) -> Tuple[str, List[dict]]:
        """Process a user question through the RAG pipeline.

        Args:
            question: The user's question.

        Returns:
            Tuple of (answer_string, source_documents_list).
        """
        # Step 1: Retrieve relevant context
        context, sources = self.retriever.retrieve(question)

        # Step 2: Build the prompt
        history_str = self._format_history()

        if context:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                history=history_str,
                question=question,
            )
        else:
            prompt = NO_CONTEXT_PROMPT.format(question=question)

        # Step 3: Generate response from LLM
        response = self.llm.invoke(prompt)

        # Step 4: Update conversation history
        self.conversation_history.append((question, response))

        # Keep last 10 exchanges to avoid context overflow
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return response, sources

    def stream_chat(self, question: str) -> Generator[Tuple[str, List[dict]], None, None]:
        """Stream a response token by token.

        Args:
            question: The user's question.

        Yields:
            Tuples of (partial_answer, source_documents).
        """
        # Retrieve context
        context, sources = self.retriever.retrieve(question)
        history_str = self._format_history()

        if context:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                history=history_str,
                question=question,
            )
        else:
            prompt = NO_CONTEXT_PROMPT.format(question=question)

        # Stream response
        full_response = ""
        for chunk in self.llm.stream(prompt):
            full_response += chunk
            yield full_response, sources

        # Update history after complete
        self.conversation_history.append((question, full_response))
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def _format_history(self) -> str:
        """Format conversation history for the prompt."""
        if not self.conversation_history:
            return "No previous conversation."

        lines = []
        for q, a in self.conversation_history[-5:]:  # Last 5 exchanges
            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")
        return "\n".join(lines)
