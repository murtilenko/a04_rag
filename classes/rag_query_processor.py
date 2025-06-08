from .llm_client import LLMClient
from .chromadb_retriever import ChromaDBRetriever
import logging

class RAGQueryProcessor:

    def __init__(self,
                 llm_client: LLMClient,
                 retriever: ChromaDBRetriever,
                 use_rag: bool = False):
        self.use_rag = use_rag
        self.llm_client = llm_client
        self.retriever = retriever if use_rag else None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RAGQueryProcessor: use_rag: {use_rag}")

    def query(self, query_text: str):
        """
        Processes the query with optional RAG.
        """
        self.logger.debug(f"Received query: {query_text}")

        context = ""
        if self.use_rag:
            self.logger.info("-"*80)
            self.logger.info("Using RAG pipeline...")
            retrieved_chunks = self.retriever.query(query_text)

            if not retrieved_chunks:
                self.logger.info("*** No relevant context found.")
                context = "No relevant context found."
            else:
                # Combine top-N contexts (can change number in retriever)
                context = "\n\n---\n\n".join(
                    f"[{chunk['source']}] {chunk['context']}" for chunk in retrieved_chunks
                )

                for i, chunk in enumerate(retrieved_chunks):
                    self.logger.info(f"[{i+1}] ID: {chunk.get('id', 'N/A')} | Score: {chunk.get('score', 'N/A')} | Source: {chunk.get('source', 'Unknown')}")
                    self.logger.debug(f"Context chunk: {chunk['context'][:200]}...")

            self.logger.info("-" * 80)

        # Construct structured prompt
        final_prompt = f"""You are an AI assistant answering user queries using retrieved context.
If the context is insufficient, say 'I don't know'. 

Context:
{context}

Question:
{query_text}
"""

        self.logger.debug(f"Prompt to LLM:\n{final_prompt}")

        response = self.llm_client.query(final_prompt)
        self.logger.debug(f"LLM Response: {response}")

        return response.strip()
