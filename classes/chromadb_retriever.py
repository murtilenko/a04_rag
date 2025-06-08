import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
from pathlib import Path
import logging

class ChromaDBRetriever:
    """Retrieves relevant chunks from ChromaDB based on a search phrase."""

    def __init__(self, embedding_model_name: str,
                 collection_name: str,
                 vectordb_dir: str,
                 score_threshold: float = 0.5):

        self.vectordb_path = Path(vectordb_dir)
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.score_threshold = score_threshold

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized ChromaDBRetriever with model '{embedding_model_name}', "
            f"collection '{collection_name}', score threshold {score_threshold}"
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for the input text."""
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def extract_context(self, full_text: str, search_str: str) -> str:
        """
        Return the paragraph containing the search term or a fallback snippet.
        """
        paragraphs = full_text.split("\n\n")
        for para in paragraphs:
            if search_str.lower() in para.lower():
                return para.strip()
        return full_text[:300]

    def query(self, search_phrase: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query ChromaDB and return best-matching chunks (with context and metadata).
        """
        embedding_vector = self.embed_text(search_phrase)
        results = self.collection.query(
            query_embeddings=[embedding_vector],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        print("RAW CHROMA QUERY RESULTS:")
        print(results)

        retrieved_docs = []
        query_words = set(search_phrase.lower().split())

        for doc_id, metadata, document, distance in zip(
            results.get("ids", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("distances", [[]])[0]
        ):
            similarity = 1 - distance
            #if similarity < self.score_threshold:
                #continue
            print(f"SIMILARITY={similarity:.4f} for ID={doc_id}")    

            # Use metadata text if available, else fall back to Chroma's document field
            text = metadata.get("text") or document
            if not text:
                continue

            # Optional keyword match filtering (skip if none match)
            if not any(word in text.lower() for word in query_words):
                continue

            extracted_context = self.extract_context(text, search_phrase)

            retrieved_docs.append({
                "id": doc_id,
                "score": round(similarity, 4),
                "context": extracted_context,
                "source": metadata.get("source", "Unknown"),
            })

        retrieved_docs.sort(key=lambda x: x["score"], reverse=True)
        return retrieved_docs[:1] if retrieved_docs else []
