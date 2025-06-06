import logging
from pathlib import Path
import chromadb
import json
from typing import List
from chromadb.utils import embedding_functions


class EmbeddingLoader:
    def __init__(self,
                 cleaned_text_file_list: List[str],
                 cleaned_text_dir: str,
                 embeddings_dir: str,
                 vectordb_dir: str,
                 collection_name: str,
                 batch_size: int = 16):

        self.cleaned_text_file_list = cleaned_text_file_list
        self.cleaned_text_path = Path(cleaned_text_dir)
        self.embeddings_path = Path(embeddings_dir)
        self.vectordb_path = Path(vectordb_dir)
        self.collection_name = collection_name
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_or_create_collection(collection_name)

    def _load_chunked_embeddings(self, file_path: Path) -> List[dict]:
        """Loads chunk-level embeddings from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                embeddings = json.load(f)
                if isinstance(embeddings, list) and all(
                    isinstance(e, dict) and "embedding" in e and "chunk_text" in e for e in embeddings
                ):
                    return embeddings
                else:
                    raise ValueError("Invalid chunked embedding format.")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing embeddings file {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading embeddings file {file_path}: {e}")
        return []

    def process_files(self):
        total_files = 0
        total_chunks = 0
        skipped_files = 0

        for cleaned_text_file in self.cleaned_text_file_list:
            embedding_file_path = self.embeddings_path / f"{Path(cleaned_text_file).stem}_embeddings.json"

            if not embedding_file_path.exists():
                self.logger.warning(f"Missing embedding file for {cleaned_text_file}, skipping.")
                skipped_files += 1
                continue

            chunks = self._load_chunked_embeddings(embedding_file_path)
            if not chunks:
                self.logger.warning(f"No valid chunks found in {embedding_file_path.name}, skipping.")
                skipped_files += 1
                continue

            ids, embeddings, metadatas = [], [], []

            for i, chunk in enumerate(chunks):
                try:
                    chunk_id = f"{Path(cleaned_text_file).stem}_chunk_{i:03d}"
                    chunk_text = chunk["chunk_text"]
                    chunk_embedding = chunk["embedding"]

                    ids.append(chunk_id)
                    embeddings.append(chunk_embedding)
                    metadatas.append({
                        "text": chunk_text,
                        "source": cleaned_text_file
                    })
                except KeyError as e:
                    self.logger.warning(f"Chunk missing key {e} in file {cleaned_text_file}")
                    continue

            if len(ids) == len(embeddings) == len(metadatas) and ids:
                try:
                    self.logger.debug(f"Adding {len(ids)} vectors from {embedding_file_path.name}")
                    self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                    self.logger.info(f"Stored {len(ids)} chunks from {cleaned_text_file}.")
                    total_files += 1
                    total_chunks += len(ids)
                except Exception as e:
                    self.logger.error(f"Failed to store vectors from {embedding_file_path.name}: {e}")
            else:
                self.logger.warning(f"Inconsistent data lengths for {cleaned_text_file}. Skipping.")

        self.logger.info(f"Embedding storage completed. {total_files} files processed, {total_chunks} chunks stored, {skipped_files} files skipped.")
