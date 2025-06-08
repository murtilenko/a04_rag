import logging
import json
from pathlib import Path
from typing import List
import chromadb

class EmbeddingLoader:
    def __init__(self,
                 embedding_file_list: List[str],
                 embeddings_dir: str,
                 vectordb_dir: str,
                 collection_name: str):

        self.embedding_file_list = embedding_file_list
        self.embeddings_path = Path(embeddings_dir)
        self.vectordb_path = Path(vectordb_dir)
        self.collection_name = collection_name

        self.logger = logging.getLogger(__name__)
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_or_create_collection(collection_name)

    def process_files(self):
        print(f"[DEBUG] Files to process: {len(self.embedding_file_list)}")

        for embedding_file in self.embedding_file_list:
            embedding_file_path = self.embeddings_path / embedding_file

            if not embedding_file_path.exists():
                self.logger.warning(f"Missing file: {embedding_file_path}")
                continue

            try:
                with open(embedding_file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load {embedding_file_path}: {e}")
                continue

            print(f"[DEBUG] Processing {embedding_file} with {len(chunks)} chunks")

            ids, embeddings, metadatas = [], [], []
            for i, chunk in enumerate(chunks):
                try:
                    chunk_id = f"{embedding_file}_chunk_{i:03d}"
                    ids.append(chunk_id)
                    embeddings.append(chunk["embedding"])
                    metadatas.append({
                        "text": chunk["chunk_text"],
                        "source": embedding_file
                    })
                except KeyError as e:
                    self.logger.warning(f"Missing key in {embedding_file}: {e}")
                    continue

            if ids:
                self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                print(f"[DEBUG] Added {len(ids)} vectors from {embedding_file}")
