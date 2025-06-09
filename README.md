# a04_rag
#  A4-RAG-PROJECT

A Retrieval-Augmented Generation (RAG) system designed to answer queries using custom documents with embeddings stored in ChromaDB and LLM-based response generation.

---

##  Features

- Document ingestion and chunking
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- Persistent ChromaDB vector store
- Contextual retrieval of top-k chunks
- LLM-based answer generation using LLaMA2 via Ollama (local HTTP API)
- Modular pipeline with step-based execution (`step01`, `step02`, etc.)
