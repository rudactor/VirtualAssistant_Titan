# -*- coding: utf-8 -*-
import os, glob
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from utils import read_file, chunk_text, make_ids

DB_PATH = "./chroma_db"
COLLECTION = "rzd_docs"
DATA_DIR = "./documentation"

class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        prep = [f"passage: {t}" for t in texts]
        return self.model.encode(prep, normalize_embeddings=True).tolist()

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        prep = [f"query: {t}" for t in texts]
        return self.model.encode(prep, normalize_embeddings=True).tolist()

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return col

def upsert_document(path: str, embedder: E5Embedder):
    text, pages = read_file(path)
    if not text:
        return

    chunks = chunk_text(text, max_chars=1200, overlap=200)
    if not chunks:
        return

    doc_name = os.path.basename(path)
    doc_id_prefix = f"{doc_name}"
    ids = make_ids(doc_id_prefix, len(chunks))
    embeddings = embedder.embed_passages(chunks)

    col = get_collection()
    existing = col.get(ids=None, where={"source": doc_name})
    if existing and existing.get("ids"):
        col.delete(where={"source": doc_name})

    col.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": doc_name} for _ in chunks]
    )
    print(f"Indexed: {doc_name} ({len(chunks)} chunks)")

if __name__ == "__main__":
    embedder = E5Embedder()
    paths = []
    paths += glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    paths += glob.glob(os.path.join(DATA_DIR, "*.docx"))
    paths += glob.glob(os.path.join(DATA_DIR, "*.txt"))

    for p in paths:
        upsert_document(p, embedder)
    print("✅ Done. Chroma persisted to", DB_PATH)
