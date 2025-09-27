# ingest.py
import os
from typing import Iterable, List, Tuple, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document

from embeddings_local_bge import LocalBGEM3Embeddings

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDING_MODEL = "BAAI/bge-m3"
DB_PATH = os.path.join(BACKEND_DIR, "sql_chroma_db")
BASE_DIR = os.path.join(BACKEND_DIR, "documentation")
COLLECTION_NAME = "rzd_docs"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
MIN_CHARS = 60
MIN_ALPHA_RATIO = 0.2
MAX_DIGIT_RATIO = 0.6
BATCH_SIZE = 128
DEFAULT_SCOPE = "global"

def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, add_start_index=True
    )

def _good(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < MIN_CHARS:
        return False
    alpha = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    total = len(s)
    if total == 0:
        return False
    if alpha / total < MIN_ALPHA_RATIO:
        return False
    if digits / total > MAX_DIGIT_RATIO:
        return False
    return True

def _doc_id(source: str, page: int, start: int, text: str, scope: str) -> str:
    import hashlib
    head = text[:80].replace("\n", " ")
    key = f"{scope}|{source}|p={page}|s={start}|n={len(text)}|h={head}"
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()

def _walk(base_dir: str) -> Iterable[Tuple[str, str]]:
    for root, _, files in os.walk(base_dir):
        for name in files:
            low = name.lower()
            if low.endswith(".pdf") or low.endswith(".docx"):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, base_dir)
                yield full, rel

def _load_pdf(path_full: str, path_rel: str, splitter) -> List[Document]:
    pages = PyPDFLoader(path_full).load_and_split()
    chunks = splitter.split_documents(pages)
    out: List[Document] = []
    for d in chunks:
        txt = (d.page_content or "").strip()
        if not _good(txt):
            continue
        meta = dict(d.metadata or {})
        page = int(meta.get("page", 0))
        start = int(meta.get("start_index", 0))
        meta.update({"source": path_rel, "page": page, "start_index": start, "n_chars": len(txt)})
        out.append(Document(page_content=txt, metadata=meta))
    return out

def _load_docx(path_full: str, path_rel: str, splitter) -> List[Document]:
    docs = Docx2txtLoader(path_full).load()
    chunks = splitter.split_documents(docs)
    out: List[Document] = []
    for d in chunks:
        txt = (d.page_content or "").strip()
        if not _good(txt):
            continue
        start = int((d.metadata or {}).get("start_index", 0))
        meta = {"source": path_rel, "page": 0, "start_index": start, "n_chars": len(txt)}
        out.append(Document(page_content=txt, metadata=meta))
    return out

def _vs() -> Chroma:
    embeddings = LocalBGEM3Embeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

def _add(vs: Chroma, docs: List[Document], scope: str):
    if not docs:
        return 0, 0
    ids = [
        _doc_id(
            d.metadata["source"],
            int(d.metadata.get("page", 0)),
            int(d.metadata.get("start_index", 0)),
            d.page_content,
            scope,
        )
        for d in docs
    ]
    for d in docs:
        d.metadata["scope"] = scope

    added, skipped = 0, 0
    for i in range(0, len(docs), BATCH_SIZE):
        bd, bi = docs[i:i+BATCH_SIZE], ids[i:i+BATCH_SIZE]
        try:
            _ = vs.add_documents(documents=bd, ids=bi)
            added += len(bd)
        except Exception:
            for doc, _id in zip(bd, bi):
                try:
                    _ = vs.add_documents(documents=[doc], ids=[_id])
                    added += 1
                except Exception:
                    skipped += 1
    try:
        vs._client.persist()
    except Exception:
        pass

    return added, skipped


def ingest(scope: Optional[str] = None):
    scope = str(scope or DEFAULT_SCOPE)
    splitter = _splitter()
    vs = _vs()

    total_in = total_add = total_dup = 0
    if not os.path.isdir(BASE_DIR):
        print(f"Папка '{BASE_DIR}' не найдена. Положи .pdf/.docx в backend/documentation/")
        print(f"Готово. Чанков: 0, добавлено: 0, дубликаты: 0, scope='{scope}'")
        return

    for full, rel in _walk(BASE_DIR):
        ext = os.path.splitext(full)[1].lower()
        chunks = _load_pdf(full, rel, splitter) if ext == ".pdf" else _load_docx(full, rel, splitter)
        if not chunks:
            print(f"— {rel}: пусто после фильтра")
            continue
        added, skipped = _add(vs, chunks, scope)
        total_in += len(chunks); total_add += added; total_dup += skipped
        print(f"✓ {rel}: подготовлено {len(chunks)}, добавлено {added}, дубликаты {skipped}")

    print(f"\nГотово. Чанков: {total_in}, добавлено: {total_add}, дубликаты: {total_dup}, scope='{scope}'")

if __name__ == "__main__":
    ingest()
