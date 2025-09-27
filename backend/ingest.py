import os
import hashlib
from typing import Iterable, List, Tuple

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from embeddings_local_bge import LocalBGEM3Embeddings

EMBEDDING_MODEL = "BAAI/bge-m3"
DB_PATH = "./sql_chroma_db"
BASE_DIR = "documentation"
COLLECTION_NAME = "rzd_docs"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
MIN_CHARS = 60
MIN_ALPHA_RATIO = 0.2
MAX_DIGIT_RATIO = 0.6
BATCH_SIZE = 128


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )


def is_good(text: str) -> bool:
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


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def doc_id(source: str, page: int, start: int, text: str) -> str:
    head = text[:80].replace("\n", " ")
    key = f"{source}|p={page}|s={start}|n={len(text)}|h={head}"
    return sha1(key)


def iter_files(base_dir: str) -> Iterable[Tuple[str, str]]:
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name.lower().endswith(".pdf") or name.lower().endswith(".docx"):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, base_dir)
                yield full, rel


def load_pdf(path_full: str, path_rel: str, splitter) -> List[Document]:
    loader = PyPDFLoader(path_full)
    pages = loader.load_and_split()  # по страницам
    chunks = splitter.split_documents(pages)
    out: List[Document] = []
    for d in chunks:
        txt = (d.page_content or "").strip()
        if not is_good(txt):
            continue
        meta = dict(d.metadata or {})
        page = int(meta.get("page", 0))
        start = int(meta.get("start_index", 0))
        meta.update({
            "source": path_rel,
            "page": page,
            "start_index": start,
            "n_chars": len(txt),
        })
        out.append(Document(page_content=txt, metadata=meta))
    return out


def load_docx(path_full: str, path_rel: str, splitter) -> List[Document]:
    loader = Docx2txtLoader(path_full)
    docs = loader.load()  # обычно один элемент
    chunks = splitter.split_documents(docs)
    out: List[Document] = []
    for d in chunks:
        txt = (d.page_content or "").strip()
        if not is_good(txt):
            continue
        start = int((d.metadata or {}).get("start_index", 0))
        meta = {
            "source": path_rel,
            "page": 0,
            "start_index": start,
            "n_chars": len(txt),
        }
        out.append(Document(page_content=txt, metadata=meta))
    return out


def get_vector_store() -> Chroma:
    embeddings = LocalBGEM3Embeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,                # уже есть
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},   # ← добавь ЯВНО
    )


def add_chunks(vs: Chroma, docs: List[Document]) -> Tuple[int, int]:
    if not docs:
        return 0, 0
    ids = [doc_id(d.metadata["source"], d.metadata["page"], d.metadata["start_index"], d.page_content) for d in docs]

    added, skipped = 0, 0
    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        try:
            vs.add_documents(documents=batch_docs, ids=batch_ids)
            added += len(batch_docs)
        except Exception as e:
            # чаще всего «ID already exists» — будем пытаться по одному, чтобы посчитать дубликаты
            for d, _id in zip(batch_docs, batch_ids):
                try:
                    vs.add_documents(documents=[d], ids=[_id])
                    added += 1
                except Exception:
                    skipped += 1
    vs.persist()
    return added, skipped


def ingest():
    splitter = make_splitter()
    vs = get_vector_store()

    total_added = total_skipped = 0
    total_docs = 0

    for full, rel in iter_files(BASE_DIR):
        ext = os.path.splitext(full)[1].lower()
        if ext == ".pdf":
            chunks = load_pdf(full, rel, splitter)
        elif ext == ".docx":
            chunks = load_docx(full, rel, splitter)
        else:
            continue

        if not chunks:
            print(f"— {rel}: пусто после фильтра")
            continue

        added, skipped = add_chunks(vs, chunks)
        total_added += added
        total_skipped += skipped
        total_docs += len(chunks)

        print(f"✓ {rel}: подготовлено {len(chunks)}, добавлено {added}, дубликаты {skipped}")

    print(f"\nГотово. Всего чанков подготовлено: {total_docs}, добавлено: {total_added}, дубликаты: {total_skipped}")


if __name__ == "__main__":
    ingest()
