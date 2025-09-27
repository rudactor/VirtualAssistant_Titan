import os
import json
import hashlib
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

MANIFEST_PATH = os.path.join(BACKEND_DIR, ".ingest_manifest.json")
DELETE_MISSING_SOURCES = True  # если True — удаляем из БД источники, которых больше нет на диске


# ======================= Утилиты =======================
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
    head = text[:80].replace("\n", " ")
    key = f"{scope}|{source}|p={page}|s={start}|n={len(text)}|h={head}"
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()

def _walk_all(base_dir: str) -> Iterable[Tuple[str, str]]:
    for root, _, files in os.walk(base_dir):
        for name in files:
            low = name.lower()
            if low.endswith(".pdf") or low.endswith(".docx"):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, base_dir)
                yield full, rel

def _file_fingerprint(path: str) -> dict:
    """sha1 + mtime + size — надёжно определяем изменения."""
    stat = os.stat(path)
    size = stat.st_size
    mtime = int(stat.st_mtime)
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha1.update(chunk)
    return {"size": size, "mtime": mtime, "sha1": sha1.hexdigest()}

def _manifest_load() -> dict:
    if not os.path.isfile(MANIFEST_PATH):
        return {"files": {}}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": {}}

def _manifest_save(manif: dict):
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manif, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MANIFEST_PATH)


# ======================= Загрузка документов =======================
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


# ======================= Векторное хранилище =======================
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
            # поштучная попытка — на случай коллизий id
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


# ======================= Инкрементальный ingest =======================
def ingest(scope: Optional[str] = None):
    scope = str(scope or DEFAULT_SCOPE)
    splitter = _splitter()
    vs = _vs()

    manifest = _manifest_load()
    known = manifest.get("files", {})
    current_files = {}

    if not os.path.isdir(BASE_DIR):
        print(f"Папка '{BASE_DIR}' не найдена. Положи .pdf/.docx в backend/documentation/")
        print(f"Готово. Чанков: 0, добавлено: 0, дубликаты: 0, scope='{scope}'")
        return

    # Скан текущих файлов и отпечатков
    for full, rel in _walk_all(BASE_DIR):
        fp = _file_fingerprint(full)
        current_files[rel] = fp

    # Список задач
    to_add_or_update: List[Tuple[str, str, dict]] = []  # (full, rel, fingerprint)
    unchanged: List[str] = []

    for rel, fp in current_files.items():
        full = os.path.join(BASE_DIR, rel)
        old = known.get(rel)
        if old is None:
            # новый файл
            to_add_or_update.append((full, rel, fp))
        else:
            # сравнение по sha1/mtime/size — если что-то отличается, переиндексируем
            if fp.get("sha1") != old.get("sha1") or fp.get("size") != old.get("size") or fp.get("mtime") != old.get("mtime"):
                to_add_or_update.append((full, rel, fp))
            else:
                unchanged.append(rel)

    # Удаление источников, которых больше нет
    removed_sources = []
    if DELETE_MISSING_SOURCES:
        old_set = set(known.keys())
        cur_set = set(current_files.keys())
        removed_sources = sorted(list(old_set - cur_set))
        for rel in removed_sources:
            try:
                vs.delete(where={"source": rel, "scope": scope})
                print(f"✗ удалены старые чанки: {rel}")
            except Exception as e:
                print(f"! не удалось удалить {rel}: {e}")

    total_in = total_add = total_dup = 0

    # Обработка только новых/изменённых
    for full, rel, fp in to_add_or_update:
        ext = os.path.splitext(full)[1].lower()
        # если файл существовал раньше — сначала чистим его прежние чанки (для чистоты версии)
        try:
            vs.delete(where={"source": rel, "scope": scope})
        except Exception:
            pass

        if ext == ".pdf":
            chunks = _load_pdf(full, rel, splitter)
        else:
            chunks = _load_docx(full, rel, splitter)

        if not chunks:
            print(f"— {rel}: пусто после фильтра")
            # фиксируем отпечаток даже для пустых (чтоб не пытаться каждый раз)
            manifest["files"][rel] = fp
            continue

        added, skipped = _add(vs, chunks, scope)
        total_in += len(chunks)
        total_add += added
        total_dup += skipped
        manifest["files"][rel] = fp
        print(f"✓ {rel}: подготовлено {len(chunks)}, добавлено {added}, дубликаты {skipped}")

    # Отчёт по пропущенным (не изменялись)
    if unchanged:
        print(f"= пропущено без изменений: {len(unchanged)} файлов")

    _manifest_save(manifest)
    print(f"\nГотово. Чанков: {total_in}, добавлено: {total_add}, дубликаты: {total_dup}, scope='{scope}'")

if __name__ == "__main__":
    ingest()
