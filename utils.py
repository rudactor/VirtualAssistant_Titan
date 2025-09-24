import os, re, uuid
from typing import List, Dict, Tuple
from pypdf import PdfReader
import docx2txt

def read_file(path: str) -> Tuple[str, List[Dict]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    elif ext in [".docx"]:
        return _read_docx(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text, [{"page": 1, "text": text}]

def _read_pdf(path: str) -> Tuple[str, List[Dict]]:
    reader = PdfReader(path)
    pages = []
    all_text = []
    for i, p in enumerate(reader.pages):
        t = p.extract_text() or ""
        t = _clean(t)
        pages.append({"page": i+1, "text": t})
        all_text.append(t)
    return "\n".join(all_text), pages

def _read_docx(path: str) -> Tuple[str, List[Dict]]:
    t = docx2txt.process(path) or ""
    t = _clean(t)
    return t, [{"page": 1, "text": t}]

def _clean(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + "\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            if overlap > 0 and len(cur) > 0:
                tail = cur[-overlap:]
                cur = (tail + "\n" + p).strip()
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks

def make_ids(prefix: str, n: int) -> List[str]:
    return [f"{prefix}::{i}" for i in range(n)]
