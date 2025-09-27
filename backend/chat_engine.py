import os, sqlite3, time, math
from typing import List, Tuple
from langchain_chroma import Chroma
from embeddings_local_bge import LocalBGEM3Embeddings
from llama_cpp_local_llm import chat_with_model, count_tokens

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# ========= Конфиг =========
DB_FILE = os.path.join(BACKEND_DIR, "chatbot.sqlite3")

# Векторное хранилище (Chroma)
EMBEDDING_MODEL = "BAAI/bge-m3"
DB_PATH = os.path.join(BACKEND_DIR, "sql_chroma_db")
COLLECTION_NAME = "rzd_docs"

# Контекст
MAX_CTX_CHARS = 3000          # остаётся как вторичная «символьная» страховка
SUMMARIZE_AFTER_CHARS = 8000  # можно снизить до 6000 при желании

# RAG
STRICT_RAG = True
RAG_K = 12
RAG_MAX_CHARS = 6000          # вторичная «символьная» отсечка; основной контроль — по токенам
BASE_BAD_DISTANCE = 0.6
ADAPT_DELTA = 0.08
SHOW_RAG_DEBUG = True

MODEL_N_CTX = int(os.environ.get("LLM_CONTEXT", "4096"))
TOKEN_MARGIN = 64  # небольшой резерв под стоп-токены и служебные

# ========= SQLite =========
def _db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _db(); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        summary TEXT DEFAULT '',
        created_at REAL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        role TEXT CHECK(role IN ('user','assistant','system')) NOT NULL,
        content TEXT NOT NULL,
        created_at REAL,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )""")
    conn.commit(); conn.close()

def start_chat(title: str = "Новый диалог") -> int:
    conn = _db(); cur = conn.cursor()
    cur.execute("INSERT INTO chats(title, created_at) VALUES (?,?)", (title, time.time()))
    conn.commit(); chat_id = cur.lastrowid; conn.close()
    return chat_id

def list_chats(limit: int = 50):
    conn = _db(); cur = conn.cursor()
    cur.execute("SELECT id, title, created_at FROM chats ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall(); conn.close()
    return [(r["id"], r["title"], r["created_at"]) for r in rows]

def chat_exists(chat_id: int) -> bool:
    conn = _db(); cur = conn.cursor()
    cur.execute("SELECT 1 FROM chats WHERE id=?", (chat_id,))
    ok = cur.fetchone() is not None
    conn.close()
    return ok

def add_message(chat_id: int, role: str, content: str):
    conn = _db(); cur = conn.cursor()
    cur.execute("INSERT INTO messages(chat_id, role, content, created_at) VALUES (?,?,?,?)",
                (chat_id, role, content, time.time()))
    conn.commit(); conn.close()

def get_chat_summary(chat_id: int) -> str:
    conn = _db(); cur = conn.cursor()
    cur.execute("SELECT summary FROM chats WHERE id=?", (chat_id,))
    row = cur.fetchone(); conn.close()
    return (row["summary"] or "") if row else ""

def set_chat_summary(chat_id: int, summary: str):
    conn = _db(); cur = conn.cursor()
    cur.execute("UPDATE chats SET summary=? WHERE id=?", (summary, chat_id))
    conn.commit(); conn.close()

def get_messages(chat_id: int) -> List[Tuple[str, str]]:
    conn = _db(); cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC", (chat_id,))
    rows = cur.fetchall(); conn.close()
    return [(r["role"], r["content"]) for r in rows]

# ========= История =========
def _total_chars(pairs: List[Tuple[str, str]]) -> int:
    return sum(len(c) for _, c in pairs)

def _last_window(pairs: List[Tuple[str, str]], budget: int) -> List[Tuple[str, str]]:
    acc: List[Tuple[str, str]] = []; used = 0
    for role, content in reversed(pairs):
        L = len(content)
        if used + L > budget:
            if not acc and budget > 200:
                acc.append((role, content[-budget:]))
            break
        acc.append((role, content)); used += L
    return list(reversed(acc))

# ========= Токенные утилиты =========
def _join_until_token_budget(chunks: list[str], budget_tokens: int) -> str:
    out, used = [], 0
    for t in chunks:
        need = count_tokens(t) + 1
        if used + need > budget_tokens:
            break
        out.append(t); used += need
    return "\n\n".join(out)

def _trim_to_tokens(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        cand = text[:mid]
        if count_tokens(cand) <= max_tokens:
            best = cand; low = mid + 1
        else:
            high = mid - 1
    return best

# ========= RAG (Chroma) =========
_embeddings = LocalBGEM3Embeddings(model_name=EMBEDDING_MODEL)
_vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
    embedding_function=_embeddings,
    collection_metadata={"hnsw:space": "cosine"},  # фиксируем метрику
)

def _to_distance(score):
    # Chroma для cosine возвращает distance (меньше — лучше). Возвращаем как есть.
    try:
        return float(score)
    except Exception:
        return None

def _retrieve(query: str, scope: str, k: int = RAG_K) -> List[str]:
    def fetch(filter_dict=None):
        try:
            return _vector_store.similarity_search_with_score(query, k=k, filter=filter_dict) \
                   if filter_dict is not None else \
                   _vector_store.similarity_search_with_score(query, k=k)
        except TypeError:
            return _vector_store.similarity_search_with_score(query, k=k)

    buckets = [("chat", {"scope": scope}), ("global", {"scope": "global"}), ("legacy", None)]
    picked = []
    debug_cache = []

    for label, flt in buckets:
        results = fetch(flt) or []
        dbg_rows = []
        for doc, score in results:
            dist = _to_distance(score)
            meta = getattr(doc, "metadata", {}) or {}
            dbg_rows.append({
                "dist": dist, "score_raw": score,
                "source": meta.get("source"), "page": meta.get("page"),
                "len": len((doc.page_content or "").strip())
            })
        debug_cache.append((label, dbg_rows))

        numeric = [_to_distance(s) for _, s in results if isinstance(_to_distance(s), (int, float))]
        numeric = [x for x in numeric if x is not None]
        if not numeric:
            continue

        best = min(numeric)
        bad_threshold = max(BASE_BAD_DISTANCE, best + ADAPT_DELTA)

        used_chars = 0
        for doc, score in sorted(results, key=lambda x: _to_distance(x[1]) if _to_distance(x[1]) is not None else 9e9):
            dist = _to_distance(score)
            if dist is None or dist > bad_threshold:
                continue
            t = (doc.page_content or "").strip()
            if not t:
                continue
            if used_chars + len(t) + 2 > RAG_MAX_CHARS:
                break
            picked.append((label, doc, dist))
            used_chars += len(t) + 2

        if picked:
            break

    if SHOW_RAG_DEBUG:
        print("\n[RAG DEBUG] запрос:", query)
        for label, rows in debug_cache:
            print(f"  [{label}] найдено: {len(rows)}")
            rows_sorted = sorted([r for r in rows if isinstance(r.get('dist'), (int, float))],
                                 key=lambda r: r["dist"])[:5]
            for i, r in enumerate(rows_sorted, 1):
                print(f"    #{i} dist={r['dist']:.4f} | raw={r['score_raw']} | len={r['len']} | source={r['source']} | page={r['page']}")
        if picked:
            print("[RAG DEBUG] ОТБРАНО:")
            for i, (label, doc, dist) in enumerate(picked, 1):
                src = doc.metadata.get("source"); page = doc.metadata.get("page")
                print(f"    #{i} [{label}] dist={dist:.4f} | source={src} | page={page} | len={len((doc.page_content or '').strip())}")
        else:
            print("[RAG DEBUG] ничего не отобрано по адаптивному порогу")

    return [doc.page_content.strip() for _, doc, _ in picked]

# ========= Сжатие истории =========
def _summarize(chat_id: int):
    msgs = get_messages(chat_id)
    if _total_chars(msgs) <= SUMMARIZE_AFTER_CHARS:
        return
    current = get_chat_summary(chat_id)
    sys = ("Ты помощник. Суммируй диалог кратко, по пунктам. "
           "Сохрани намерения пользователя, факты, решения и договорённости. Не выдумывай.")
    buf = []
    if current:
        buf.append(f"[Текущее резюме]\n{current}\n")
    for role, content in msgs:
        tag = "Пользователь" if role == "user" else "Ассистент" if role == "assistant" else "Система"
        buf.append(f"{tag}: {content}")
    new_summary = chat_with_model(sys, "\n".join(buf)).strip()
    if new_summary:
        set_chat_summary(chat_id, new_summary)

# ========= Промпты =========
def _clarify(user_input: str) -> str:
    return (
        "Недостаточно данных в базе для точного ответа.\n"
        "Уточните, пожалуйста:\n"
        "• Что именно нужно (регламент, пошаговая процедура, нормы, ссылки на пункты)?\n"
        "• Объект/подсистема (напр. вагон, тележка, узел, серия/модель).\n"
        "• Источник (ПТЭ/ГОСТ/СТО/инструкция) и, если знаете, номер/раздел/пункт.\n"
        f"Запрос: «{user_input}»"
    )

def _system_prompt(summary: str, rag_block: str) -> str:
    parts = [
        "Ты виртуальный ассистент для инженеров и сотрудников РЖД.",
        "Отвечай только по предоставленным ниже материалам.",
        "Не ссылайся на источники, названия разделов или внутренние механизмы поиска.",
        "Если данных недостаточно — вежливо попроси уточнить, без упоминания того, как ты ищешь информацию.",
    ]
    if summary:
        parts.append("\n[Краткое резюме диалога]\n" + summary)
    parts.append("\n[Материалы]\n" + (rag_block if rag_block else "(пусто)"))
    return "\n".join(parts)

# ========= Диагностика =========
def debug_find(query: str, scope: str, k: int = RAG_K) -> str:
    def fetch(filter_dict=None):
        try:
            return _vector_store.similarity_search_with_score(query, k=k, filter=filter_dict) \
                   if filter_dict is not None else \
                   _vector_store.similarity_search_with_score(query, k=k)
        except TypeError:
            return _vector_store.similarity_search_with_score(query, k=k)

    lines = [f"[DEBUG FIND] query={query!r} scope={scope} k={k}"]
    for label, flt in [("chat", {"scope": scope}), ("global", {"scope": "global"}), ("legacy", None)]:
        res = fetch(flt) or []
        lines.append(f"\n[{label}] total={len(res)}")
        for i, (doc, score) in enumerate(res, 1):
            meta = getattr(doc, "metadata", {}) or {}
            preview = (doc.page_content or "").strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            lines.append(f"  #{i} score={score} | source={meta.get('source')} | page={meta.get('page')}")
            lines.append(f"      {preview}")
    return "\n".join(lines)

# ========= Основной вызов =========
def ask(chat_id: int, user_input: str) -> str:
    add_message(chat_id, "user", user_input)

    _summarize(chat_id)
    summary = get_chat_summary(chat_id)

    scope = str(chat_id)
    rag_texts = _retrieve(user_input, scope, k=RAG_K)

    if STRICT_RAG and not rag_texts:
        msg = _clarify(user_input)
        add_message(chat_id, "assistant", msg)
        return msg

    # История (символьное окно — как раньше, чтобы не раздувать; при сборке ниже ещё раз проверим токенами)
    msgs = _last_window(get_messages(chat_id), MAX_CTX_CHARS)
    history_text = ""
    for role, content in msgs:
        if role == "system":
            continue
        who = "Пользователь" if role == "user" else "Ассистент"
        history_text += f"{who}: {content}\n"

    instruction = (
        "Инструкция: используй только предоставленные материалы. "
        "Если нужного факта нет — вежливо попроси уточнить. "
        "Не упоминай материалы, источники или механики поиска."
    )

    prompt_header = history_text + "\n[Текущий вопрос]\n" + user_input + "\n\n" + instruction
    system_preview = _system_prompt(summary, "(материалы будут добавлены ниже)")

    # ===== токенный бюджет на промпт =====
    n_ctx = MODEL_N_CTX
    reserve = TOKEN_MARGIN
    base_budget = n_ctx - reserve

    base_used = count_tokens(system_preview) + count_tokens(prompt_header)
    rag_budget = max(0, base_budget - base_used)

    # Склеиваем RAG с учётом токенного бюджета
    rag_block = _join_until_token_budget(rag_texts, rag_budget)

    # Финальные system и prompt
    system_msg = _system_prompt(summary, rag_block)
    prompt = prompt_header

    def total_prompt_tokens(sys_msg: str, usr_msg: str) -> int:
        return count_tokens(f"[SYSTEM]\n{sys_msg}\n[USER]\n{usr_msg}")

    ptoks = total_prompt_tokens(system_msg, prompt)

    # Если не влезает — подрезаем по приоритету: RAG -> история -> инструкция
    if ptoks > n_ctx - reserve:
        need = (ptoks - (n_ctx - reserve)) + 32
        target = max(0, count_tokens(rag_block) - need)
        rag_block = _trim_to_tokens(rag_block, target)
        system_msg = _system_prompt(summary, rag_block)
        ptoks = total_prompt_tokens(system_msg, prompt)

    if ptoks > n_ctx - reserve:
        lines = [ln for ln in history_text.splitlines() if ln.strip()]
        last_turn = "\n".join(lines[-6:]) if lines else ""
        history_text = last_turn + ("\n" if last_turn else "")
        prompt_header = history_text + "\n[Текущий вопрос]\n" + user_input + "\n\n" + instruction
        prompt = prompt_header
        ptoks = total_prompt_tokens(system_msg, prompt)

    if ptoks > n_ctx - reserve:
        short_instr = "Инструкция: отвечай только по материалам. Если факта нет — попроси уточнить."
        prompt_header = history_text + "\n[Текущий вопрос]\n" + user_input + "\n\n" + short_instr
        prompt = prompt_header
        ptoks = total_prompt_tokens(system_msg, prompt)

    answer = chat_with_model(system_msg, prompt)
    add_message(chat_id, "assistant", answer)
    return answer
