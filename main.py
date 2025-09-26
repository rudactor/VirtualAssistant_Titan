import os
import time
from typing import List, Tuple, Optional

import ollama
from langchain_chroma import Chroma
from transformers import logging as hf_logging

from embeddings_local_bge import LocalBGEM3Embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_logging.set_verbosity_error()

EMBEDDING_MODEL = "BAAI/bge-m3"
LANGUAGE_MODEL = "gemma3:4b"
DB_PATH = "./sql_chroma_db"

# Retrieval / filtering
FETCH_K = 30          # сколько кандидатов тянуть из вектора
TOP_N = 5             # сколько оставить после фильтра
BAD_DISTANCE = 1.3   # порог «слишком далеко», хуже этого — просим уточнить

# Текстовые пороги
PREVIEW_CHARS = 220
MAX_CONTEXT_CHARS = 6000
MIN_CHARS = 80
MIN_ALPHA_RATIO = 0.2
MAX_DIGIT_RATIO = 0.5

# --------------------
# Vector store
# --------------------
COLLECTION_NAME = "rzd_docs"   # ← добавь ту же коллекцию

embeddings = LocalBGEM3Embeddings(model_name=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,            # ← обязательно
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)
# --------------------
# Small utilities
# --------------------
def retrieve(query, top_n=3):
    results = vector_store.similarity_search_with_score(query, k=top_n)
    return results

def text_stats(text: str) -> Tuple[int, float, float]:
    """Подсчёт длины и долей букв/цифр."""
    s = text.strip()
    if not s:
        return 0, 0.0, 0.0
    total = len(s)
    alpha = sum(ch.isalpha() for ch in s)
    digit = sum(ch.isdigit() for ch in s)
    return total, alpha / total, digit / total


def is_decent_chunk(text: str) -> bool:
    """Фильтр мусорных фрагментов по простым статистикам."""
    total, alpha_ratio, digit_ratio = text_stats(text)
    if total < MIN_CHARS:
        return False
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False
    if digit_ratio > MAX_DIGIT_RATIO:
        return False
    return True

# --------------------
# Retrieval
# --------------------

def retrieve_raw(query: str, k: int = FETCH_K) -> List[Tuple[object, float]]:
    return vector_store.similarity_search_with_score(query, k=k)


def best_distance(pairs: List[Tuple[object, float]]) -> Optional[float]:
    if not pairs:
        return None
    numeric_scores = [score for _, score in pairs if isinstance(score, (int, float, float))]
    return min(numeric_scores) if numeric_scores else None


def filter_results(pairs, top_n: int = TOP_N):
    good = []
    for doc, score in pairs:
        if not isinstance(score, (int, float)):
            continue
        if not is_decent_chunk(doc.page_content):
            continue
        good.append((doc, score))
    good.sort(key=lambda x: x[1])
    return good[:top_n]


def print_neighbors(title: str, results: List[Tuple[object, float]]) -> None:
    print(f"\n--- {title} ---")
    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        dist = score
        head = f"[#{i}] dist={dist:.3f}"
        if src:
            head += f" | source={src}"
        if page is not None:
            head += f" | page={page}"
        preview = (doc.page_content or "").strip().replace("\n", " ")
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + "..."
        print(head)
        print(f"text_len={len(doc.page_content)} | {preview if preview else '<empty>'}")
        print("-" * 60)

# --------------------
# Context building
# --------------------

def build_context(results: List[Tuple[object, float]]) -> str:
    parts: List[str] = []
    total_len = 0
    for doc, _ in results:
        block = (doc.page_content or "").strip()
        if not block:
            continue
        block += "\n"
        if total_len + len(block) > MAX_CONTEXT_CHARS:
            remain = MAX_CONTEXT_CHARS - total_len
            if remain > 0:
                parts.append(block[:remain])
            break
        parts.append(block)
        total_len += len(block)
    return "\n".join(parts)


def make_system_prompt(ctx: str) -> str:
    return (
        "Ты виртуальный ассистент для инженеров и сотрудников РЖД. "
        "Отвечай по-русски, коротко и по делу, дружелюбно. "
        "Используй только данные ниже. Если данных недостаточно — скажи об этом и попроси уточнить.\n\n"
        "Данные:\n" + ctx
    )


def build_clarification_request(user_query: str) -> str:
    """Что сказать пользователю, если совпадения слишком слабые."""
    return (
        "Похоже, в базе нет точных совпадений по вашему запросу, поэтому нужен контекст.\n"
        "Пожалуйста, уточните:\n"
        "• Что именно требуется: справка, регламент, порядок действий, нормы, определение?\n"
        "• Объект/подсистема/участок (например: путь, СЦБ, Тяга, локомотив, вагон, ИТ-система и т.п.).\n"
        "• Источник (ГОСТ/СТО/ПТЭ/инструкция/распоряжение) и, если знаете, номер/раздел/пункт.\n"
        "• Любые конкретные параметры (модель, серия, диапазоны, единицы измерения).\n\n"
    )

# --------------------
# Chat
# --------------------

def chat_with_model(system_msg: str, user_msg: str) -> str:
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
        options={"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    )
    out = ""
    for chunk in stream:
        out += chunk["message"]["content"]
    return out

# --------------------
# Entry point
# --------------------

def main(input_query: str) -> str:
    t0 = time.time()

    raw = retrieve_raw(input_query, k=FETCH_K)
    used = filter_results(raw, top_n=TOP_N)

    # Логи в консоль для отладки
    print_neighbors("Nearest (raw)", raw[:TOP_N])
    print_neighbors("Used (filtered)", used)
    min_dist = best_distance(raw)
    need_more_info = (min_dist is None) or (min_dist >= BAD_DISTANCE) or (not used)

    if need_more_info:
        result = build_clarification_request(input_query)
        print("\nlatency_sec:", round(time.time() - t0, 3))
        return result

    # Нормальные соседи есть — строим контекст и зовём LLM
    context = build_context(used)
    system_msg = make_system_prompt(context)
    answer = chat_with_model(system_msg, input_query)

    print("\nlatency_sec:", round(time.time() - t0, 3))
    return answer


if __name__ == "__main__":
    user_query = input("Введите ваш вопрос: ")
    answer = main(user_query)
    print("\nОтвет модели:\n", answer)
