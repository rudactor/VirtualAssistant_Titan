import os
from functools import lru_cache
from typing import Dict, Any
from llama_cpp import Llama

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GGUF = os.path.join(BACKEND_DIR, "models", "qwen2.5-3b-instruct-q4_k_m.gguf")

MODEL_PATH = os.environ.get("LLM_GGUF_PATH", DEFAULT_GGUF)

def _detect_n_gpu_layers() -> int:
    if "LLM_N_GPU_LAYERS" in os.environ:
        return int(os.environ["LLM_N_GPU_LAYERS"])
    try:
        import torch
        has_cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
        has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        return 20 if (has_cuda or has_mps) else 0
    except Exception:
        return 0

def _safe_batch_default() -> int:
    if "LLM_BATCH" in os.environ:
        return int(os.environ["LLM_BATCH"])
    try:
        import torch
        has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        return 128 if has_mps else 256
    except Exception:
        return 256

@lru_cache(maxsize=1)
def _llm() -> Llama:
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"GGUF-модель не найдена: {MODEL_PATH}\n"
            "Укажи путь в переменной окружения LLM_GGUF_PATH."
        )
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=int(os.environ.get("LLM_CONTEXT", "4096")),
        n_threads=int(os.environ.get("LLM_THREADS", str(os.cpu_count() or 4))),
        n_batch=_safe_batch_default(),
        n_gpu_layers=_detect_n_gpu_layers(),
        chat_format=os.environ.get("LLM_CHAT_FORMAT", "qwen"),
        verbose=bool(int(os.environ.get("LLM_VERBOSE", "0"))),
        seed=int(os.environ.get("LLM_SEED", "-1")),
    )

# ===== токены =====
def count_tokens(text: str) -> int:
    """Подсчёт токенов тем же токенайзером llama.cpp (точно для нашего контекста)."""
    try:
        return len(_llm().tokenize(text.encode("utf-8"), add_bos=True))
    except Exception:
        # запасной грубый вариант
        return len(text) // 4 + 1

def safe_max_new_tokens(prompt_tokens: int, reserve: int = 64) -> int:
    """Гарантирует, что prompt_tokens + max_new_tokens <= n_ctx - reserve."""
    n_ctx = int(os.environ.get("LLM_CONTEXT", "4096"))
    room = max(0, n_ctx - reserve - prompt_tokens)
    cap = int(os.environ.get("LLM_MAX_NEW_TOKENS", "700"))
    return max(16, min(cap, room))

def chat_with_model(system_msg: str, user_msg: str) -> str:
    params: Dict[str, Any] = {
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        "top_p": float(os.environ.get("LLM_TOP_P", "0.9")),
        "max_tokens": int(os.environ.get("LLM_MAX_NEW_TOKENS", "700")),  # временно, ниже ужмём
        "repeat_penalty": float(os.environ.get("LLM_REPEAT_PENALTY", "1.05")),
    }
    stop_raw = os.environ.get("LLM_STOP", "").strip()
    if stop_raw:
        params["stop"] = [s for s in stop_raw.split("|||") if s]

    approx_prompt = f"[SYSTEM]\n{system_msg}\n[USER]\n{user_msg}"
    ptoks = count_tokens(approx_prompt)
    params["max_tokens"] = safe_max_new_tokens(ptoks)

    resp = _llm().create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        **params,
    )
    try:
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""
