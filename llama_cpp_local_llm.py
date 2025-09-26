from llama_cpp import Llama
import os

MODEL_PATH = os.environ.get("LLM_GGUF_PATH", "./models/qwen2.5-3b-instruct-q4_k_m.gguf")

def detect_n_gpu_layers() -> int:
    if "LLM_N_GPU_LAYERS" in os.environ:
        return int(os.environ["LLM_N_GPU_LAYERS"])
    try:
        import torch
        if torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return 32   # GPU берёт часть нагрузки
    except ImportError:
        pass
    return 0  # fallback: CPU-only

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=int(os.environ.get("LLM_CONTEXT", "4096")),
    n_threads=int(os.environ.get("LLM_THREADS", str(os.cpu_count() or 4))),
    n_batch=int(os.environ.get("LLM_BATCH", "256")),
    n_gpu_layers=detect_n_gpu_layers(),
    chat_format="qwen",
    verbose=False, 
)

def chat_with_model(system_msg: str, user_msg: str) -> str:
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        top_p=float(os.environ.get("LLM_TOP_P", "0.9")),
        max_tokens=int(os.environ.get("LLM_MAX_NEW_TOKENS", "600")),
        repeat_penalty=1.05,
        stop=["</s>"],
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()
