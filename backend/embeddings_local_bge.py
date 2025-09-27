# embeddings_local_bge.py
from typing import List
import numpy as np
from langchain.embeddings.base import Embeddings

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:
    raise RuntimeError(
        "Нет FlagEmbedding. Установи: pip install FlagEmbedding==1.2.10"
    ) from e


class LocalBGEM3Embeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model_name = model_name
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=True if device != "cpu" else False,
            device=device
        )

    def _emb(self, texts: List[str]) -> List[List[float]]:
        outs = self._model.encode(texts, batch_size=32, max_length=4096)
        dense = outs["dense_vecs"]
        dense = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)
        return dense.astype(np.float32).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        return self._emb(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._emb([text])[0]
