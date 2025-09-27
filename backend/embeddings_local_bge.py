# ДАРОУ, это код для того чтобы библиотека с эмбеддингом нормально дружила с ingest.py

from typing import List
from langchain.embeddings.base import Embeddings
from FlagEmbedding import BGEM3FlagModel
import numpy as np

class LocalBGEM3Embeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        normalize: bool = True,
        batch_size: int = 32,
        max_length: int = 8192,
        use_fp16: bool = True,
    ):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_length = max_length

    def _l2_normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _encode(self, texts: List[str]) -> List[List[float]]:
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = out["dense_vecs"]
        if isinstance(vecs, list):
            vecs = np.asarray(vecs, dtype=np.float32)
        if self.normalize:
            vecs = self._l2_normalize(vecs)
        return vecs.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]

