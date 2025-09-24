import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from utils import make_ids

DB_PATH = "./chroma_db"
COLLECTION = "rzd_docs"

class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, q: str):
        return self.model.encode([f"query: {q}"], normalize_embeddings=True).tolist()[0]

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION)

def retrieve(query: str, top_k: int = 5):
    col = get_collection()
    emb = E5Embedder().embed_query(query)
    res = col.query(query_embeddings=[emb], n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))

def build_prompt(query: str, contexts):
    ctx = "\n\n---\n\n".join([c[0] for c in contexts])
    sys = (
        "Ты — ассистент по нормативам РЖД. Отвечай кратко, опираясь на Контекст. "
        "Если в контексте нет ответа, прямо скажи об этом. Вставляй ссылки на источник (имя файла)."
    )
    prompt = (
        f"{sys}\n\n"
        f"Вопрос: {query}\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Ответ (укажи источники в скобках, например [Инструкция ЦТ 685.pdf]):"
    )
    return prompt

class Gemma:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=True
        )

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens=300):
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    q = input("Вопрос: ").strip()
    ctx = retrieve(q, top_k=6)
    prompt = build_prompt(q, ctx)
    llm = Gemma()
    ans = llm.generate(prompt, max_new_tokens=350)
    print("\n" + "="*60 + "\nОтвет:\n")
    print(ans)
