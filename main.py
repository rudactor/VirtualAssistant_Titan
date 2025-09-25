import ollama
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

EMBEDDING_MODEL = 'bge-m3'
LANGUAGE_MODEL = 'gemma3:4b'
DB_PATH = "./sql_chroma_db"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embeddings)

def retrieve(query, top_n=3):
    results = vector_store.similarity_search_with_score(query, k=top_n)
    return results

# def cosine_similarity(a, b):
#     dot_product = sum([x * y for x, y in zip(a, b)])
#     norm_a = sum([x ** 2 for x in a]) ** 0.5
#     norm_b = sum([x ** 2 for x in b]) ** 0.5
#     return dot_product / (norm_a * norm_b)

# for chunk, similarity in retrieved_knowledge:
#     print(f' - (similarity: {similarity:.2f}) {chunk}')

def main(input_query: str):
    start = time.time()
    retrieved_knowledge = retrieve(input_query)
        
    instruction_prompt = "Ты полезный и умный чат бот. Используй только информацию которую тебе дают и дай четкие ответы по контексту: " + str({'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])})

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {
                'role': 'system',
                'content': instruction_prompt
            },
            {
                'role': 'user',
                'content': input_query
            },
        ],
        stream=True
    )

    result = ''
    for chunk in stream:
        result += chunk['message']['content']
    
    end = time.time()
    print(end - start)
    return result