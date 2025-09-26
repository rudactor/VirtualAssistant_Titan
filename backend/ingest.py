from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os

EMBEDDING_MODEL = 'bge-m3'
BASE = "documentation/"
DB_PATH = "./sql_chroma_db"

list_files = os.listdir(BASE)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

def add_chunk_to_database(chunks):
    Chroma.from_documents(documents=chunks,  embedding=embeddings, persist_directory=DB_PATH)
    print(f"✅ Добавлено {len(chunks)} чанков в базу")

def ingest_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} documents into {len(chunks)} chunks.")
    add_chunk_to_database(chunks)
    
    
def ingest_docx(docx_path: str):
    loader = Docx2txtLoader(docx_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"[DOCX] Split {len(documents)} documents into {len(chunks)} chunks.")
    add_chunk_to_database(chunks)

def parser_files():
    for file in list_files:
        if file.endswith(".pdf"):
            ingest_pdf(f"{BASE}{file}")
            
        elif file.endswith(".docx"):
            ingest_docx(f"{BASE}{file}")
            
parser_files()