import chromadb
from docx_parser import DocumentParser
import pdfplumber
from sentence_transformers import SentenceTransformer
import os

def main(docs):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="docs")
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs).tolist()
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=["doc1", "doc2"]
    )
    
def parser_docx(document: DocumentParser):
    string_result = ""
    for type, item in document.parse():
        print(item)
    
def parser_pdf(pdfpath):
    with pdfplumber.open(pdfpath) as pdf:
        all_text = ""
        all_tables = []
        for page in pdf.pages:
            all_text += page.extract_text()
            all_tables.extend(page.extract_tables())
        return all_text, all_tables
    
docs = ''
    
list_files = os.listdir("documentation/")
for file in list_files:
    if file.endswith(".docx"):
        docs += str(parser_docx(DocumentParser(f"documentation/{file}")))
    elif file.endswith(".pdf"):
        docs += str(parser_pdf(f"documentation/{file}"))
        
