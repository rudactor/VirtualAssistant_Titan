import chromadb
from docx_parser import DocumentParser
import pdfplumber
import os

def main():
    client = chromadb.Client()
    collection = client.create_collection(name="train_documentation")
    
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
    
list_files = os.listdir("documentation/")
for file in list_files:
    if file.endswith(".docx"):
        print(parser_docx(DocumentParser(f"documentation/{file}")))
    elif file.endswith(".pdf"):
        print(parser_pdf(f"documentation/{file}"))