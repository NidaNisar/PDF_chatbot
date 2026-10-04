import os
from langchain_community.document_loaders import PyMuPDFLoader,PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_reader(pdf_directory):
    all_documents=[]
    pdf_dir=Path(pdf_directory)
    pdf_files=list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files) } for Processing")
    for pdf in pdf_files:
        print(f"Processing {pdf.name}")
        try:
             pdf_loader=PyPDFLoader(str(pdf))
             pdf_documents=pdf_loader.load()
             ### add source information to metadata
             for doc in pdf_documents:
              doc.metadata['source_file']=pdf.name
              doc.metadata['file_type']='pdf'
             print(f"Document {doc.metadata['producer']}")
             all_documents.extend(pdf_documents)
        except Exception as e:
            print(f"Error {e}")
        print(f"total documents loaded :{len(all_documents)}")
        
    return all_documents