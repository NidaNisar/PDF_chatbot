from src.data_loader import pdf_reader
from src.chunking import split_document
from Embeddings import Embedding_manager
if __name__=="__main__":
    doc=pdf_reader("data")
    chunks=split_document(doc)
    embedding_model = Embedding_manager()
    
    texts = [doc.page_content for doc in chunks]
    
    embeddings = embedding_model.generate_embeddings(texts)
    print(embeddings)
  