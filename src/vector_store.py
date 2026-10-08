from sentence_transformers import SentenceTransformer
import uuid
from chromadb.config import Settings
from typing import List,Tuple,Dict,Any
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


"""Manage documet embedding in the choromadb vector store
A client is an object that your Python application uses to interact with ChromaDB.

"""

class Vectorstore:
    def __init__(self,collection_name="pdf_documents",persistent_directory:str="../data/vector_store"):
        self.collection_name=collection_name
        self.persistent_directory=persistent_directory
        self.client=None
        self.collection=None
        self._initialize_store()
    def _initialize_store(self):
        """Initialize choromadb collection and client"""
        try:
            # create persistent choromadb collection
            os.makedirs(self.persistent_directory ,exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persistent_directory)
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embedding for RAG"}
            )
            print(f"Vector store initialize. {self.collection_name}")
            print(f"Existing document in collection {self.collection.count()}")
        except Exception as e:
            print(f"error in collection {e}")
    def add_documents(self, documents:List[Any],embeddings:np.ndarray):
        """add documents and their embeddings to the vector store in the chromadb
        arg:
        documents:list of langchain document
        embeddings: coresponding embedings of document
        """
        if len(documents)!=len(embeddings):
            raise ValueError("Number of document must match the embedding")
        print(f"ADD {len(documents)}  documents to the chroma db")
     #prepare data for choroma db
        ids=[] 
        metadatas=[]
        documents_text=[]
        embeddings_list=[]

        for i ,(doc,embedding) in enumerate(zip(documents,embeddings)):
            ## prepare for id
            doc_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            #prepare for documents
            documents_text.append(doc.page_content)
            #prepare for embbedings
            embeddings_list.append(embedding.tolist())
            #prepare for the metadatas
            metadata=dict(doc.metadata)
            metadata['doc_index']=i
            metadata['content_length']=len(doc.page_content)
            metadatas.append(metadata)
            
        try:
         self.collection.add(
         ids=ids,
         metadatas=metadatas,
         embeddings=embeddings_list,
         documents=documents_text          
                      )   
         print(f"Metadata length {len(metadata)}")                
         print(f" successfully Add the {len(documents)} document to the vector db")
         print(f"Total {(self.collection.count())} documents in the vecttore db")
        except Exception as e:
            print(f"error in adding the documents {e}")
            raise
