from sentence_transformers import SentenceTransformer
import uuid
from chromadb.config import Settings
from typing import List,Tuple,Dict,Any
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


### EMBEDDING convert text into vector
class Embedding_manager:
    """ Handles document embedding using sentence-transformer"""
    def __init__(self,model_name:str='all-MiniLM-L6-v2'):
        self.model_name=model_name
        self.model=None
        self._load_model()
    def _load_model(self):
      try:
         print(f"Loading embedding model {self.model_name}")
         self.model=SentenceTransformer(self.model_name)
         """ Embedding dimension is the number of numerical values used to represent a piece of data—such as text, an image, or a document—as a vector."""
         print(f"Model loaded Sucessfully .Embedding Dimension:{self.model.get_embedding_dimension()}")
      except  Exception as e:
         print(f"Error loading model: {e}")
         raise
    def generate_embeddings(self,Text:List[str]):
       """
       Generate embedding for list of text
       Text: list of text string to embed
       return: numpy array for embbeding shape
       """
       if not self.model:
          raise ValueError("Model not loaded")
       print(f"Generate embbedings for the  {len(Text)}....") 
      
       embedding=self.model.encode(Text)
       print(f"Generate embbeding with shape {embedding.shape}")
       return embedding
    """Embedding shape tells you how many numbers are inside the vector and how many vectors you have."""
         

        