import google.generativeai as genai
import faiss
import numpy as np
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingManager:
    def __init__(self, api_key: str):
        """
        Initialize EmbeddingManager with Gemini API key and FAISS index.
        """
        try:
            genai.configure(api_key=api_key)
            self.embedding_model = "models/embedding-001"
            self.index = None
            logging.info("EmbeddingManager initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize EmbeddingManager: {e}")
            raise

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks using Gemini API.
        """
        embeddings = []
        try:
            responses = [genai.embed_content(model=self.embedding_model, content=chunk) for chunk in chunks]
            embeddings = [response['embedding'] for response in responses]
            logging.info("Successfully generated embeddings for text chunks.")
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise
        return embeddings

    def build_faiss_index(self, embeddings: List[List[float]]):
        """
        Build a FAISS index for fast vector search.
        """
        try:
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            logging.info("FAISS index built successfully with %d embeddings.", len(embeddings))
        except Exception as e:
            logging.error(f"Error building FAISS index: {e}")
            raise

    def search(self, query_embedding: List[float], k: int = 3) -> List[int]:
        """
        Search for the top-k most relevant chunks in the FAISS index.
        """
        if self.index is None:
            logging.error("FAISS index is not initialized. Please build the index first.")
            raise ValueError("FAISS index is not initialized.")
        
        try:
            distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
            logging.info("Search completed. Found %d relevant chunks.", len(indices[0]))
            return indices[0].tolist()
        except Exception as e:
            logging.error(f"Error searching FAISS index: {e}")
            raise