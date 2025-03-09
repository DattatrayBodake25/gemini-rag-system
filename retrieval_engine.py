import logging
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RetrievalEngine:
    """
    Handles the retrieval of relevant text chunks based on a query.
    Uses FAISS for vector search and cosine similarity for re-ranking.
    """
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager

    def retrieve_chunks(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
        """
        Retrieve and rerank the top-k most relevant chunks for a given query.
        
        Args:
            query (str): The user's search query.
            chunks (List[str]): List of document chunks.
            top_k (int): Number of top relevant chunks to retrieve (default: 3).
        
        Returns:
            List[str]: The most relevant text chunks ranked by similarity.
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            
            # Retrieve top-k most relevant chunk indices from FAISS
            indices = self.embedding_manager.search(query_embedding, top_k)
            
            # Fetch embeddings for retrieved chunks in a single batch
            chunk_embeddings = np.array([self.embedding_manager.generate_embeddings([chunks[i]])[0] for i in indices])
            
            # Compute cosine similarity between query and retrieved chunks
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
            
            # Sort retrieved chunks based on similarity scores
            sorted_chunks = sorted(zip(indices, similarities), key=lambda x: x[1], reverse=True)
            reranked_chunks = [chunks[i] for i, _ in sorted_chunks]
            
            logging.info(f"Successfully retrieved {len(reranked_chunks)} relevant chunks.")
            return reranked_chunks
        
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            return []