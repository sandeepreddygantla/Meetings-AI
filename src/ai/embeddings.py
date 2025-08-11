"""
Embedding operations for Meetings AI application.
Handles text embedding generation and vector operations.
"""
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple

from meeting_processor import get_embedding_model_safe as get_embedding_client, embedding_model

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for handling embedding operations."""
    
    def __init__(self):
        """Initialize embedding service."""
        self.embedding_client = None
    
    def ensure_client(self) -> bool:
        """
        Ensure embedding client is available.
        
        Returns:
            True if client is available
        """
        if self.embedding_client is None:
            self.embedding_client = get_embedding_client()
        
        return self.embedding_client is not None
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            if not self.ensure_client():
                logger.error("Embedding client not available")
                return None
            
            embedding_client = get_embedding_client()
            if embedding_client:
                embeddings = embedding_client.embed_documents([text])
            else:
                embeddings = []
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                logger.error("No embeddings returned")
                return None
                
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None
    
    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors or None if failed
        """
        try:
            if not texts:
                return []
            
            if not self.ensure_client():
                logger.error("Embedding client not available")
                return None
            
            embedding_client = get_embedding_client()
            if embedding_client:
                embeddings = embedding_client.embed_documents(texts)
            else:
                embeddings = []
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return None
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed document chunks and add embeddings to chunk data.
        
        Args:
            chunks: List of chunk dictionaries with 'content' field
            
        Returns:
            List of chunks with 'embedding' field added
        """
        try:
            if not chunks:
                return []
            
            # Extract texts from chunks
            texts = [chunk.get('content', '') for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embed_texts(texts)
            
            if not embeddings:
                logger.error("Failed to generate embeddings for chunks")
                return chunks  # Return original chunks without embeddings
            
            # Add embeddings to chunks
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = chunk.copy()
                if i < len(embeddings):
                    enhanced_chunk['embedding'] = embeddings[i]
                enhanced_chunks.append(enhanced_chunk)
            
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            return chunks  # Return original chunks on error
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[Tuple[Any, List[float]]], 
        top_k: int = 5
    ) -> List[Tuple[Any, float]]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of (identifier, embedding) tuples
            top_k: Number of top results to return
            
        Returns:
            List of (identifier, similarity_score) tuples, sorted by similarity
        """
        try:
            similarities = []
            
            for identifier, embedding in candidate_embeddings:
                similarity = self.calculate_similarity(query_embedding, embedding)
                similarities.append((identifier, similarity))
            
            # Sort by similarity in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar embeddings: {e}")
            return []
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of embeddings from the current model.
        
        Returns:
            Embedding dimension or None if unavailable
        """
        try:
            if not self.ensure_client():
                return None
            
            # Test embedding to get dimension
            test_embedding = self.embed_text("test")
            if test_embedding:
                return len(test_embedding)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return None
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding is properly formatted.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if embedding is valid
        """
        try:
            if not isinstance(embedding, list):
                return False
            
            if len(embedding) == 0:
                return False
            
            # Check that all elements are numbers
            for value in embedding:
                if not isinstance(value, (int, float)):
                    return False
            
            # Check for NaN or infinite values
            array = np.array(embedding)
            if np.any(np.isnan(array)) or np.any(np.isinf(array)):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating embedding: {e}")
            return False