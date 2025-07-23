"""
Simplified vector database operations using standard FAISS IndexFlatIP.
This module handles vector storage and search operations only - no deletion complexity.
"""

import os
import logging
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sqlite3

# Import global variables from the main module
from meeting_processor import access_token, embedding_model, llm

logger = logging.getLogger(__name__)


class VectorOperations:
    """Handles simplified FAISS vector database operations - upload and search only"""
    
    def __init__(self, index_path: str = "vector_index.faiss", dimension: int = 3072):
        """
        Initialize vector operations with simple IndexFlatIP
        
        Args:
            index_path: Path to FAISS index file
            dimension: Vector dimension (3072 for text-embedding-3-large)
        """
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new simple IndexFlatIP"""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}, creating new one")
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info("Created new FAISS IndexFlatIP")
        else:
            # Create new simple IndexFlatIP
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created new empty FAISS IndexFlatIP")
    
    def add_vectors(self, vectors: List[np.ndarray], chunk_ids: List[str]):
        """
        Add vectors to simple FAISS IndexFlatIP
        
        Args:
            vectors: List of embedding vectors
            chunk_ids: List of corresponding chunk IDs (for logging only)
        """
        if not vectors:
            return
        
        try:
            vectors_array = np.array(vectors).astype('float32')
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors_array)
            
            # Add vectors to simple FAISS index (no IDs needed)
            self.index.add(vectors_array)
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index. Total vectors now: {self.index.ntotal}")
            
            # Automatically save index after adding vectors
            self.save_index()
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {e}")
            raise
    
    def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using simple FAISS IndexFlatIP
        
        Args:
            query_embedding: Query vector
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty - no vectors to search")
            return []
        
        try:
            logger.info(f"Starting FAISS search: index has {self.index.ntotal} vectors, requesting top {top_k}")
            
            # Normalize query vector
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_vector)
            
            # Search in simple FAISS index - returns positions
            similarities, positions = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            logger.info(f"FAISS returned {len(positions[0])} results")
            
            # Convert vector positions to chunk_ids using database
            results = []
            if len(positions[0]) > 0:
                results = self._map_positions_to_chunk_ids(positions[0], similarities[0])
            
            logger.info(f"Final search results: {len(results)} chunks returned")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return []
    
    def _map_positions_to_chunk_ids(self, positions: List[int], similarities: List[float]) -> List[Tuple[str, float]]:
        """
        Map FAISS vector positions to chunk_ids using database order
        
        Args:
            positions: Vector positions from FAISS search
            similarities: Similarity scores from FAISS search
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        try:
            db_path = "meeting_documents.db"
            if not os.path.exists(db_path):
                logger.error(f"Database not found at {db_path}")
                return []
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get chunk_ids in the same order they were added to FAISS
            cursor.execute('SELECT chunk_id FROM chunks ORDER BY rowid')
            all_chunk_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            results = []
            for i, position in enumerate(positions):
                if position != -1 and position < len(all_chunk_ids):
                    chunk_id = all_chunk_ids[position]
                    similarity = float(similarities[i])
                    results.append((chunk_id, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error mapping positions to chunk_ids: {e}")
            return []
    
    def search_similar_chunks_by_folder(self, query_embedding: np.ndarray, user_id: str, 
                                      folder_path: str, db_path: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using FAISS, filtered by folder
        
        Args:
            query_embedding: Query vector
            user_id: User ID for filtering
            folder_path: Folder path for filtering
            db_path: Path to SQLite database
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # First get all chunks from semantic search
            all_results = self.search_similar_chunks(query_embedding, top_k * 3)  # Get more to filter
            
            # Filter results by folder
            filtered_results = []
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for chunk_id, similarity in all_results:
                # Check if this chunk belongs to a document in the specified folder
                cursor.execute('''
                    SELECT 1 FROM chunks c
                    JOIN documents d ON c.document_id = d.document_id
                    WHERE c.chunk_id = ? AND d.user_id = ? AND d.folder_path = ?
                ''', (chunk_id, user_id, folder_path))
                
                if cursor.fetchone():
                    filtered_results.append((chunk_id, similarity))
                    if len(filtered_results) >= top_k:
                        break
            
            conn.close()
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks by folder: {e}")
            return []
    
    def save_index(self):
        """Save FAISS index to disk"""
        try:
            if self.index:
                logger.info(f"Attempting to save FAISS index to {self.index_path}")
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Successfully saved FAISS index with {self.index.ntotal} vectors to {self.index_path}")
                
                # Verify file was created
                import os
                if os.path.exists(self.index_path):
                    file_size = os.path.getsize(self.index_path)
                    logger.info(f"FAISS index file created: {self.index_path} ({file_size} bytes)")
                else:
                    logger.error(f"FAISS index file was NOT created: {self.index_path}")
            else:
                logger.warning("No FAISS index to save (index is None)")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index
        
        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            return {"total_vectors": 0, "dimension": self.dimension, "index_type": None}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__
        }
    
    def clear_index(self):
        """Clear the FAISS index"""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Cleared FAISS index")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise
    
    def rebuild_chunk_metadata(self, db_path: str):
        """No-op method for backward compatibility - not needed in simplified architecture"""
        logger.info("Chunk metadata rebuild not needed - using simplified IndexFlatIP")
        pass
