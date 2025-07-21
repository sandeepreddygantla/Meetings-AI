"""
Vector database operations using FAISS for similarity search.
This module handles all FAISS-related operations extracted from the VectorDatabase class.
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
    """Handles all FAISS vector database operations"""
    
    def __init__(self, index_path: str = "vector_index.faiss", dimension: int = 3072):
        """
        Initialize vector operations
        
        Args:
            index_path: Path to FAISS index file
            dimension: Vector dimension (3072 for text-embedding-3-large)
        """
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.chunk_metadata = {}  # Maps FAISS index positions to chunk IDs
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created new FAISS index")
    
    def rebuild_chunk_metadata(self, db_path: str):
        """
        Rebuild chunk_metadata mapping from database after loading FAISS index
        
        Args:
            db_path: Path to SQLite database
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all chunks ordered by their creation (to match FAISS index order)
            cursor.execute('''
                SELECT chunk_id, chunk_index, document_id 
                FROM chunks 
                ORDER BY document_id, chunk_index
            ''')
            
            chunks_data = cursor.fetchall()
            conn.close()
            
            # Rebuild the mapping assuming chunks were added in order
            self.chunk_metadata = {}
            for i, (chunk_id, chunk_index, document_id) in enumerate(chunks_data):
                self.chunk_metadata[i] = chunk_id
            
            logger.info(f"Rebuilt chunk metadata mapping with {len(self.chunk_metadata)} entries")
            
        except Exception as e:
            logger.error(f"Error rebuilding chunk metadata: {e}")
            self.chunk_metadata = {}
    
    def add_vectors(self, vectors: List[np.ndarray], chunk_ids: List[str]):
        """
        Add vectors to FAISS index with corresponding chunk IDs
        
        Args:
            vectors: List of embedding vectors
            chunk_ids: List of corresponding chunk IDs
        """
        if not vectors:
            return
        
        try:
            vectors_array = np.array(vectors).astype('float32')
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors_array)
            
            # Store chunk metadata mapping before adding to index
            start_idx = self.index.ntotal
            for i, chunk_id in enumerate(chunk_ids):
                self.chunk_metadata[start_idx + i] = chunk_id
            
            # Add vectors to FAISS index
            self.index.add(vectors_array)
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {e}")
            raise
    
    def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using FAISS
        
        Args:
            query_embedding: Query vector
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query vector
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_vector)
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx in self.chunk_metadata:
                    chunk_id = self.chunk_metadata[idx]
                    similarity = float(similarities[0][i])
                    results.append((chunk_id, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
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
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
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
            "index_type": type(self.index).__name__,
            "metadata_entries": len(self.chunk_metadata)
        }
    
    def clear_index(self):
        """Clear the FAISS index and metadata"""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunk_metadata = {}
            logger.info("Cleared FAISS index and metadata")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise