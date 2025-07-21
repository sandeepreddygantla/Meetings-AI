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
            logger.info("Created new empty FAISS index")
            
            # Try to rebuild from existing database data if available
            self._attempt_rebuild_from_database()
    
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
            
            # Validate consistency between FAISS index and metadata
            if self.index and self.index.ntotal != len(self.chunk_metadata):
                logger.error(f"Vector index size ({self.index.ntotal}) doesn't match metadata entries ({len(self.chunk_metadata)})")
                logger.error("This indicates the FAISS index and database are out of sync")
                logger.error("Please delete vector_index.faiss and restart to rebuild from database")
            
        except Exception as e:
            logger.error(f"Error rebuilding chunk metadata: {e}")
            self.chunk_metadata = {}
    
    def _attempt_rebuild_from_database(self):
        """
        Attempt to rebuild FAISS index from existing database chunks
        This is called when vector_index.faiss is missing but database has data
        """
        try:
            # We need to import here to avoid circular imports
            from meeting_processor import embedding_model
            
            if embedding_model is None:
                logger.warning("Embedding model not available for index rebuild")
                return
            
            # Connect to database and get all chunks
            db_path = "meeting_documents.db"  # Default database path
            if not os.path.exists(db_path):
                logger.info("No database found to rebuild from")
                return
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all chunks ordered by document and chunk index
            cursor.execute('''
                SELECT chunk_id, content
                FROM chunks 
                ORDER BY document_id, chunk_index
            ''')
            
            chunks_data = cursor.fetchall()
            conn.close()
            
            if not chunks_data:
                logger.info("No chunks found in database to rebuild index from")
                return
                
            logger.info(f"Found {len(chunks_data)} chunks in database. Starting FAISS index rebuild...")
            
            # Generate embeddings for all chunks in batches
            chunk_ids = []
            vectors = []
            batch_size = 100
            
            for i in range(0, len(chunks_data), batch_size):
                batch = chunks_data[i:i + batch_size]
                batch_content = [chunk[1] for chunk in batch]
                batch_ids = [chunk[0] for chunk in batch]
                
                # Generate embeddings
                try:
                    batch_embeddings = embedding_model.embed_documents(batch_content)
                    vectors.extend([np.array(emb) for emb in batch_embeddings])
                    chunk_ids.extend(batch_ids)
                    logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(chunks_data)-1)//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    continue
            
            if vectors:
                # Add all vectors to FAISS index
                self.add_vectors(vectors, chunk_ids)
                logger.info(f"Successfully rebuilt FAISS index with {len(vectors)} vectors from database")
            else:
                logger.warning("No vectors generated during rebuild attempt")
                
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index from database: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
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
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index. Total vectors now: {self.index.ntotal}")
            
            # Automatically save index after adding vectors
            self.save_index()
            
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
                else:
                    logger.warning(f"FAISS index {idx} not found in chunk_metadata. Available indices: {list(self.chunk_metadata.keys())[:10]}...")
            
            logger.info(f"Vector search returned {len(results)} results from {len(indices[0])} FAISS matches")
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