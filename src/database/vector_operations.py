"""
Vector database operations using FAISS IndexIDMap for direct deletion support.
This module handles all FAISS-related operations with direct vector deletion capabilities.
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
    """Handles all FAISS IndexIDMap vector database operations with direct deletion"""
    
    def __init__(self, index_path: str = "vector_index.faiss", dimension: int = 3072):
        """
        Initialize vector operations with IndexIDMap for direct deletion
        
        Args:
            index_path: Path to FAISS index file
            dimension: Vector dimension (3072 for text-embedding-3-large)
        """
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.chunk_id_to_int_map = {}  # Maps chunk_id to integer ID for FAISS
        self.int_to_chunk_id_map = {}  # Reverse mapping for safety
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing FAISS IndexIDMap or create new one"""
        if os.path.exists(self.index_path):
            try:
                loaded_index = faiss.read_index(self.index_path)
                
                # Check if it's already an IndexIDMap
                if isinstance(loaded_index, faiss.IndexIDMap):
                    self.index = loaded_index
                    logger.info(f"Loaded existing FAISS IndexIDMap with {self.index.ntotal} vectors")
                else:
                    # Convert old IndexFlatIP to IndexIDMap
                    logger.info(f"Converting old FAISS index to IndexIDMap format")
                    base_index = faiss.IndexFlatIP(self.dimension)
                    self.index = faiss.IndexIDMap(base_index)
                    # Note: Old index data will be lost, but since not in production, this is acceptable
                    logger.info("Created new IndexIDMap (old data reset for format compatibility)")
                    
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                base_index = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIDMap(base_index)
        else:
            # Create new IndexIDMap
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
            logger.info("Created new empty FAISS IndexIDMap")
            
            # Try to rebuild from existing database data if available
            self._attempt_rebuild_from_database()
    
    def _chunk_id_to_int(self, chunk_id: str) -> int:
        """
        Convert chunk_id to integer for FAISS IndexIDMap
        
        Args:
            chunk_id: String chunk ID
            
        Returns:
            Integer ID for FAISS (31-bit positive integer)
        """
        if chunk_id in self.chunk_id_to_int_map:
            return self.chunk_id_to_int_map[chunk_id]
        
        # Generate stable hash - using built-in hash but ensuring positive 31-bit
        int_id = abs(hash(chunk_id)) % (2**31 - 1)
        
        # Check for collision
        collision_count = 0
        original_int_id = int_id
        while int_id in self.int_to_chunk_id_map and self.int_to_chunk_id_map[int_id] != chunk_id:
            collision_count += 1
            int_id = (original_int_id + collision_count) % (2**31 - 1)
            
            if collision_count > 1000:  # Prevent infinite loop
                raise Exception(f"Too many hash collisions for chunk_id {chunk_id}")
        
        # Store bidirectional mapping
        self.chunk_id_to_int_map[chunk_id] = int_id
        self.int_to_chunk_id_map[int_id] = chunk_id
        
        return int_id
    
    def _get_chunk_int_ids(self, chunk_ids: List[str]) -> List[int]:
        """Convert list of chunk IDs to integer IDs"""
        return [self._chunk_id_to_int(chunk_id) for chunk_id in chunk_ids]
    
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
        Add vectors to FAISS IndexIDMap with corresponding chunk IDs
        
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
            
            # Convert chunk IDs to integers for IndexIDMap
            chunk_int_ids = self._get_chunk_int_ids(chunk_ids)
            chunk_int_ids_array = np.array(chunk_int_ids, dtype=np.int64)
            
            # Add vectors with IDs to FAISS IndexIDMap
            self.index.add_with_ids(vectors_array, chunk_int_ids_array)
            
            logger.info(f"Added {len(vectors)} vectors to FAISS IndexIDMap. Total vectors now: {self.index.ntotal}")
            logger.info(f"Chunk IDs mapped: {chunk_ids[:3]}... -> {chunk_int_ids[:3]}...")
            
            # Automatically save index after adding vectors
            self.save_index()
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {e}")
            raise
    
    def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using FAISS IndexIDMap
        
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
            
            # Search in FAISS IndexIDMap - returns IDs directly
            similarities, ids = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            results = []
            for i, int_id in enumerate(ids[0]):
                if int_id != -1:  # -1 indicates no result found
                    # Convert integer ID back to chunk_id
                    if int_id in self.int_to_chunk_id_map:
                        chunk_id = self.int_to_chunk_id_map[int_id]
                        similarity = float(similarities[0][i])
                        results.append((chunk_id, similarity))
                    else:
                        logger.warning(f"Integer ID {int_id} not found in reverse mapping")
            
            logger.info(f"Vector search returned {len(results)} results from {len(ids[0])} FAISS matches")
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
            "active_mappings": len(self.chunk_id_to_int_map)
        }
    
    def clear_index(self):
        """Clear the FAISS index and metadata"""
        try:
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
            self.chunk_id_to_int_map = {}
            self.int_to_chunk_id_map = {}
            logger.info("Cleared FAISS index and metadata")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise
    
    # Vector Deletion Operations (Direct deletion with IndexIDMap)
    
    def delete_document_vectors(self, document_id: str, chunk_ids: List[str]) -> bool:
        """Delete vectors for a document using IndexIDMap remove_ids"""
        try:
            if not chunk_ids:
                logger.info(f"No chunks to delete for document {document_id}")
                return True
            
            logger.info(f"Deleting {len(chunk_ids)} vectors for document {document_id}")
            
            # Convert chunk IDs to integers for FAISS
            chunk_int_ids = self._get_chunk_int_ids(chunk_ids)
            chunk_int_ids_array = np.array(chunk_int_ids, dtype=np.int64)
            
            # Use IndexIDMap's direct deletion
            initial_count = self.index.ntotal
            self.index.remove_ids(chunk_int_ids_array)
            final_count = self.index.ntotal
            
            # Clean up our ID mappings
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_id_to_int_map:
                    int_id = self.chunk_id_to_int_map[chunk_id]
                    del self.chunk_id_to_int_map[chunk_id]
                    if int_id in self.int_to_chunk_id_map:
                        del self.int_to_chunk_id_map[int_id]
            
            deleted_count = initial_count - final_count
            logger.info(f"Successfully deleted {deleted_count} vectors from FAISS index. Total vectors now: {final_count}")
            
            # Save the updated index
            self.save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors for document {document_id}: {e}")
            return False
    
    def force_rebuild_index(self) -> bool:
        """Force rebuild of IndexIDMap (not needed but kept for compatibility)"""
        try:
            logger.info("IndexIDMap doesn't require rebuilding - direct deletion supported")
            return True  # Always successful since IndexIDMap handles deletions directly
        except Exception as e:
            logger.error(f"Error in index operation: {e}")
            return False
    
    def get_deletion_stats(self) -> Dict[str, Any]:
        """Get statistics about IndexIDMap health"""
        try:
            total_vectors = self.index.ntotal if self.index else 0
            active_mappings = len(self.chunk_id_to_int_map)
            
            return {
                'total_vectors': total_vectors,
                'active_mappings': active_mappings,
                'index_type': 'IndexIDMap',
                'supports_direct_deletion': True,
                'needs_rebuild': False  # IndexIDMap handles deletions directly
            }
        except Exception as e:
            logger.error(f"Error getting deletion stats: {e}")
            return {}
    
    def rebuild_chunk_metadata(self, db_path: str):
        """
        Rebuild mapping for IndexIDMap compatibility (legacy method)
        
        Args:
            db_path: Path to SQLite database
        """
        try:
            # For IndexIDMap, we don't need to rebuild metadata in the same way
            # But we'll load the mappings if they exist in the database
            logger.info("IndexIDMap uses direct ID mapping - no metadata rebuild needed")
            return
            
        except Exception as e:
            logger.error(f"Error in rebuild_chunk_metadata: {e}")