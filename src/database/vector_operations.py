"""
Simplified vector database operations using standard FAISS IndexFlatIP.
This module handles vector storage and search operations only - no deletion complexity.
"""

import os
import logging
import faiss
import numpy as np
import hashlib
import time
import threading
from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache
import sqlite3
from pathlib import Path

# Import global variables from the main module
from meeting_processor import access_token, embedding_model, llm

logger = logging.getLogger(__name__)


def get_application_root() -> Path:
    """
    Get the application root directory (where flask_app.py and meeting_processor.py are located).
    This ensures FAISS index files are saved in the correct location regardless of working directory.
    """
    # Get the path of this file (src/database/vector_operations.py)
    current_file = Path(__file__)
    
    # Go up two levels: src/database -> src -> application root
    app_root = current_file.parent.parent.parent
    
    # Verify we're in the right place by checking for key files
    if (app_root / "flask_app.py").exists() and (app_root / "meeting_processor.py").exists():
        logger.debug(f"Application root detected: {app_root.absolute()}")
        return app_root.absolute()
    else:
        # Fallback: use current working directory
        logger.warning(f"Could not detect application root from {current_file}, using current directory")
        return Path.cwd()


class VectorOperations:
    """Handles simplified FAISS vector database operations with search caching"""
    
    def __init__(self, index_path: str = "vector_index.faiss", dimension: int = 3072, cache_size: int = 100):
        """
        Initialize vector operations with simple IndexFlatIP and search caching
        
        Args:
            index_path: Path to FAISS index file (relative paths will be resolved to application root)
            dimension: Vector dimension (3072 for text-embedding-3-large)  
            cache_size: Maximum number of search results to cache
        """
        # Convert relative paths to absolute paths based on application root
        if not os.path.isabs(index_path):
            app_root = get_application_root()
            self.index_path = str(app_root / index_path)
            logger.info(f"Converted relative path '{index_path}' to absolute: '{self.index_path}'")
        else:
            self.index_path = index_path
            logger.info(f"Using absolute path: '{self.index_path}'")
        self.dimension = dimension
        self.index = None
        
        # Initialize search result cache
        self.cache_size = cache_size
        self._search_cache = {}  # query_hash -> (results, timestamp)
        self._cache_lock = threading.Lock()
        self._cache_ttl = 300  # Cache entries expire after 5 minutes
        
        self._load_or_create_index()
    
    def _generate_query_hash(self, query_embedding: np.ndarray, top_k: int) -> str:
        """Generate a hash for the query to use as cache key"""
        # Create hash from embedding values and top_k parameter
        query_str = f"{query_embedding.tobytes()}{top_k}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _cleanup_expired_cache(self):
        """Remove expired entries from search cache"""
        current_time = time.time()
        expired_keys = []
        
        for query_hash, (results, timestamp) in self._search_cache.items():
            if current_time - timestamp > self._cache_ttl:
                expired_keys.append(query_hash)
        
        for key in expired_keys:
            del self._search_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _get_cached_search_results(self, query_hash: str) -> Optional[List[Tuple[str, float]]]:
        """Get cached search results if available and not expired"""
        with self._cache_lock:
            if query_hash in self._search_cache:
                results, timestamp = self._search_cache[query_hash]
                if time.time() - timestamp <= self._cache_ttl:
                    logger.debug("Returning cached search results")
                    return results
                else:
                    # Remove expired entry
                    del self._search_cache[query_hash]
        return None
    
    def _cache_search_results(self, query_hash: str, results: List[Tuple[str, float]]):
        """Cache search results with timestamp"""
        with self._cache_lock:
            # Clean up expired entries periodically
            if len(self._search_cache) >= self.cache_size:
                self._cleanup_expired_cache()
                
                # If cache is still full, remove oldest entries
                if len(self._search_cache) >= self.cache_size:
                    # Remove 20% of oldest entries to make room
                    items = sorted(self._search_cache.items(), key=lambda x: x[1][1])
                    remove_count = max(1, len(items) // 5)
                    for query_hash_to_remove, _ in items[:remove_count]:
                        del self._search_cache[query_hash_to_remove]
            
            self._search_cache[query_hash] = (results, time.time())
            logger.debug(f"Cached search results. Cache size: {len(self._search_cache)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        with self._cache_lock:
            current_time = time.time()
            expired_count = sum(1 for _, (_, timestamp) in self._search_cache.items() 
                              if current_time - timestamp > self._cache_ttl)
            
            return {
                'total_entries': len(self._search_cache),
                'expired_entries': expired_count,
                'active_entries': len(self._search_cache) - expired_count,
                'max_cache_size': self.cache_size,
                'cache_ttl_seconds': self._cache_ttl
            }
    
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
        Search for similar chunks using FAISS IndexFlatIP with caching
        
        Args:
            query_embedding: Query vector
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty - no vectors to search")
            return []
        
        # Check cache first
        query_hash = self._generate_query_hash(query_embedding, top_k)
        cached_results = self._get_cached_search_results(query_hash)
        if cached_results is not None:
            return cached_results
        
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
            
            # Cache the results before returning
            self._cache_search_results(query_hash, results)
            
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
            
            # Get chunk_ids in the same order they were added to FAISS (document processing order)
            cursor.execute('SELECT chunk_id FROM chunks ORDER BY document_id, chunk_index')
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
        """Save FAISS index to disk with comprehensive path diagnostics"""
        try:
            if self.index:
                # Add comprehensive path diagnostics
                logger.info(f"=== FAISS Save Index Diagnostics ===")
                logger.info(f"Current working directory: {os.getcwd()}")
                logger.info(f"Target index path: {self.index_path}")
                logger.info(f"Index path is absolute: {os.path.isabs(self.index_path)}")
                
                # Check directory permissions
                index_dir = os.path.dirname(self.index_path)
                logger.info(f"Index directory: {index_dir}")
                logger.info(f"Directory exists: {os.path.exists(index_dir)}")
                logger.info(f"Directory writable: {os.access(index_dir, os.W_OK)}")
                
                # Check if file already exists
                if os.path.exists(self.index_path):
                    logger.info(f"Index file exists, size: {os.path.getsize(self.index_path)} bytes")
                    logger.info(f"File writable: {os.access(self.index_path, os.W_OK)}")
                
                # Test write permissions with a temporary file
                try:
                    test_file = self.index_path + ".tmp"
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    logger.info("✅ Write permission test: SUCCESS")
                except Exception as perm_e:
                    logger.error(f"❌ Write permission test: FAILED - {perm_e}")
                
                logger.info(f"Attempting to save FAISS index with {self.index.ntotal} vectors...")
                faiss.write_index(self.index, self.index_path)
                logger.info(f"✅ Successfully saved FAISS index to {self.index_path}")
                
                # Verify file was created and get detailed info
                if os.path.exists(self.index_path):
                    file_size = os.path.getsize(self.index_path)
                    file_stat = os.stat(self.index_path)
                    logger.info(f"✅ FAISS index file verified: {self.index_path}")
                    logger.info(f"   - File size: {file_size} bytes")
                    logger.info(f"   - Last modified: {time.ctime(file_stat.st_mtime)}")
                else:
                    logger.error(f"❌ FAISS index file was NOT created: {self.index_path}")
                    
                logger.info(f"=== End FAISS Save Diagnostics ===")
            else:
                logger.warning("No FAISS index to save (index is None)")
        except Exception as e:
            logger.error(f"❌ Error saving FAISS index: {type(e).__name__}: {e}")
            logger.error(f"   - Index path: {self.index_path}")
            logger.error(f"   - Working directory: {os.getcwd()}")
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
