"""
Main database manager that combines vector and SQLite operations.
This module provides a unified interface for all database operations.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

from .vector_operations import VectorOperations
from .sqlite_operations import SQLiteOperations

# Import global variables and data classes from the main module
from meeting_processor import (
    access_token, embedding_model, llm,
    DocumentChunk, User, Project, Meeting, MeetingDocument
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Unified database manager combining vector and SQLite operations.
    Provides a single interface for all database functionality.
    """
    
    def __init__(self, db_path: str = "meeting_documents.db", index_path: str = "vector_index.faiss"):
        """
        Initialize the database manager
        
        Args:
            db_path: Path to SQLite database file
            index_path: Path to FAISS index file
        """
        self.db_path = db_path
        self.index_path = index_path
        self.dimension = 3072  # text-embedding-3-large dimension
        
        # Initialize both operations handlers
        self.sqlite_ops = SQLiteOperations(db_path)
        self.vector_ops = VectorOperations(index_path, self.dimension)
        
        # Load existing chunk metadata from database
        self.vector_ops.rebuild_chunk_metadata(db_path)
        
        # Keep track of document metadata for compatibility
        self.document_metadata = {}
        
        logger.info("Database manager initialized with SQLite and FAISS operations")
    
    # Combined Operations (Vector + SQLite)
    def add_document(self, document, chunks: List):
        """
        Add document and its chunks to both SQLite and FAISS databases
        
        Args:
            document: MeetingDocument object
            chunks: List of DocumentChunk objects
        """
        try:
            # Extract embeddings for FAISS
            vectors = []
            chunk_ids = []
            
            for chunk in chunks:
                if chunk.embedding is not None:
                    vectors.append(chunk.embedding)
                    chunk_ids.append(chunk.chunk_id)
            
            # Add to SQLite first (includes document and chunk metadata)
            self.sqlite_ops.add_document_and_chunks(document, chunks)
            
            # Add vectors to FAISS index
            if vectors:
                self.vector_ops.add_vectors(vectors, chunk_ids)
            
            # Store document metadata for compatibility
            self.document_metadata[document.document_id] = document
            
            logger.info(f"Successfully added document {document.filename} with {len(chunks)} chunks to both databases")
            
        except Exception as e:
            logger.error(f"Error adding document {document.filename}: {e}")
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
        return self.vector_ops.search_similar_chunks(query_embedding, top_k)
    
    def search_similar_chunks_by_folder(self, query_embedding: np.ndarray, user_id: str, 
                                      folder_path: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using FAISS, filtered by folder
        
        Args:
            query_embedding: Query vector
            user_id: User ID for filtering
            folder_path: Folder path for filtering
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        return self.vector_ops.search_similar_chunks_by_folder(
            query_embedding, user_id, folder_path, self.db_path, top_k
        )
    
    def get_chunks_by_ids(self, chunk_ids: List[str]):
        """
        Retrieve chunks by their IDs with enhanced intelligence metadata
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of DocumentChunk objects
        """
        return self.sqlite_ops.get_chunks_by_ids(chunk_ids)
    
    def enhanced_search_with_metadata(self, query_embedding: np.ndarray, user_id: str, 
                                    filters: Dict = None, top_k: int = 20) -> List[Dict]:
        """
        Enhanced search combining vector similarity with metadata filtering
        
        Args:
            query_embedding: Query vector
            user_id: User ID for filtering
            filters: Additional metadata filters
            top_k: Number of top results to return
            
        Returns:
            List of enhanced search results with metadata
        """
        try:
            # Get initial vector search results
            vector_results = self.search_similar_chunks(query_embedding, top_k * 2)
            
            if not vector_results:
                return []
            
            # Get chunk data from SQLite
            chunk_ids = [chunk_id for chunk_id, _ in vector_results]
            logger.info(f"Vector search returned chunk IDs: {chunk_ids[:5]}...") # Show first 5 IDs
            chunks = self.get_chunks_by_ids(chunk_ids)
            logger.info(f"Retrieved {len(chunks)} chunks from database for {len(chunk_ids)} chunk IDs")
            
            # Debug: Check user IDs in chunks
            chunk_user_ids = set(chunk.user_id for chunk in chunks)
            logger.info(f"Chunk user IDs found: {chunk_user_ids}")
            logger.info(f"Query user ID: '{user_id}'")
            
            # Create results with scores
            enhanced_results = []
            score_map = {chunk_id: score for chunk_id, score in vector_results}
            
            for chunk in chunks:
                logger.debug(f"Checking chunk {chunk.chunk_id}: chunk.user_id='{chunk.user_id}', query user_id='{user_id}'")
                if chunk.user_id == user_id or user_id is None or chunk.user_id is None:
                    result = {
                        'chunk': chunk,
                        'similarity_score': score_map.get(chunk.chunk_id, 0.0),
                        'context': self._reconstruct_chunk_context(chunk)
                    }
                    enhanced_results.append(result)
                else:
                    logger.debug(f"Chunk {chunk.chunk_id} filtered out due to user_id mismatch")
            
            logger.info(f"Enhanced search: {len(enhanced_results)} chunks passed user_id filter (query user_id: '{user_id}')")
            
            # Apply metadata filters if provided
            if filters:
                enhanced_results = self._apply_metadata_filters(enhanced_results, filters, user_id)
            
            # Sort by similarity score and limit results
            enhanced_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return enhanced_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return []
    
    def _apply_metadata_filters(self, search_results: List[Dict], filters: Dict, user_id: str) -> List[Dict]:
        """Apply metadata filters to search results"""
        filtered_results = []
        
        # Debug logging
        logger.info(f"Applying metadata filters: {filters}")
        logger.info(f"Total search results before filtering: {len(search_results)}")
        
        
        for result in search_results:
            chunk = result['chunk']
            
            # Debug logging for each chunk
            logger.info(f"Chunk {chunk.chunk_id}: project_id={getattr(chunk, 'project_id', 'None')}, meeting_id={getattr(chunk, 'meeting_id', 'None')}")
            
            # Apply filters
            if filters.get('date_range'):
                start_date, end_date = filters['date_range']
                if chunk.date:
                    if start_date and chunk.date < start_date:
                        logger.info(f"Filtering out chunk {chunk.chunk_id} - date too early")
                        continue
                    if end_date and chunk.date > end_date:
                        logger.info(f"Filtering out chunk {chunk.chunk_id} - date too late")
                        continue
            
            if filters.get('project_id') and chunk.project_id != filters['project_id']:
                logger.info(f"Filtering out chunk {chunk.chunk_id} - project_id mismatch: {chunk.project_id} != {filters['project_id']}")
                continue
            
            if filters.get('meeting_id') and chunk.meeting_id != filters['meeting_id']:
                logger.info(f"Filtering out chunk {chunk.chunk_id} - meeting_id mismatch")
                continue
            
            if filters.get('keywords'):
                content_lower = chunk.content.lower()
                if not any(keyword.lower() in content_lower for keyword in filters['keywords']):
                    logger.info(f"Filtering out chunk {chunk.chunk_id} - keyword mismatch")
                    continue
            
            if filters.get('folder_path'):
                filter_folder_path = filters['folder_path']
                
                # Handle synthetic folder paths from frontend (e.g., "user_folder/project_XXX") 
                # Extract project_id if the folder_path contains "project_"
                target_project_id = None
                target_project_name = filter_folder_path
                
                if 'project_' in filter_folder_path:
                    # Extract project_id from synthetic folder path like "user_folder/project_0457768f-2769-405b-9f94-bad765055754"
                    parts = filter_folder_path.split('project_')
                    if len(parts) > 1:
                        target_project_id = parts[1]
                        logger.info(f"Extracted project_id from folder_path: {target_project_id}")
                
                # For folder filtering, match against project name or project_id
                # First try to match by actual folder_path if available
                chunk_folder_path = getattr(chunk, 'folder_path', None)
                if chunk_folder_path and chunk_folder_path == filter_folder_path:
                    # Direct folder_path match
                    logger.info(f"Including chunk {chunk.chunk_id} - direct folder_path match: {chunk_folder_path}")
                else:
                    # Try matching by project_id or project name
                    chunk_project_id = getattr(chunk, 'project_id', None)
                    if chunk_project_id:
                        # If we have a target project_id from synthetic path, match against it
                        if target_project_id and chunk_project_id == target_project_id:
                            logger.info(f"Including chunk {chunk.chunk_id} - project_id match: {chunk_project_id}")
                        else:
                            # Otherwise try matching by project name
                            try:
                                project_info = self.sqlite_ops.get_project_by_id(chunk_project_id)
                                if project_info:
                                    project_name = project_info.project_name
                                    if project_name == target_project_name:
                                        logger.info(f"Including chunk {chunk.chunk_id} - project name match: {project_name}")
                                    else:
                                        logger.info(f"Filtering out chunk {chunk.chunk_id} - project name mismatch for folder filtering: {project_name} != {target_project_name}")
                                        continue
                                else:
                                    logger.info(f"Filtering out chunk {chunk.chunk_id} - no project found for project_id: {chunk_project_id}")
                                    continue
                            except Exception as e:
                                logger.error(f"Error getting project info for folder filtering: {e}")
                                continue
                    else:
                        logger.info(f"Filtering out chunk {chunk.chunk_id} - no project_id or folder_path for folder filtering")
                        continue
            
            logger.info(f"Including chunk {chunk.chunk_id} in results")
            filtered_results.append(result)
        
        logger.info(f"Total search results after filtering: {len(filtered_results)}")
        return filtered_results
    
    def _reconstruct_chunk_context(self, chunk) -> Dict:
        """Reconstruct complete context around a chunk"""
        try:
            context = {
                'document_title': getattr(chunk, 'document_title', ''),
                'document_date': chunk.date.isoformat() if chunk.date else '',
                'chunk_position': f"{chunk.chunk_index + 1}",
                'document_summary': getattr(chunk, 'content_summary', ''),
                'main_topics': getattr(chunk, 'main_topics', ''),
                'participants': getattr(chunk, 'participants', ''),
                'related_chunks': []
            }
            
            # Get meeting context if available
            if hasattr(chunk, 'document_id'):
                meeting_context = self._get_meeting_context(chunk.document_id)
                context.update(meeting_context)
            
            return context
            
        except Exception as e:
            logger.error(f"Error reconstructing chunk context: {e}")
            return {}
    
    def _get_meeting_context(self, document_id: str) -> Dict:
        """Get meeting-level context for a document"""
        try:
            # This would typically fetch additional meeting metadata
            # For now, return basic context
            return {
                'meeting_type': 'regular',
                'importance_level': 'medium'
            }
        except Exception as e:
            logger.error(f"Error getting meeting context: {e}")
            return {}
    
    # SQLite Operations Pass-through
    def get_documents_by_timeframe(self, timeframe: str, user_id: str = None):
        """Get documents filtered by intelligent timeframe calculation"""
        return self.sqlite_ops.get_documents_by_timeframe(timeframe, user_id)
    
    def keyword_search_chunks(self, keywords: List[str], limit: int = 50) -> List[str]:
        """Perform keyword search on chunk content"""
        return self.sqlite_ops.keyword_search_chunks(keywords, limit)
    
    def get_all_documents(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get all documents with metadata for document selection"""
        return self.sqlite_ops.get_all_documents(user_id)
    
    def get_user_documents_by_scope(self, user_id: str, project_id: str = None, 
                                  meeting_id: Union[str, List[str]] = None) -> List[str]:
        """Get document IDs for a user filtered by project or meeting(s)"""
        # This method would need to be implemented in SQLiteOperations
        # For now, return empty list
        return []
    
    def keyword_search_chunks_by_user(self, keywords: List[str], user_id: str, 
                                    project_id: str = None, meeting_id: Union[str, List[str]] = None, 
                                    limit: int = 50) -> List[str]:
        """Perform keyword search on chunk content filtered by user/project/meeting"""
        # This method would need to be implemented in SQLiteOperations
        # For now, use basic keyword search
        return self.sqlite_ops.keyword_search_chunks(keywords, limit)
    
    # User Management
    def create_user(self, username: str, email: str, full_name: str, password_hash: str) -> str:
        """Create a new user"""
        return self.sqlite_ops.create_user(username, email, full_name, password_hash)
    
    def get_user_by_username(self, username: str):
        """Get user by username"""
        return self.sqlite_ops.get_user_by_username(username)
    
    def get_user_by_id(self, user_id: str):
        """Get user by user_id"""
        return self.sqlite_ops.get_user_by_id(user_id)
    
    def update_user_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        self.sqlite_ops.update_user_last_login(user_id)
    
    # Project Management
    def create_project(self, user_id: str, project_name: str, description: str = "") -> str:
        """Create a new project for a user"""
        return self.sqlite_ops.create_project(user_id, project_name, description)
    
    def get_user_projects(self, user_id: str):
        """Get all projects for a user"""
        return self.sqlite_ops.get_user_projects(user_id)
    
    # Meeting Management
    def create_meeting(self, user_id: str, project_id: str, meeting_name: str, meeting_date: datetime) -> str:
        """Create a new meeting"""
        return self.sqlite_ops.create_meeting(user_id, project_id, meeting_name, meeting_date)
    
    def get_user_meetings(self, user_id: str, project_id: str = None):
        """Get meetings for a user, optionally filtered by project"""
        return self.sqlite_ops.get_user_meetings(user_id, project_id)
    
    # Session Management
    def create_session(self, user_id: str, session_id: str, expires_at: datetime) -> bool:
        """Create a new user session"""
        return self.sqlite_ops.create_session(user_id, session_id, expires_at)
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate a session and return user_id if valid"""
        return self.sqlite_ops.validate_session(session_id)
    
    def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a session"""
        return self.sqlite_ops.deactivate_session(session_id)
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions"""
        return self.sqlite_ops.cleanup_expired_sessions()
    
    def extend_session(self, session_id: str, new_expires_at: datetime) -> bool:
        """Extend a session's expiry time"""
        try:
            # This method needs to be implemented in SQLiteOperations
            # For now, return True as a placeholder
            return True
        except Exception as e:
            logger.error(f"Error extending session: {e}")
            return False
    
    # File Management
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        return self.sqlite_ops.calculate_file_hash(file_path)
    
    def is_file_duplicate(self, file_hash: str, filename: str, user_id: str) -> Optional[Dict]:
        """Check if file is a duplicate based on hash and return original file info"""
        return self.sqlite_ops.is_file_duplicate(file_hash, filename, user_id)
    
    def store_file_hash(self, file_hash: str, filename: str, original_filename: str, 
                       file_size: int, user_id: str, project_id: str = None, 
                       meeting_id: str = None, document_id: str = None) -> str:
        """Store file hash information for deduplication"""
        return self.sqlite_ops.store_file_hash(
            file_hash, filename, original_filename, file_size, 
            user_id, project_id, meeting_id, document_id
        )
    
    # Job Management
    def create_upload_job(self, user_id: str, total_files: int, project_id: str = None, 
                         meeting_id: str = None) -> str:
        """Create a new upload job for tracking batch processing"""
        return self.sqlite_ops.create_upload_job(user_id, total_files, project_id, meeting_id)
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status and progress"""
        return self.sqlite_ops.get_job_status(job_id)
    
    def update_job_status(self, job_id: str, status: str, processed_files: int = None, 
                         failed_files: int = None, error_message: str = None):
        """Update job status and progress"""
        self.sqlite_ops.update_job_status(job_id, status, processed_files, failed_files, error_message)
    
    def create_file_processing_status(self, job_id: str, filename: str, file_size: int, 
                                    file_hash: str) -> str:
        """Create file processing status entry"""
        # This method would need to be implemented in SQLiteOperations
        # For now, return a UUID as placeholder
        import uuid
        return str(uuid.uuid4())
    
    def update_file_processing_status(self, status_id: str, status: str, 
                                    error_message: str = None, document_id: str = None, 
                                    chunks_created: int = None):
        """Update file processing status"""
        # This method would need to be implemented in SQLiteOperations
        # For now, pass silently
        pass
    
    # Vector Operations Pass-through
    def save_index(self):
        """Save FAISS index to disk"""
        self.vector_ops.save_index()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index"""
        return self.vector_ops.get_index_stats()
    
    def clear_index(self):
        """Clear the FAISS index and metadata"""
        self.vector_ops.clear_index()
    
    # Combined Statistics
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            sqlite_stats = self.sqlite_ops.get_database_stats()
            vector_stats = self.vector_ops.get_index_stats()
            
            combined_stats = {
                'database': sqlite_stats,
                'vector_index': vector_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    # Utility Methods
    def store_document_metadata(self, filename: str, content: str, user_id: str, 
                              project_id: str = None, meeting_id: str = None) -> str:
        """Store document metadata and return document_id"""
        return self.sqlite_ops.store_document_metadata(filename, content, user_id, project_id, meeting_id)
    
    def get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get metadata for a specific document"""
        # This method would need to be implemented in SQLiteOperations
        # For now, return empty dict
        return {}
    
    def get_project_documents(self, project_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a specific project"""
        # This method would need to be implemented in SQLiteOperations
        # For now, return empty list
        return []
    
    # Backward Compatibility Properties
    @property
    def index(self):
        """Provide access to the FAISS index for backward compatibility"""
        return self.vector_ops.index
    
    @property
    def chunk_metadata(self):
        """Provide access to chunk metadata for backward compatibility"""
        return self.vector_ops.chunk_metadata
    
    @chunk_metadata.setter
    def chunk_metadata(self, value):
        """Allow setting chunk metadata for backward compatibility"""
        self.vector_ops.chunk_metadata = value
    
    def _rebuild_chunk_metadata(self):
        """Rebuild chunk metadata for backward compatibility"""
        self.vector_ops.rebuild_chunk_metadata(self.db_path)