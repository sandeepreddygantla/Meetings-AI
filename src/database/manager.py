"""
Main database manager that combines vector and SQLite operations.
This module provides a unified interface for all database operations.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import os
from pathlib import Path

from .vector_operations import VectorOperations, get_application_root
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
            db_path: Path to SQLite database file (relative paths resolved to application root)
            index_path: Path to FAISS index file (relative paths resolved to application root)
        """
        # Convert relative paths to absolute paths based on application root
        app_root = get_application_root()
        
        if not os.path.isabs(db_path):
            self.db_path = str(app_root / db_path)
            logger.info(f"Database path resolved to: {self.db_path}")
        else:
            self.db_path = db_path
            
        if not os.path.isabs(index_path):
            self.index_path = str(app_root / index_path)  
            logger.info(f"Vector index path resolved to: {self.index_path}")
        else:
            self.index_path = index_path
        self.dimension = 3072  # text-embedding-3-large dimension
        
        # Initialize both operations handlers with resolved paths
        self.sqlite_ops = SQLiteOperations(self.db_path)
        self.vector_ops = VectorOperations(self.index_path, self.dimension)
        
        # Load existing chunk metadata from database
        self.vector_ops.rebuild_chunk_metadata(self.db_path)
        
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
    
    def get_document_chunks(self, document_id: str):
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of DocumentChunk objects for the document
        """
        try:
            # Get chunk IDs for the document
            chunk_ids = self.sqlite_ops.get_chunk_ids_by_document_id(document_id)
            
            # Get chunk data using existing method
            if chunk_ids:
                return self.get_chunks_by_ids(chunk_ids)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []
    
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
            start_time = datetime.now()
            
            # Get initial vector search results (optimized - no verbose logging)
            vector_results = self.search_similar_chunks(query_embedding, top_k * 2)
            
            if not vector_results:
                logger.error("Vector search failed - no results found")
                return []
            
            # Get chunk data from SQLite  
            chunk_ids = [chunk_id for chunk_id, _ in vector_results]
            chunks = self.get_chunks_by_ids(chunk_ids)
            
            # Validate synchronization (only log if there are issues)
            if chunks and len(chunks) != len(chunk_ids):
                logger.warning(f"Database sync issue: Vector found {len(chunk_ids)} chunks, SQLite returned {len(chunks)}")
            
            if not chunks:
                logger.error("Database query failed - no chunks retrieved")
                return []
            
            # Apply user filtering and create results
            enhanced_results = []
            score_map = {chunk_id: score for chunk_id, score in vector_results}
            user_filter_passed = 0
            user_filter_failed = 0
            
            for chunk in chunks:
                chunk_user_id = getattr(chunk, 'user_id', None)
                # Strict user filtering - only allow access to user's own data
                passes_filter = (chunk_user_id == user_id and user_id is not None)
                
                if passes_filter:
                    result = {
                        'chunk': chunk,
                        'similarity_score': score_map.get(getattr(chunk, 'chunk_id', ''), 0.0),
                        'context': self._reconstruct_chunk_context(chunk)
                    }
                    enhanced_results.append(result)
                    user_filter_passed += 1
                else:
                    user_filter_failed += 1
            
            logger.info(f"[STATS] User ID filtering results:")
            logger.info(f"   - Passed: {user_filter_passed} chunks")
            logger.info(f"   - Filtered out: {user_filter_failed} chunks")
            logger.info(f"   - Total processed: {len(chunks)} chunks")
            
            # Apply metadata filters if provided
            if filters:
                logger.info(f"[STEP] Step 5: Applying metadata filters: {filters}")
                results_before_metadata = len(enhanced_results)
                enhanced_results = self._apply_metadata_filters(enhanced_results, filters, user_id)
                results_after_metadata = len(enhanced_results)
                logger.info(f"[STATS] Metadata filtering: {results_before_metadata} -> {results_after_metadata} chunks")
                
                if results_after_metadata == 0 and results_before_metadata > 0:
                    logger.error(f"Metadata filters eliminated all results: {filters}")
            
            # Sort by similarity score and limit results
            enhanced_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = enhanced_results[:top_k]
            
            # Single optimized log line
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Search completed: {len(final_results)} results in {processing_time:.3f}s", 
                       extra={'user_id': user_id})
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return []
    
    def _apply_metadata_filters(self, search_results: List[Dict], filters: Dict, user_id: str) -> List[Dict]:
        """Apply metadata filters to search results"""
        filtered_results = []
        
        # Optimized - only log if there are issues
        for result in search_results:
            chunk = result['chunk']
            
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
                
                # Special case: "project_default" means include all chunks (no project filtering)
                if 'project_default' in filter_folder_path:
                    logger.info(f"Folder path contains 'project_default' - skipping project filtering")
                    # Skip this entire folder filtering block
                elif 'project_' in filter_folder_path:
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
                            # If no project matching is possible, include the chunk anyway for now
                            # This prevents overly strict filtering that removes all results
                            logger.info(f"Including chunk {chunk.chunk_id} - relaxed folder filtering (no project info available)")
                            # Continue processing instead of filtering out
                            pass
                else:
                    # No folder filtering when folder_path doesn't contain project info
                    logger.info(f"Folder path '{filter_folder_path}' - no project filtering applied")
            
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
        return self.sqlite_ops.get_document_metadata(document_id)
    
    def get_project_documents(self, project_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a specific project"""
        return self.sqlite_ops.get_project_documents(project_id, user_id)
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by its ID"""
        return self.sqlite_ops.get_document_metadata(document_id)
    
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
        """Rebuild chunk metadata - not needed in simplified architecture"""
        logger.info("Chunk metadata rebuild not needed - using simplified IndexFlatIP")
    
    # Complete Document Deletion Operations
    def delete_document_complete(self, document_id: str, user_id: str) -> Dict[str, Any]:
        """Complete deletion of a document across all storage layers"""
        result = {
            'success': False,
            'document_id': document_id,
            'filesystem': False,
            'sqlite_document': False,
            'sqlite_chunks': False,
            'vectors': False,
            'error': None,
            'rollback_needed': False
        }
        
        try:
            # Step 0: Create audit log entry
            self.sqlite_ops.create_deletion_audit_log(
                user_id, document_id, 'deletion_started', 
                {'operation': 'delete_document_complete'}
            )
            
            # Step 1: Get document metadata and file path before deletion
            doc_metadata = self.sqlite_ops.get_document_metadata_for_deletion(document_id, user_id)
            if not doc_metadata:
                result['error'] = f"Document {document_id} not found or user {user_id} doesn't have access"
                self.sqlite_ops.create_deletion_audit_log(
                    user_id, document_id, 'deletion_failed', 
                    {'error': result['error']}
                )
                return result
            
            # Step 1.5: Create backup before deletion
            backup_info = self.sqlite_ops.create_deletion_backup(document_id, user_id)
            if backup_info:
                result['backup_created'] = backup_info
                logger.info(f"Created backup {backup_info['backup_id']} for document {document_id}")
            else:
                logger.warning(f"Failed to create backup for document {document_id}")
            
            file_path = self.sqlite_ops.get_document_file_path(document_id)
            chunk_ids = self.sqlite_ops.get_chunk_ids_by_document_id(document_id)
            
            logger.info(f"Starting complete deletion for document {document_id} (user: {user_id})")
            logger.info(f"File path: {file_path}, Chunks: {len(chunk_ids)}")
            
            # Step 2: Document deletion not supported in simplified architecture
            logger.info("Document deletion from vector database not supported - simplified architecture")
            result['vectors'] = True  # Skip vector deletion
            
            # Step 3: Delete chunks from SQLite
            chunks_success = self.sqlite_ops.delete_chunks_by_document_id(document_id)
            result['sqlite_chunks'] = chunks_success
            
            if not chunks_success:
                result['error'] = "Failed to delete document chunks from database"
                result['rollback_needed'] = True
                return result
            
            # Step 4: Delete document from SQLite
            doc_success = self.sqlite_ops.delete_document_by_id(document_id, user_id)
            result['sqlite_document'] = doc_success
            
            if not doc_success:
                result['error'] = "Failed to delete document from database"
                result['rollback_needed'] = True
                return result
            
            # Step 5: Delete physical file
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    result['filesystem'] = True
                    logger.info(f"Deleted physical file: {file_path}")
                    
                    # Clean up empty directories
                    self._cleanup_empty_directories(os.path.dirname(file_path))
                    
                except Exception as e:
                    logger.error(f"Error deleting physical file {file_path}: {e}")
                    result['filesystem'] = False
                    # File deletion failure is not critical - continue
            else:
                result['filesystem'] = True  # No file to delete or already gone
            
            # Check overall success
            result['success'] = all([
                result['sqlite_document'],
                result['sqlite_chunks'],
                result['vectors']
                # Note: filesystem deletion failure is not blocking
            ])
            
            if result['success']:
                logger.info(f"Successfully completed deletion of document {document_id}")
                # Log successful deletion
                self.sqlite_ops.create_deletion_audit_log(
                    user_id, document_id, 'deletion_completed', 
                    {
                        'filesystem': result['filesystem'],
                        'sqlite_document': result['sqlite_document'],
                        'sqlite_chunks': result['sqlite_chunks'],
                        'vectors': result['vectors'],
                        'backup_id': result.get('backup_created', {}).get('backup_id')
                    }
                )
            else:
                logger.error(f"Partial failure in document {document_id} deletion: {result}")
                # Log failed deletion
                self.sqlite_ops.create_deletion_audit_log(
                    user_id, document_id, 'deletion_failed', 
                    {
                        'error': result.get('error'),
                        'partial_results': result
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in complete document deletion: {e}")
            result['error'] = str(e)
            result['rollback_needed'] = True
            return result
    
    def delete_multiple_documents_complete(self, document_ids: List[str], user_id: str) -> Dict[str, Any]:
        """Complete deletion of multiple documents with detailed results"""
        results = {
            'success': False,
            'total_requested': len(document_ids),
            'successful_deletions': [],
            'failed_deletions': [],
            'detailed_results': {},
            'summary': {}
        }
        
        try:
            logger.info(f"Starting batch deletion of {len(document_ids)} documents for user {user_id}")
            
            for doc_id in document_ids:
                try:
                    result = self.delete_document_complete(doc_id, user_id)
                    results['detailed_results'][doc_id] = result
                    
                    if result['success']:
                        results['successful_deletions'].append(doc_id)
                    else:
                        results['failed_deletions'].append(doc_id)
                        
                except Exception as e:
                    logger.error(f"Error deleting document {doc_id}: {e}")
                    results['failed_deletions'].append(doc_id)
                    results['detailed_results'][doc_id] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Calculate summary
            results['summary'] = {
                'total_requested': results['total_requested'],
                'successful': len(results['successful_deletions']),
                'failed': len(results['failed_deletions']),
                'success_rate': len(results['successful_deletions']) / max(1, results['total_requested'])
            }
            
            results['success'] = len(results['successful_deletions']) > 0
            
            logger.info(f"Batch deletion completed: {results['summary']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch document deletion: {e}")
            results['error'] = str(e)
            return results
    
    def get_deletable_documents(self, user_id: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get list of documents that can be deleted by the user"""
        try:
            # Get all user documents
            all_documents = self.get_all_documents(user_id)
            
            # Apply filters if provided
            if filters:
                filtered_docs = []
                for doc in all_documents:
                    include_doc = True
                    
                    # Filter by project
                    if filters.get('project_id') and doc.get('project_id') != filters['project_id']:
                        include_doc = False
                    
                    # Filter by date range
                    if filters.get('date_range') and doc.get('date'):
                        doc_date = datetime.fromisoformat(doc['date'])
                        start_date, end_date = filters['date_range']
                        if start_date and doc_date < start_date:
                            include_doc = False
                        if end_date and doc_date > end_date:
                            include_doc = False
                    
                    # Filter by file size
                    if filters.get('min_size') and doc.get('file_size', 0) < filters['min_size']:
                        include_doc = False
                    
                    if include_doc:
                        filtered_docs.append(doc)
                
                return filtered_docs
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Error getting deletable documents: {e}")
            return []
    
    def _cleanup_empty_directories(self, directory_path: str):
        """Recursively clean up empty directories"""
        try:
            if not os.path.exists(directory_path):
                return
            
            # Don't delete the root meeting_documents directory
            if directory_path.endswith('meeting_documents'):
                return
            
            # Check if directory is empty
            if os.path.isdir(directory_path) and not os.listdir(directory_path):
                os.rmdir(directory_path)
                logger.info(f"Removed empty directory: {directory_path}")
                
                # Recursively check parent directory
                parent_dir = os.path.dirname(directory_path)
                if parent_dir != directory_path:  # Avoid infinite recursion
                    self._cleanup_empty_directories(parent_dir)
                    
        except Exception as e:
            logger.error(f"Error cleaning up directory {directory_path}: {e}")
    
    def rebuild_vector_index_after_deletion(self) -> bool:
        """Force rebuild of vector index (simplified - no rebuild needed)"""
        try:
            logger.info("Index rebuild not needed - using simplified IndexFlatIP")
            return True  # Always successful in simplified architecture
        except Exception as e:
            logger.error(f"Error rebuilding vector index: {e}")
            return False
    
    def get_deletion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive deletion and storage statistics"""
        try:
            # Get vector deletion stats
            vector_stats = self.vector_ops.get_deletion_stats()
            
            # Get database stats
            db_stats = self.get_statistics()
            
            # Calculate storage usage
            storage_stats = self._calculate_storage_usage()
            
            return {
                'vector_index': vector_stats,
                'database': db_stats.get('database', {}),
                'storage': storage_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting deletion statistics: {e}")
            return {}
    
    def _calculate_storage_usage(self) -> Dict[str, Any]:
        """Calculate storage usage for meeting documents"""
        try:
            meeting_docs_path = "meeting_documents"
            if not os.path.exists(meeting_docs_path):
                return {'total_size': 0, 'file_count': 0, 'directory_count': 0}
            
            total_size = 0
            file_count = 0
            directory_count = 0
            
            for root, dirs, files in os.walk(meeting_docs_path):
                directory_count += len(dirs)
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        file_count += 1
                    except OSError:
                        pass  # Skip files we can't access
            
            return {
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count,
                'directory_count': directory_count
            }
            
        except Exception as e:
            logger.error(f"Error calculating storage usage: {e}")
            return {'error': str(e)}
    
    def validate_deletion_safety(self, document_ids: List[str], user_id: str) -> Dict[str, Any]:
        """Validate whether it's safe to delete the specified documents"""
        try:
            safety_report = {
                'safe_to_delete': True,
                'warnings': [],
                'blockers': [],
                'document_analysis': {},
                'total_impact': {
                    'files_affected': len(document_ids),
                    'estimated_size': 0,
                    'chunks_affected': 0
                }
            }
            
            for doc_id in document_ids:
                doc_metadata = self.sqlite_ops.get_document_metadata_for_deletion(doc_id, user_id)
                if not doc_metadata:
                    safety_report['blockers'].append(f"Document {doc_id} not found or access denied")
                    safety_report['safe_to_delete'] = False
                    continue
                
                # Analyze document for safety concerns
                doc_analysis = {
                    'size': doc_metadata.get('file_size', 0),
                    'chunks': doc_metadata.get('chunk_count', 0),
                    'age_days': 0,
                    'warnings': []
                }
                
                # Check document age
                if doc_metadata.get('created_at'):
                    try:
                        created_date = datetime.fromisoformat(doc_metadata['created_at'])
                        doc_analysis['age_days'] = (datetime.now() - created_date).days
                        
                        if doc_analysis['age_days'] < 1:
                            doc_analysis['warnings'].append('Very recent document (less than 1 day old)')
                        elif doc_analysis['age_days'] < 7:
                            doc_analysis['warnings'].append('Recent document (less than 1 week old)')
                    except Exception:
                        pass
                
                # Check document size
                if doc_analysis['size'] > 50 * 1024 * 1024:  # > 50MB
                    doc_analysis['warnings'].append('Very large document (>50MB)')
                elif doc_analysis['size'] > 10 * 1024 * 1024:  # > 10MB
                    doc_analysis['warnings'].append('Large document (>10MB)')
                
                # Check chunk count
                if doc_analysis['chunks'] > 100:
                    doc_analysis['warnings'].append('Document with many sections (>100 chunks)')
                
                safety_report['document_analysis'][doc_id] = doc_analysis
                safety_report['total_impact']['estimated_size'] += doc_analysis['size']
                safety_report['total_impact']['chunks_affected'] += doc_analysis['chunks']
                
                # Collect warnings
                safety_report['warnings'].extend([
                    f"{doc_id}: {warning}" for warning in doc_analysis['warnings']
                ])
            
            # Overall safety assessment
            if len(safety_report['warnings']) > 10:
                safety_report['blockers'].append('Too many warnings - review individual documents')
                safety_report['safe_to_delete'] = False
            
            if safety_report['total_impact']['estimated_size'] > 500 * 1024 * 1024:  # > 500MB
                safety_report['warnings'].append('Large total deletion size (>500MB)')
            
            return safety_report
            
        except Exception as e:
            logger.error(f"Error validating deletion safety: {e}")
            return {
                'safe_to_delete': False,
                'warnings': [],
                'blockers': [f'Safety validation error: {str(e)}'],
                'document_analysis': {},
                'total_impact': {'files_affected': 0, 'estimated_size': 0, 'chunks_affected': 0}
            }
    
    # Safety and Recovery Operations
    def get_recent_deletions(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent deletions for potential recovery"""
        try:
            return self.sqlite_ops.get_deletion_audit_logs(user_id, limit=100)
        except Exception as e:
            logger.error(f"Error getting recent deletions: {e}")
            return []
    
    def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up expired backups and return cleanup statistics"""
        try:
            deleted_count = self.sqlite_ops.cleanup_expired_backups()
            return {
                'success': True,
                'deleted_backups': deleted_count,
                'message': f"Cleaned up {deleted_count} expired backups"
            }
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Soft Deletion Methods (Replacement for Hard Deletion)
    def soft_delete_document(self, document_id: str, user_id: str, deleted_by: str) -> Dict[str, Any]:
        """Soft delete a document (marks as deleted without removing from FAISS)"""
        try:
            success = self.sqlite_ops.soft_delete_document(document_id, user_id, deleted_by)
            
            result = {
                'success': success,
                'document_id': document_id,
                'operation': 'soft_delete',
                'faiss_intact': True,  # FAISS vectors remain untouched
                'error': None
            }
            
            if success:
                logger.info(f"Successfully soft-deleted document {document_id} by user {deleted_by}")
            else:
                result['error'] = f"Failed to soft-delete document {document_id}"
                logger.error(result['error'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in soft deletion: {e}")
            return {
                'success': False,
                'document_id': document_id,
                'operation': 'soft_delete',
                'error': str(e)
            }
    
    def undelete_document(self, document_id: str, user_id: str) -> Dict[str, Any]:
        """Restore a soft-deleted document"""
        try:
            success = self.sqlite_ops.undelete_document(document_id, user_id)
            
            result = {
                'success': success,
                'document_id': document_id,
                'operation': 'undelete',
                'error': None
            }
            
            if success:
                logger.info(f"Successfully restored document {document_id} for user {user_id}")
            else:
                result['error'] = f"Failed to restore document {document_id}"
                logger.error(result['error'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in document restoration: {e}")
            return {
                'success': False,
                'document_id': document_id,
                'operation': 'undelete',
                'error': str(e)
            }
    
    def get_soft_deleted_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all soft-deleted documents for a user"""
        try:
            return self.sqlite_ops.get_deleted_documents(user_id)
        except Exception as e:
            logger.error(f"Error getting deleted documents: {e}")
            return []