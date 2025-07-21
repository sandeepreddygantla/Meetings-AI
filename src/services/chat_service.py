"""
Chat service for Meetings AI application.
Handles AI-powered chat interactions and query processing.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat and AI query operations."""
    
    def __init__(self, db_manager: DatabaseManager, processor=None):
        """
        Initialize chat service.
        
        Args:
            db_manager: Database manager instance
            processor: Document processor instance (optional for backwards compatibility)
        """
        self.db_manager = db_manager
        self.processor = processor  # For backwards compatibility with existing processor methods
    
    def process_chat_query(
        self,
        message: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        project_ids: Optional[List[str]] = None,
        meeting_ids: Optional[List[str]] = None,
        date_filters: Optional[Dict[str, Any]] = None,
        folder_path: Optional[str] = None
    ) -> Tuple[str, List[str], str]:
        """
        Process a chat query and generate AI response.
        
        Args:
            message: User's chat message
            user_id: ID of the user
            document_ids: Optional list of specific document IDs to search
            project_id: Optional project ID filter
            project_ids: Optional list of project IDs to filter
            meeting_ids: Optional list of meeting IDs to filter
            date_filters: Optional date filters
            folder_path: Optional folder path filter
            
        Returns:
            Tuple of (response, follow_up_questions, timestamp)
        """
        try:
            # Check if documents are available
            try:
                vector_size = self.db_manager.get_index_stats().get('total_vectors', 0)
            except Exception as e:
                logger.error(f"Error checking vector database: {e}")
                vector_size = 0
            
            if vector_size == 0:
                response = "I don't have any documents to analyze yet. Please upload some meeting documents first! 📁"
                follow_up_questions = []
            else:
                # Process query using existing processor logic
                if self.processor:
                    try:
                        # Combine project filters
                        combined_project_ids = []
                        if project_id:
                            combined_project_ids.append(project_id)
                        if project_ids:
                            combined_project_ids.extend(project_ids)
                        final_project_id = combined_project_ids[0] if combined_project_ids else None
                        
                        # Debug logging
                        logger.info(f"Chat query filters - project_id: {project_id}, project_ids: {project_ids}, final_project_id: {final_project_id}")
                        logger.info(f"Other filters - document_ids: {document_ids}, meeting_ids: {meeting_ids}, folder_path: {folder_path}")
                        
                        # Detect if this is a summary query to use enhanced context
                        is_summary_query = self.processor.detect_summary_query(message)
                        context_limit = 100 if is_summary_query else 50
                        
                        response, context = self.processor.answer_query_with_intelligence(
                            message, 
                            user_id=user_id, 
                            document_ids=document_ids, 
                            project_id=final_project_id,
                            meeting_ids=meeting_ids,
                            date_filters=date_filters,
                            folder_path=folder_path,
                            context_limit=context_limit, 
                            include_context=True
                        )
                        
                        logger.info(f"Chat service received response from processor. Length: {len(response)} characters")
                        logger.info(f"Response preview: {response[:100]}..." if response else "Response is empty!")
                        
                        # Generate follow-up questions
                        try:
                            follow_up_questions = self.processor.generate_follow_up_questions(message, response, context)
                        except Exception as follow_up_error:
                            logger.error(f"Error generating follow-up questions: {follow_up_error}")
                            follow_up_questions = []
                            
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        response = f"I encountered an error while processing your question: {str(e)}"
                        follow_up_questions = []
                else:
                    # Fallback response if processor not available
                    response = "Chat processing is temporarily unavailable. Please try again later."
                    follow_up_questions = []
            
            timestamp = datetime.now().isoformat()
            return response, follow_up_questions, timestamp
            
        except Exception as e:
            logger.error(f"Chat query processing error: {e}")
            return f"An error occurred while processing your query: {str(e)}", [], datetime.now().isoformat()
    
    def get_chat_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get chat-related statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        try:
            # If processor is available, use its comprehensive statistics
            if self.processor and hasattr(self.processor, 'get_meeting_statistics'):
                try:
                    processor_stats = self.processor.get_meeting_statistics()
                    if "error" not in processor_stats:
                        logger.info(f"Retrieved processor stats: {processor_stats}")
                        return processor_stats
                    else:
                        logger.warning(f"Processor stats error: {processor_stats.get('error')}")
                except Exception as e:
                    logger.error(f"Error getting processor statistics: {e}")
            
            # Fallback to database manager comprehensive statistics
            try:
                comprehensive_stats = self.db_manager.get_statistics()
                logger.info(f"Database manager comprehensive stats: {comprehensive_stats}")
                
                if comprehensive_stats and 'vector_index' in comprehensive_stats:
                    vector_info = comprehensive_stats['vector_index']
                    database_info = comprehensive_stats.get('database', {})
                    
                    stats = {
                        # Original format compatibility
                        'total_meetings': len(self.db_manager.get_all_documents(user_id)),
                        'total_chunks': vector_info.get('total_vectors', 0),
                        'vector_index_size': vector_info.get('total_vectors', 0),
                        'average_chunk_length': database_info.get('avg_chunk_length', 0),
                        'earliest_meeting': database_info.get('earliest_date'),
                        'latest_meeting': database_info.get('latest_date'),
                        
                        # Additional stats
                        'document_count': len(self.db_manager.get_all_documents(user_id)),
                        'vector_count': vector_info.get('total_vectors', 0),
                        'index_dimension': vector_info.get('dimension', 0),
                        'index_type': vector_info.get('index_type'),
                        'metadata_entries': vector_info.get('metadata_entries', 0),
                        'project_count': len(self.db_manager.get_user_projects(user_id)),
                        'meeting_count': len(self.db_manager.get_user_meetings(user_id))
                    }
                else:
                    raise Exception("No comprehensive stats available")
                    
            except Exception as e:
                logger.error(f"Error getting comprehensive stats, using fallback: {e}")
                
                # Basic fallback statistics
                stats = {
                    'total_meetings': 0,
                    'total_chunks': 0,
                    'vector_index_size': 0,
                    'average_chunk_length': 0,
                    'earliest_meeting': None,
                    'latest_meeting': None,
                    'document_count': 0,
                    'vector_count': 0,
                    'project_count': 0,
                    'meeting_count': 0
                }
                
                # Try individual calls
                try:
                    user_documents = self.db_manager.get_all_documents(user_id)
                    stats['total_meetings'] = len(user_documents)
                    stats['document_count'] = len(user_documents)
                except Exception:
                    pass
                    
                try:
                    vector_stats = self.db_manager.get_index_stats()
                    stats['total_chunks'] = vector_stats.get('total_vectors', 0)
                    stats['vector_index_size'] = vector_stats.get('total_vectors', 0)
                    stats['vector_count'] = vector_stats.get('total_vectors', 0)
                    stats['index_dimension'] = vector_stats.get('dimension', 0)
                    stats['metadata_entries'] = vector_stats.get('metadata_entries', 0)
                except Exception:
                    pass
                    
                try:
                    projects = self.db_manager.get_user_projects(user_id)
                    stats['project_count'] = len(projects)
                    
                    meetings = self.db_manager.get_user_meetings(user_id)
                    stats['meeting_count'] = len(meetings)
                except Exception:
                    pass
            
            logger.info(f"Fallback stats generated: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting chat statistics: {e}")
            return {'error': str(e)}
    
    def validate_chat_filters(
        self,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        meeting_ids: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Validate that the user has access to the specified filters.
        
        Args:
            user_id: User ID
            document_ids: Document IDs to validate
            project_id: Project ID to validate
            meeting_ids: Meeting IDs to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate project access
            if project_id:
                user_projects = self.db_manager.get_user_projects(user_id)
                project_exists = any(p.project_id == project_id for p in user_projects)
                if not project_exists:
                    return False, 'Invalid project selection'
            
            # Validate meeting access
            if meeting_ids:
                user_meetings = self.db_manager.get_user_meetings(user_id, project_id)
                for meeting_id in meeting_ids:
                    meeting_exists = any(m.meeting_id == meeting_id for m in user_meetings)
                    if not meeting_exists:
                        return False, f'Invalid meeting selection: {meeting_id}'
            
            # Validate document access (if needed)
            if document_ids:
                user_documents = self.db_manager.get_all_documents(user_id)
                user_doc_ids = [doc['document_id'] for doc in user_documents]
                for doc_id in document_ids:
                    if doc_id not in user_doc_ids:
                        return False, f'Invalid document selection: {doc_id}'
            
            return True, 'Valid'
            
        except Exception as e:
            logger.error(f"Error validating chat filters: {e}")
            return False, f'Validation error: {str(e)}'
    
    def get_available_filters(self, user_id: str) -> Dict[str, Any]:
        """
        Get available filters for chat queries.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of available filters
        """
        try:
            filters = {}
            
            # Get projects
            try:
                projects = self.db_manager.get_user_projects(user_id)
                filters['projects'] = [
                    {
                        'project_id': p.project_id,
                        'project_name': p.project_name,
                        'description': p.description,
                        'created_at': p.created_at.isoformat()
                    }
                    for p in projects
                ]
            except Exception as e:
                logger.error(f"Error getting projects: {e}")
                filters['projects'] = []
            
            # Get meetings
            try:
                meetings = self.db_manager.get_user_meetings(user_id)
                filters['meetings'] = [
                    {
                        'meeting_id': m.meeting_id,
                        'meeting_name': m.meeting_name,
                        'meeting_date': m.meeting_date.isoformat() if m.meeting_date else None,
                        'project_id': m.project_id,
                        'created_at': m.created_at.isoformat()
                    }
                    for m in meetings
                ]
            except Exception as e:
                logger.error(f"Error getting meetings: {e}")
                filters['meetings'] = []
            
            # Get documents
            try:
                documents = self.db_manager.get_all_documents(user_id)
                filters['documents'] = documents
            except Exception as e:
                logger.error(f"Error getting documents: {e}")
                filters['documents'] = []
            
            return filters
            
        except Exception as e:
            logger.error(f"Error getting available filters: {e}")
            return {'error': str(e)}