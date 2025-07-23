"""
Chat service for Meetings AI application.
Handles AI-powered chat interactions and query processing with enhanced context management.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.database.manager import DatabaseManager
from src.ai.context_manager import EnhancedContextManager, QueryContext
from src.ai.enhanced_prompts import EnhancedPromptManager

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat and AI query operations."""
    
    def __init__(self, db_manager: DatabaseManager, processor=None):
        """
        Initialize chat service with enhanced context management.
        
        Args:
            db_manager: Database manager instance
            processor: Document processor instance (optional for backwards compatibility)
        """
        self.db_manager = db_manager
        self.processor = processor  # For backwards compatibility with existing processor methods
        
        # Initialize enhanced components
        self.enhanced_context_manager = EnhancedContextManager(db_manager, processor)
        self.prompt_manager = EnhancedPromptManager()
        
        # Feature flags for gradual rollout
        self.use_enhanced_processing = True  # Enable enhanced processing by default
        self.enhanced_summary_threshold = 10  # Use enhanced for queries with 10+ potential documents
    
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
            # ===== DEBUG LOGGING: CHAT SERVICE ENTRY =====
            logger.info("[SERVICE] ChatService.process_chat_query() - ENTRY POINT")
            logger.info(f"[PARAMS] Parameters received:")
            logger.info(f"   - message: '{message}'")
            logger.info(f"   - user_id: {user_id}")
            logger.info(f"   - document_ids: {document_ids}")
            logger.info(f"   - project_id: {project_id}")
            logger.info(f"   - project_ids: {project_ids}")
            logger.info(f"   - meeting_ids: {meeting_ids}")
            logger.info(f"   - date_filters: {date_filters}")
            logger.info(f"   - folder_path: {folder_path}")
            
            # Check if documents are available
            logger.info("[STEP1] Checking vector database status...")
            try:
                index_stats = self.db_manager.get_index_stats()
                vector_size = index_stats.get('total_vectors', 0)
                logger.info(f"[STATS] Vector database stats: {index_stats}")
            except Exception as e:
                logger.error(f"[ERROR] Error checking vector database: {e}")
                vector_size = 0
            
            logger.info(f"[VECTORS] Total vectors available: {vector_size}")
            
            if vector_size == 0:
                logger.warning("[WARNING] NO VECTORS FOUND - Returning 'no documents' response")
                response = "I don't have any documents to analyze yet. Please upload some meeting documents first!"
                follow_up_questions = []
            else:
                logger.info("[OK] Vectors available - proceeding with query processing")
                
                # Determine processing strategy
                should_use_enhanced = self._should_use_enhanced_processing(
                    message, user_id, document_ids, project_id, project_ids, meeting_ids
                )
                
                if should_use_enhanced and self.use_enhanced_processing:
                    logger.info("[ENHANCED] Using enhanced context processing")
                    response, follow_up_questions = self._process_with_enhanced_context(
                        message, user_id, document_ids, project_id, project_ids, 
                        meeting_ids, date_filters, folder_path
                    )
                else:
                    logger.info("[LEGACY] Using legacy processor for query processing")
                    response, follow_up_questions = self._process_with_legacy_processor(
                        message, user_id, document_ids, project_id, project_ids,
                        meeting_ids, date_filters, folder_path
                    )
            
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
                    
                    # Get active document and chunk counts (excludes soft-deleted)
                    active_documents = len(self.db_manager.get_all_documents(user_id))
                    active_chunks = database_info.get('chunks_count', 0)  # Active chunks only from database
                    
                    stats = {
                        # Original format compatibility - use ACTIVE counts for user-facing stats
                        'total_meetings': active_documents,
                        'total_chunks': active_chunks,  # Use database count (active only) instead of FAISS count (includes soft-deleted)
                        'vector_index_size': active_chunks,  # Show active chunks for user display
                        'average_chunk_length': database_info.get('avg_chunk_length', 0),
                        'earliest_meeting': database_info.get('earliest_date'),
                        'latest_meeting': database_info.get('latest_date'),
                        
                        # Additional stats
                        'document_count': active_documents,
                        'vector_count': vector_info.get('total_vectors', 0),  # Keep actual FAISS count for technical monitoring
                        'index_dimension': vector_info.get('dimension', 0),
                        'index_type': vector_info.get('index_type'),
                        'metadata_entries': vector_info.get('metadata_entries', 0),
                        'project_count': len(self.db_manager.get_user_projects(user_id)),
                        'meeting_count': len(self.db_manager.get_user_meetings(user_id)),
                        
                        # Soft deletion monitoring stats (for admin/debugging)
                        'soft_deleted_documents': database_info.get('documents_soft_deleted', 0),
                        'soft_deleted_chunks': database_info.get('chunks_soft_deleted', 0)
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
    
    def _should_use_enhanced_processing(
        self,
        message: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        project_ids: Optional[List[str]] = None,
        meeting_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Determine if enhanced processing should be used based on query characteristics.
        
        Args:
            message: User query message
            user_id: User ID
            document_ids: Document filter
            project_id: Project filter
            project_ids: Multiple project filters
            meeting_ids: Meeting filters
            
        Returns:
            True if enhanced processing should be used
        """
        try:
            # Check for summary/comprehensive query indicators
            message_lower = message.lower()
            summary_indicators = [
                'summary', 'summarize', 'overview', 'comprehensive', 
                'all meetings', 'all documents', 'everything',
                'across all', 'complete picture', 'full scope'
            ]
            
            is_summary_query = any(indicator in message_lower for indicator in summary_indicators)
            
            # Check if no specific filters are applied (user wants all data)
            no_specific_filters = not any([document_ids, project_id, project_ids, meeting_ids])
            
            # Check document count to determine if enhanced processing is beneficial
            try:
                user_documents = self.db_manager.get_all_documents(user_id)
                document_count = len(user_documents)
            except:
                document_count = 0
            
            # Use enhanced processing for:
            # 1. Summary queries with many documents
            # 2. Queries without specific filters and many documents
            # 3. Comprehensive analysis requests
            should_use_enhanced = (
                (is_summary_query and document_count >= self.enhanced_summary_threshold) or
                (no_specific_filters and document_count >= self.enhanced_summary_threshold) or
                ('comprehensive' in message_lower and document_count > 5)
            )
            
            logger.info(f"[ENHANCED_DECISION] Enhanced processing decision:")
            logger.info(f"  - Is summary query: {is_summary_query}")
            logger.info(f"  - No specific filters: {no_specific_filters}")
            logger.info(f"  - Document count: {document_count}")
            logger.info(f"  - Threshold: {self.enhanced_summary_threshold}")
            logger.info(f"  - Decision: {should_use_enhanced}")
            
            return should_use_enhanced
            
        except Exception as e:
            logger.error(f"[ERROR] Error determining processing strategy: {e}")
            return False  # Default to legacy processing on error
    
    def _process_with_enhanced_context(
        self,
        message: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        project_ids: Optional[List[str]] = None,
        meeting_ids: Optional[List[str]] = None,
        date_filters: Optional[Dict[str, Any]] = None,
        folder_path: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Process query using enhanced context management.
        
        Returns:
            Tuple of (response, follow_up_questions)
        """
        try:
            # Combine project filters
            combined_project_ids = []
            if project_id:
                combined_project_ids.append(project_id)
            if project_ids:
                combined_project_ids.extend(project_ids)
            final_project_id = combined_project_ids[0] if combined_project_ids else None
            
            # Detect query characteristics
            is_summary_query = (
                self.processor.detect_summary_query(message) if self.processor 
                else self._detect_summary_query_fallback(message)
            )
            
            # Create query context
            query_context = QueryContext(
                query=message,
                user_id=user_id,
                document_ids=document_ids,
                project_id=final_project_id,
                meeting_ids=meeting_ids,
                date_filters=date_filters,
                folder_path=folder_path,
                is_summary_query=is_summary_query,
                is_comprehensive=self._is_comprehensive_query(message),
                context_limit=200 if is_summary_query else 100  # Enhanced context limits
            )
            
            # Process with enhanced context manager
            response, follow_up_questions, _ = self.enhanced_context_manager.process_enhanced_query(query_context)
            
            logger.info(f"[ENHANCED] Enhanced processing completed:")
            logger.info(f"  - Response length: {len(response)} characters")
            logger.info(f"  - Follow-up questions: {len(follow_up_questions)}")
            
            return response, follow_up_questions
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced processing failed: {e}")
            # Fallback to legacy processing
            return self._process_with_legacy_processor(
                message, user_id, document_ids, project_id, project_ids,
                meeting_ids, date_filters, folder_path
            )
    
    def _process_with_legacy_processor(
        self,
        message: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        project_ids: Optional[List[str]] = None,
        meeting_ids: Optional[List[str]] = None,
        date_filters: Optional[Dict[str, Any]] = None,
        folder_path: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Process query using legacy processor.
        
        Returns:
            Tuple of (response, follow_up_questions)
        """
        try:
            if not self.processor:
                return "Chat processing is temporarily unavailable. Please try again later.", []
            
            # Combine project filters
            combined_project_ids = []
            if project_id:
                combined_project_ids.append(project_id)
            if project_ids:
                combined_project_ids.extend(project_ids)
            final_project_id = combined_project_ids[0] if combined_project_ids else None
            
            logger.info(f"[LEGACY_FILTERS] Filter processing results:")
            logger.info(f"   - Original project_id: {project_id}")
            logger.info(f"   - Original project_ids: {project_ids}")
            logger.info(f"   - Final project_id: {final_project_id}")
            logger.info(f"   - document_ids: {document_ids}")
            logger.info(f"   - meeting_ids: {meeting_ids}")
            logger.info(f"   - folder_path: {folder_path}")
            
            # Detect if this is a summary query to use enhanced context
            is_summary_query = self.processor.detect_summary_query(message)
            context_limit = 100 if is_summary_query else 50
            logger.info(f"[LEGACY_QUERY] Query type analysis:")
            logger.info(f"   - Is summary query: {is_summary_query}")
            logger.info(f"   - Context limit: {context_limit}")
            
            # THE MAIN PROCESSING CALL
            logger.info("[LEGACY_PROCESSING] Calling processor.answer_query_with_intelligence()...")
            
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
            
            # ===== DEBUG LOGGING: PROCESSOR RESPONSE =====
            logger.info("[LEGACY_RESULT] PROCESSOR RESPONSE RECEIVED")
            logger.info(f"[RESPONSE] Response length: {len(response)} characters")
            logger.info(f"[CONTEXT] Context chunks received: {len(context) if context else 0}")
            if response:
                logger.info(f"[PREVIEW] Response preview (first 200 chars): '{response[:200]}...'")
            else:
                logger.error("[CRITICAL] Processor returned empty response!")
            
            # Check for problematic responses
            if "no relevant information" in response.lower():
                logger.error("[ALERT] DETECTED: Processor returned 'no relevant information' - search pipeline failed!")
            elif "couldn't find" in response.lower():
                logger.error("[ALERT] DETECTED: Processor returned 'couldn't find' - search pipeline failed!")
            else:
                logger.info("[SUCCESS] Response appears to contain relevant information")
            
            # Generate follow-up questions
            logger.info("[LEGACY_FOLLOWUP] Generating follow-up questions...")
            try:
                follow_up_questions = self.processor.generate_follow_up_questions(message, response, context)
                logger.info(f"[FOLLOWUP] Generated {len(follow_up_questions)} follow-up questions")
            except Exception as follow_up_error:
                logger.error(f"[ERROR] Error generating follow-up questions: {follow_up_error}")
                follow_up_questions = []
            
            return response, follow_up_questions
            
        except Exception as e:
            logger.error(f"[ERROR] Legacy processing failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}", []
    
    def _detect_summary_query_fallback(self, message: str) -> bool:
        """Fallback summary query detection if processor is not available."""
        summary_keywords = [
            'summarize', 'summary', 'summaries', 'overview', 'brief', 
            'recap', 'highlights', 'key points', 'main points',
            'all meetings', 'all documents', 'overall', 'across all',
            'consolidate', 'aggregate', 'compile', 'comprehensive'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in summary_keywords)
    
    def _is_comprehensive_query(self, message: str) -> bool:
        """Detect if query is asking for comprehensive analysis."""
        comprehensive_indicators = [
            'comprehensive', 'complete picture', 'full scope', 'everything',
            'all meetings', 'all documents', 'entire', 'whole', 'total'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in comprehensive_indicators)