"""
Chat service for Meetings AI application.
Handles AI-powered chat interactions and query processing with enhanced context management.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union

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
                user_doc_ids = [doc.document_id for doc in user_documents]
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
            # Use enhanced query intelligence for better detection
            is_summary_query = self.detect_enhanced_summary_query(message)
            is_comprehensive = self._is_comprehensive_query(message)
            is_project_summary = self.detect_project_summary_query(message)
            detected_timeframe = self.detect_timeframe_from_query(message)
            query_filters = self.analyze_query_for_filters(message)
            
            # Check if no specific filters are applied (user wants all data)
            no_specific_filters = not any([document_ids, project_id, project_ids, meeting_ids])
            
            # Check document count to determine if enhanced processing is beneficial
            try:
                user_documents = self.db_manager.get_all_documents(user_id)
                document_count = len(user_documents)
            except:
                document_count = 0
            
            # Use enhanced processing for:
            # 1. Summary queries with sufficient documents
            # 2. Comprehensive queries regardless of document count
            # 3. Project summary queries (NEW - from original intelligence)
            # 4. Date-based queries (NEW - from original intelligence)
            # 5. Speaker/decision/action queries (from original intelligence)
            # 6. Queries without specific filters and many documents
            should_use_enhanced = (
                # PRIORITY: Always use enhanced when specific document IDs are provided (for @file mentions)
                (document_ids is not None and len(document_ids) > 0) or
                (is_summary_query and document_count >= self.enhanced_summary_threshold) or
                is_comprehensive or
                is_project_summary or
                detected_timeframe is not None or
                bool(query_filters) or
                (no_specific_filters and document_count >= self.enhanced_summary_threshold) or
                ('comprehensive' in message.lower() and document_count > 5)
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
            
            # PRIORITY CHECK: If specific document_ids are provided, use document-specific processing
            # This ensures @file:filename mentions work correctly and don't get mixed with other documents
            if document_ids:
                logger.info(f"[ENHANCED] Specific document IDs provided: {document_ids}")
                logger.info("[ENHANCED] Using document-specific processing (bypassing general routing)")
                
                # Debug: Get document details for verification
                try:
                    for doc_id in document_ids:
                        doc_details = self.db_manager.get_document_by_id(doc_id)
                        if doc_details:
                            logger.info(f"[DEBUG] Document {doc_id}: filename='{doc_details.get('filename', 'N/A')}', date='{doc_details.get('date', 'N/A')}'")
                        else:
                            logger.warning(f"[DEBUG] Document {doc_id} not found in database")
                except Exception as e:
                    logger.error(f"[DEBUG] Error getting document details: {e}")
                
                # Create query context for specific documents
                query_context = QueryContext(
                    query=message,
                    user_id=user_id,
                    document_ids=document_ids,
                    project_id=final_project_id,
                    meeting_ids=meeting_ids,
                    date_filters=date_filters,
                    folder_path=folder_path,
                    is_summary_query=True,  # Treat as summary when specific docs are selected
                    is_comprehensive=False,  # Focus on specific documents only
                    context_limit=150  # Moderate context for specific documents
                )
                
                # Process with enhanced context manager
                response, follow_up_questions, _ = self.enhanced_context_manager.process_enhanced_query(query_context)
                
                logger.info(f"[ENHANCED] Document-specific processing completed:")
                logger.info(f"  - Response length: {len(response)} characters")
                logger.info(f"  - Follow-up questions: {len(follow_up_questions)}")
                
                return response, follow_up_questions
            
            # Detect query characteristics using enhanced intelligence
            is_summary_query = self.detect_enhanced_summary_query(message)
            is_project_summary = self.detect_project_summary_query(message)
            detected_timeframe = self.detect_timeframe_from_query(message)
            query_filters = self.analyze_query_for_filters(message)
            
            # Route to appropriate specialized processing
            if detected_timeframe:
                logger.info(f"[ENHANCED] Routing to date-based processing for timeframe: {detected_timeframe}")
                response = self.generate_date_based_summary(
                    message, user_id, detected_timeframe, include_context=False
                )
                # Generate follow-up questions
                follow_up_questions = self._generate_date_followup_questions(message, detected_timeframe)
                return response, follow_up_questions
                
            elif is_project_summary:
                logger.info(f"[ENHANCED] Routing to project summary processing")
                response = self.generate_comprehensive_project_summary(
                    message, user_id, final_project_id, include_context=False
                )
                # Generate follow-up questions  
                follow_up_questions = self._generate_project_followup_questions(message, final_project_id)
                return response, follow_up_questions
            
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
            
            # Use enhanced query intelligence for better processing
            is_summary_query = self.detect_enhanced_summary_query(message)
            detected_timeframe = self.detect_timeframe_from_query(message)
            query_filters = self.analyze_query_for_filters(message)
            
            # Increase context limit for intelligent queries
            context_limit = 100 if is_summary_query else 50
            if detected_timeframe or query_filters:
                context_limit = min(context_limit + 25, 150)  # Boost for sophisticated queries
                
            logger.info(f"[ENHANCED_QUERY] Enhanced query analysis:")
            logger.info(f"   - Is summary query: {is_summary_query}")
            logger.info(f"   - Detected timeframe: {detected_timeframe}")
            logger.info(f"   - Query filters: {query_filters}")
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
    
    # REMOVED: _detect_summary_query_fallback - replaced by detect_enhanced_summary_query
    
    def _is_comprehensive_query(self, message: str) -> bool:
        """Detect if query is asking for comprehensive analysis."""
        comprehensive_indicators = [
            'comprehensive', 'complete picture', 'full scope', 'everything',
            'all meetings', 'all documents', 'entire', 'whole', 'total'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in comprehensive_indicators)
    
    # ===============================================
    # ENHANCED QUERY INTELLIGENCE (from original)
    # ===============================================
    
    def analyze_query_for_filters(self, query: str) -> Dict:
        """
        Analyze query to determine appropriate metadata filters using advanced pattern detection.
        
        Integrates sophisticated intelligence from original meeting_processor.py:
        - Speaker-specific queries: "What did John say?", "according to Sarah"
        - Decision-focused queries: "What decisions were made?"
        - Action item queries: "What are the action items?", "What tasks?"
        - Importance filtering: "What are the critical issues?"
        """
        filters = {}
        query_lower = query.lower()
        
        # 1. SPEAKER-SPECIFIC QUERY DETECTION
        speaker_patterns = ['what did', 'what said', 'who said', 'mentioned by', 'according to']
        if any(pattern in query_lower for pattern in speaker_patterns):
            potential_speakers = []
            words = query.split()
            
            # Enhanced pattern matching for speaker extraction
            for i, word in enumerate(words):
                word_lower = word.lower()
                if word_lower in ['what', 'who'] and i + 1 < len(words):
                    next_word = words[i + 1].lower()
                    if next_word in ['did', 'said']:
                        # Look for the speaker name after "what did" or "who said"
                        if i + 2 < len(words):
                            speaker_name = words[i + 2].strip('.,?!').title()
                            potential_speakers.append(speaker_name)
                
                # Also check for patterns like "sandeep said" or "according to john"
                if word_lower in ['said', 'mentioned'] and i > 0:
                    speaker_name = words[i - 1].strip('.,?!').title()
                    potential_speakers.append(speaker_name)
                elif word_lower == 'to' and i > 0 and words[i - 1].lower() == 'according':
                    if i + 1 < len(words):
                        speaker_name = words[i + 1].strip('.,?!').title()
                        potential_speakers.append(speaker_name)
            
            # Remove duplicates and common words
            common_words = {'the', 'in', 'at', 'on', 'for', 'with', 'by', 'from', 'meeting', 'discussion'}
            potential_speakers = list(set([s for s in potential_speakers if s.lower() not in common_words]))
            
            if potential_speakers:
                filters['speakers'] = potential_speakers
                logger.info(f"Extracted speakers from query '{query}': {potential_speakers}")
        
        # 2. DECISION-FOCUSED QUERY DETECTION
        decision_patterns = ['decision', 'decided', 'conclusion', 'resolution', 'agreed']
        if any(pattern in query_lower for pattern in decision_patterns):
            filters['has_decisions'] = True
            filters['content_type'] = 'decisions'
            logger.info(f"Detected decision-focused query: '{query}'")
        
        # 3. ACTION ITEM QUERY DETECTION
        action_patterns = ['action', 'task', 'todo', 'follow up', 'next steps', 'assigned']
        if any(pattern in query_lower for pattern in action_patterns):
            filters['has_actions'] = True
            filters['content_type'] = 'actions'
            logger.info(f"Detected action-focused query: '{query}'")
        
        # 4. HIGH-IMPORTANCE QUERY DETECTION
        importance_patterns = ['important', 'critical', 'urgent', 'key', 'major']
        if any(pattern in query_lower for pattern in importance_patterns):
            filters['min_importance'] = 0.7
            logger.info(f"Detected high-importance query: '{query}'")
        
        # 5. TOPIC-SPECIFIC QUERY DETECTION
        topic_patterns = ['topic', 'subject', 'about', 'regarding', 'concerning']
        if any(pattern in query_lower for pattern in topic_patterns):
            filters['content_type'] = 'topics'
            logger.info(f"Detected topic-focused query: '{query}'")
        
        return filters
    
    def detect_timeframe_from_query(self, query: str) -> Optional[str]:
        """
        Enhanced timeframe detection from natural language query.
        
        Supports 17 comprehensive timeframe patterns from original code:
        - Current periods: 'this week', 'current month', 'this quarter'
        - Past periods: 'last week', 'past month', 'previous quarter'  
        - Specific counts: 'last 7 days', 'past 30 days', 'last 90 days'
        - Extended periods: 'last 6 months', 'past year'
        """
        query_lower = query.lower()
        
        # Comprehensive timeframe patterns with priority (from original code)
        timeframe_patterns = [
            # Current periods (highest priority)
            (['current week', 'this week'], 'current_week'),
            (['current month', 'this month'], 'current_month'),
            (['current quarter', 'this quarter'], 'current_quarter'),
            (['current year', 'this year'], 'current_year'),
            
            # Last periods (high priority)
            (['last week', 'past week', 'previous week'], 'last_week'),
            (['last month', 'past month', 'previous month'], 'last_month'),
            (['last quarter', 'past quarter', 'previous quarter'], 'last_quarter'),
            (['last year', 'past year', 'previous year'], 'last_year'),
            
            # Specific day counts (medium priority)
            (['last 7 days', 'past 7 days', 'last seven days'], 'last_7_days'),
            (['last 14 days', 'past 14 days', 'last two weeks'], 'last_14_days'),
            (['last 30 days', 'past 30 days', 'last thirty days'], 'last_30_days'),
            (['last 60 days', 'past 60 days', 'last sixty days'], 'last_60_days'),
            (['last 90 days', 'past 90 days', 'last ninety days'], 'last_90_days'),
            
            # Extended periods (lower priority)
            (['last 3 months', 'past 3 months', 'last three months'], 'last_3_months'),
            (['last 6 months', 'past 6 months', 'last six months'], 'last_6_months'),
            (['last 12 months', 'past 12 months', 'last twelve months'], 'last_12_months'),
            
            # Recent periods (lowest priority)
            (['recent', 'recently', 'lately'], 'recent'),
        ]
        
        # Find the best match (earliest occurrence has highest priority)
        for patterns, timeframe in timeframe_patterns:
            for pattern in patterns:
                if pattern in query_lower:
                    logger.info(f"Detected timeframe '{timeframe}' from query: '{query}'")
                    return timeframe
        
        return None
    
    def detect_enhanced_summary_query(self, query: str) -> bool:
        """
        Enhanced summary query detection with comprehensive keyword patterns.
        
        Supports 16 summary patterns from original code for better detection.
        """
        summary_keywords = [
            'summarize', 'summary', 'summaries', 'overview', 'brief', 
            'recap', 'highlights', 'key points', 'main points',
            'all meetings', 'all documents', 'overall', 'across all',
            'consolidate', 'aggregate', 'compile', 'comprehensive',
            'meetings summary', 'meeting summaries', 'summarize meetings',
            'summarize the meetings', 'summary of meetings', 'summary of all'
        ]
        
        query_lower = query.lower()
        detected = any(keyword in query_lower for keyword in summary_keywords)
        
        if detected:
            logger.info(f"Enhanced summary query detected: '{query}'")
        
        return detected
    
    # ===============================================
    # DATE-BASED CHRONOLOGICAL PROCESSING (from original)
    # ===============================================
    
    def generate_date_based_summary(self, query: str, user_id: str, timeframe: str, include_context: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Generate intelligent date-based summary with chronological organization.
        
        Integrates sophisticated date processing from original meeting_processor.py:
        - Chronological document sorting by date
        - Date-based grouping and organization
        - Timeframe-specific context building
        - Natural language timeframe handling
        """
        try:
            from collections import defaultdict
            import sqlite3
            
            logger.info(f"Generating date-based summary for timeframe: {timeframe}")
            
            # Get documents filtered by timeframe
            documents = self.db_manager.get_documents_by_timeframe(timeframe, user_id)
            
            if not documents:
                error_msg = f"No documents found for the specified timeframe: {timeframe.replace('_', ' ')}"
                return (error_msg, "") if include_context else error_msg
            
            # Sort documents by date (chronological processing)
            sorted_docs = sorted(documents, key=lambda x: x.date if hasattr(x, 'date') and x.date else datetime.min)
            
            # Group documents by date for better organization
            date_groups = defaultdict(list)
            for doc in sorted_docs:
                # Extract date from document
                doc_date = doc.date if hasattr(doc, 'date') and doc.date else None
                if doc_date:
                    try:
                        if isinstance(doc_date, str):
                            date_obj = datetime.fromisoformat(doc_date.replace('Z', '+00:00'))
                        else:
                            date_obj = doc_date
                        date_key = date_obj.strftime('%Y-%m-%d')
                        date_groups[date_key].append(doc)
                    except:
                        date_groups['unknown'].append(doc)
                else:
                    date_groups['unknown'].append(doc)
            
            # Build comprehensive chronological context
            context_parts = []
            document_summaries = []
            
            for date_key, docs in sorted(date_groups.items()):
                if date_key != 'unknown':
                    try:
                        date_formatted = datetime.strptime(date_key, '%Y-%m-%d').strftime('%B %d, %Y')
                        context_parts.append(f"\n=== {date_formatted} ===")
                    except:
                        context_parts.append(f"\n=== {date_key} ===")
                else:
                    context_parts.append(f"\n=== Date Unknown ===")
                
                for doc in docs:
                    # Add document summary with metadata
                    doc_summary = f"Document: {doc.filename}\n"
                    
                    # Get document chunks for content summary
                    chunks = self.db_manager.get_document_chunks(doc.document_id)
                    if chunks:
                        # Create content preview from first few chunks
                        content_preview = ""
                        for chunk in chunks[:3]:  # First 3 chunks
                            if hasattr(chunk, 'content'):
                                content_preview += chunk.content[:200] + " "
                            elif isinstance(chunk, dict):
                                content_preview += chunk.get('content', '')[:200] + " "
                        
                        if content_preview:
                            doc_summary += f"Content Preview: {content_preview.strip()}...\n"
                    
                    # Add available metadata
                    if hasattr(doc, 'content_summary') and doc.content_summary:
                        doc_summary += f"Summary: {doc.content_summary}\n"
                    if hasattr(doc, 'main_topics') and doc.main_topics:
                        doc_summary += f"Topics: {', '.join(doc.main_topics) if isinstance(doc.main_topics, list) else doc.main_topics}\n"
                    if hasattr(doc, 'participants') and doc.participants:
                        doc_summary += f"Participants: {', '.join(doc.participants) if isinstance(doc.participants, list) else doc.participants}\n"
                    
                    context_parts.append(doc_summary)
                    document_summaries.append(doc_summary)
            
            # Create comprehensive context
            full_context = '\n'.join(context_parts)
            
            # Generate summary prompt based on query type
            timeframe_display = timeframe.replace('_', ' ').title()
            
            # Use the processor's LLM for response generation
            if self.processor and hasattr(self.processor, 'llm'):
                prompt = f"""The user asked: "{query}"

Based on the meeting documents from {timeframe_display}, please answer their question naturally and comprehensively. Focus on what they specifically asked for rather than forcing a predetermined structure.

Meeting Documents Context (Chronologically Organized):
{full_context}

Please organize your response by meeting dates to make it clear and easy to understand. Use the meeting dates as headers and provide detailed information about what happened in each meeting.

Format your response as follows:
1. Start with a brief overview of the timeframe
2. Then organize by meeting dates using clear date headers (e.g., "## July 14, 2025")
3. Under each date header, provide:
   - Meeting title/purpose (if available)
   - Key participants mentioned
   - Main topics discussed
   - Important decisions made
   - Action items or next steps
   - Relevant quotes or specific details

Make the information comprehensive and well-organized so the user can easily understand what happened in each meeting and when.

IMPORTANT: When referencing information from the documents, always cite the document filename rather than chunk numbers. This helps users know which specific document the information comes from."""

                from langchain.schema import HumanMessage, SystemMessage
                
                messages = [
                    SystemMessage(content="You are a helpful AI assistant that provides well-organized, detailed responses about meeting documents. Organize information by meeting dates with clear headers and structured details under each date. Provide comprehensive answers with specific details, quotes, and context. Always cite document filenames rather than chunk numbers when referencing information."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.processor.llm.invoke(messages).content
                
                logger.info(f"Generated date-based summary: {len(response)} characters")
                
                if include_context:
                    return response, full_context
                else:
                    return response
            else:
                error_msg = "AI processing is not available for date-based summaries."
                return (error_msg, "") if include_context else error_msg
                
        except Exception as e:
            logger.error(f"Error generating date-based summary: {e}")
            import traceback
            logger.error(f"Date summary traceback: {traceback.format_exc()}")
            error_msg = f"I encountered an error processing your date-based query: {str(e)}"
            return (error_msg, "") if include_context else error_msg
    
    # ===============================================
    # PROJECT-SPECIFIC ANALYSIS FEATURES (from original)
    # ===============================================
    
    def detect_project_summary_query(self, query: str) -> bool:
        """
        Detect if the query is asking for a comprehensive project summary.
        
        Integrates project detection patterns from original code.
        """
        project_summary_keywords = [
            'project summary', 'project summaries', 'summarize project', 'summarize the project',
            'summary of project', 'summary of all files', 'all files summary', 'comprehensive summary',
            'summarize all meetings', 'all meetings summary', 'overall project', 'entire project',
            'project overview', 'complete summary', 'full summary', 'all documents summary',
            'project recap', 'project highlights', 'all files in project', 'everything in project'
        ]
        
        query_lower = query.lower()
        detected = any(keyword in query_lower for keyword in project_summary_keywords)
        
        if detected:
            logger.info(f"Project summary query detected: '{query}'")
        
        return detected
    
    def generate_comprehensive_project_summary(self, query: str, user_id: str, project_id: str = None, include_context: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Generate flexible comprehensive project analysis.
        
        Integrates sophisticated project processing from original meeting_processor.py:
        - Project-specific document filtering
        - Comprehensive cross-document analysis  
        - User-centric flexible response generation
        - Context-aware project insights
        """
        try:
            logger.info(f"Generating comprehensive project summary for user {user_id}, project {project_id}")
            
            # Get all documents in the project
            if project_id:
                documents = self.db_manager.get_project_documents(project_id, user_id)
            else:
                documents = self.db_manager.get_all_documents(user_id)
            
            if not documents:
                error_msg = "No documents found in the project to analyze."
                return (error_msg, "") if include_context else error_msg
            
            total_files = len(documents)
            logger.info(f"Found {total_files} files to process for project query: '{query}'")
            
            # Build comprehensive project context
            context_parts = []
            project_overview = []
            
            # Organize by project structure
            for doc in documents:
                doc_summary = f"\n=== {doc.filename} ===\n"
                
                # Get document content through chunks
                chunks = self.db_manager.get_document_chunks(doc.document_id)
                
                if chunks:
                    # Extract key information from chunks
                    doc_content = ""
                    for chunk in chunks[:5]:  # First 5 chunks for overview
                        if hasattr(chunk, 'content'):
                            doc_content += chunk.content + " "
                        elif isinstance(chunk, dict):
                            doc_content += chunk.get('content', '') + " "
                    
                    if doc_content:
                        doc_summary += f"Content: {doc_content[:500]}...\n"
                
                # Add metadata if available
                if hasattr(doc, 'content_summary') and doc.content_summary:
                    doc_summary += f"Summary: {doc.content_summary}\n"
                if hasattr(doc, 'date') and doc.date:
                    doc_summary += f"Date: {doc.date}\n"
                if hasattr(doc, 'main_topics') and doc.main_topics:
                    topics_str = ', '.join(doc.main_topics) if isinstance(doc.main_topics, list) else doc.main_topics
                    doc_summary += f"Key Topics: {topics_str}\n"
                if hasattr(doc, 'participants') and doc.participants:
                    speakers_str = ', '.join(doc.participants) if isinstance(doc.participants, list) else doc.participants
                    doc_summary += f"Participants: {speakers_str}\n"
                
                context_parts.append(doc_summary)
                project_overview.append(doc_summary)
            
            # Create comprehensive project context
            full_context = '\n'.join(context_parts)
            
            # Generate project-specific response
            if self.processor and hasattr(self.processor, 'llm'):
                prompt = f"""The user asked: "{query}"

Based on ALL documents in this project ({total_files} files total), please provide a comprehensive analysis that addresses their specific question.

Project Documents Context:
{full_context}

Please organize your response by meeting dates and documents to make it clear and comprehensive. Structure your response as follows:

1. Start with a brief project overview based on the user's question
2. Organize information chronologically by meeting dates using clear headers (e.g., "## July 14, 2025 - [Meeting Title]")
3. Under each meeting/document section, provide:
   - Document source (filename)
   - Key participants mentioned
   - Main topics discussed
   - Important decisions made
   - Action items or next steps
   - Relevant quotes or specific details

Make the information well-organized and comprehensive so the user can easily understand the project timeline and what happened in each meeting.

IMPORTANT: When referencing information, always cite the specific document filename rather than document numbers or chunk references. This helps users identify the source document."""

                from langchain.schema import HumanMessage, SystemMessage
                
                messages = [
                    SystemMessage(content="You are a helpful AI assistant that provides well-organized, comprehensive responses about meeting documents and project information. Organize information chronologically by meeting dates with clear headers and structured details under each section. Provide detailed answers with specific quotes, decisions, and action items. Always cite document filenames rather than chunk numbers when referencing information."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.processor.llm.invoke(messages).content
                
                logger.info(f"Generated project summary: {len(response)} characters for {total_files} documents")
                
                if include_context:
                    return response, full_context
                else:
                    return response
            else:
                error_msg = "AI processing is not available for project summaries."
                return (error_msg, "") if include_context else error_msg
                
        except Exception as e:
            logger.error(f"Error generating project summary: {e}")
            import traceback
            logger.error(f"Project summary traceback: {traceback.format_exc()}")
            error_msg = f"I encountered an error processing your project query: {str(e)}"
            return (error_msg, "") if include_context else error_msg
    
    # ===============================================
    # FOLLOW-UP QUESTION GENERATORS (supporting methods)
    # ===============================================
    
    def _generate_date_followup_questions(self, query: str, timeframe: str) -> List[str]:
        """Generate contextual follow-up questions for date-based queries."""
        timeframe_display = timeframe.replace('_', ' ')
        
        follow_ups = [
            f"What were the key decisions made during {timeframe_display}?",
            f"Who were the main participants in meetings from {timeframe_display}?",
            f"What action items came out of {timeframe_display}?",
            f"Were there any important outcomes during {timeframe_display}?",
            f"What topics were most discussed in {timeframe_display}?"
        ]
        
        return follow_ups[:3]  # Return top 3
    
    def _generate_project_followup_questions(self, query: str, project_id: str = None) -> List[str]:
        """Generate contextual follow-up questions for project summary queries."""
        project_context = "this project" if project_id else "all projects"
        
        follow_ups = [
            f"What are the main challenges identified in {project_context}?",
            f"What are the key milestones achieved in {project_context}?",
            f"Who are the primary stakeholders involved in {project_context}?",
            f"What are the next steps planned for {project_context}?",
            f"What decisions are pending for {project_context}?"
        ]
        
        return follow_ups[:3]  # Return top 3