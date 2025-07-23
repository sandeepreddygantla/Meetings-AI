"""
Enhanced context management for improved LLM responses.
Handles large-scale document processing and intelligent context optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json
import re
from dataclasses import dataclass
from collections import defaultdict

from .enhanced_prompts import EnhancedPromptManager

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Structured container for query context and metadata."""
    query: str
    user_id: str
    document_ids: Optional[List[str]] = None
    project_id: Optional[str] = None
    meeting_ids: Optional[List[str]] = None
    date_filters: Optional[Dict[str, Any]] = None
    folder_path: Optional[str] = None
    is_summary_query: bool = False
    is_comprehensive: bool = False
    context_limit: int = 50


class EnhancedContextManager:
    """
    Advanced context manager for processing large-scale meeting document queries.
    Optimizes context usage for 100k token models and provides intelligent routing.
    """
    
    def __init__(self, db_manager, processor=None):
        """
        Initialize enhanced context manager.
        
        Args:
            db_manager: Database manager instance
            processor: Legacy processor for backwards compatibility
        """
        self.db_manager = db_manager
        self.processor = processor
        self.prompt_manager = EnhancedPromptManager()
        
        # Context optimization settings
        self.MAX_DOCUMENTS_GENERAL = 100  # For general queries
        self.MAX_DOCUMENTS_SUMMARY = 1000  # For summary queries (utilize full context)
        self.MIN_CHUNK_RELEVANCE = 0.3   # Minimum relevance score for inclusion
        
    def process_enhanced_query(
        self,
        query_context: QueryContext
    ) -> Tuple[str, List[str], str]:
        """
        Process query with enhanced context management and routing.
        
        Args:
            query_context: Structured query context
            
        Returns:
            Tuple of (response, follow_up_questions, timestamp)
        """
        try:
            logger.info(f"[ENHANCED] Processing query with enhanced context manager")
            logger.info(f"[QUERY] Query: '{query_context.query}'")
            logger.info(f"[CONTEXT] Summary query: {query_context.is_summary_query}")
            logger.info(f"[CONTEXT] Comprehensive: {query_context.is_comprehensive}")
            
            # Determine query routing based on filters and content
            routing_decision = self._determine_query_routing(query_context)
            logger.info(f"[ROUTING] Query routing: {routing_decision}")
            
            # Handle different query types
            if routing_decision == 'no_filters_comprehensive':
                return self._handle_comprehensive_all_meetings(query_context)
            elif routing_decision == 'filtered_comprehensive':
                return self._handle_filtered_comprehensive(query_context)
            elif routing_decision == 'targeted_query':
                return self._handle_targeted_query(query_context)
            else:
                return self._handle_general_query(query_context)
                
        except Exception as e:
            logger.error(f"[ERROR] Enhanced query processing failed: {e}")
            # Fallback to legacy processing
            return self._fallback_to_legacy(query_context)
    
    def _determine_query_routing(self, query_context: QueryContext) -> str:
        """Determine how to route the query based on context and filters."""
        query_lower = query_context.query.lower()
        
        # Check if user wants ALL meetings without filters
        no_filters = not any([
            query_context.document_ids,
            query_context.project_id,
            query_context.meeting_ids,
            query_context.folder_path
        ])
        
        # Comprehensive indicators
        comprehensive_indicators = [
            'all meetings', 'all documents', 'everything', 'comprehensive',
            'complete picture', 'full scope', 'entire', 'whole', 'summary',
            'across all', 'overall', 'summarize all'
        ]
        
        is_comprehensive = any(indicator in query_lower for indicator in comprehensive_indicators)
        
        # Summary indicators
        is_summary = query_context.is_summary_query or any(
            indicator in query_lower for indicator in ['summary', 'summarize', 'overview']
        )
        
        # Route decision logic
        if no_filters and (is_comprehensive or is_summary):
            return 'no_filters_comprehensive'
        elif not no_filters and (is_comprehensive or is_summary):
            return 'filtered_comprehensive'
        elif query_context.document_ids or query_context.meeting_ids:
            return 'targeted_query'
        else:
            return 'general_query'
    
    def _handle_comprehensive_all_meetings(self, query_context: QueryContext) -> Tuple[str, List[str], str]:
        """Handle comprehensive queries across ALL user meetings."""
        logger.info("[COMPREHENSIVE] Handling comprehensive all-meetings query")
        
        try:
            # Get ALL user documents (no filters)
            all_documents = self.db_manager.get_all_documents(query_context.user_id)
            logger.info(f"[SCOPE] Found {len(all_documents)} total documents for user")
            
            if not all_documents:
                return "I don't have any documents to analyze yet. Please upload some meeting documents first!", [], datetime.now().isoformat()
            
            # For large document sets, use intelligent sampling and chunking
            if len(all_documents) > self.MAX_DOCUMENTS_SUMMARY:
                logger.info(f"[OPTIMIZATION] Large document set ({len(all_documents)} docs), using intelligent sampling")
                selected_documents = self._intelligent_document_sampling(all_documents, query_context.query)
            else:
                selected_documents = all_documents
            
            # Get comprehensive context from selected documents
            context_chunks = self._get_comprehensive_context(
                selected_documents, 
                query_context,
                max_chunks=500  # Allow large context for comprehensive queries
            )
            
            logger.info(f"[CONTEXT] Collected {len(context_chunks)} context chunks from {len(selected_documents)} documents")
            
            # Generate enhanced response using comprehensive template
            query_type = 'comprehensive_summary' if len(all_documents) > 50 else 'summary_query'
            
            enhanced_prompt = self.prompt_manager.generate_enhanced_prompt(
                query_context.query,
                context_chunks,
                query_type=query_type,
                additional_metadata={
                    'total_documents': len(all_documents),
                    'selected_documents': len(selected_documents),
                    'sampling_used': len(all_documents) > self.MAX_DOCUMENTS_SUMMARY
                }
            )
            
            # Generate response using LLM
            response = self._generate_llm_response(enhanced_prompt)
            
            # Add metadata to response
            if len(all_documents) > self.MAX_DOCUMENTS_SUMMARY:
                response = f"*Comprehensive analysis of {len(all_documents)} meetings (intelligently sampled for optimal analysis)*\n\n{response}"
            else:
                response = f"*Comprehensive analysis of all {len(all_documents)} meetings*\n\n{response}"
            
            # Generate enhanced follow-up questions
            follow_up_questions = self._generate_comprehensive_follow_ups(query_context.query, response, context_chunks)
            
            return response, follow_up_questions, datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"[ERROR] Comprehensive query processing failed: {e}")
            return f"I encountered an error while processing your comprehensive query: {str(e)}", [], datetime.now().isoformat()
    
    def _handle_filtered_comprehensive(self, query_context: QueryContext) -> Tuple[str, List[str], str]:
        """Handle comprehensive queries with specific filters (project, meeting, etc.)."""
        logger.info("[FILTERED] Handling filtered comprehensive query")
        
        try:
            # Apply filters to get relevant documents
            filtered_documents = self._apply_document_filters(query_context)
            logger.info(f"[FILTERED] Found {len(filtered_documents)} documents after applying filters")
            
            if not filtered_documents:
                return "I couldn't find any documents matching your criteria. Please check your filters or try a broader search.", [], datetime.now().isoformat()
            
            # Get comprehensive context with full details
            context_chunks = self._get_comprehensive_context(
                filtered_documents,
                query_context,
                max_chunks=300  # Large context for filtered comprehensive
            )
            
            # Generate enhanced response
            query_type = 'multi_meeting_synthesis' if len(filtered_documents) > 10 else 'summary_query'
            
            enhanced_prompt = self.prompt_manager.generate_enhanced_prompt(
                query_context.query,
                context_chunks,
                query_type=query_type,
                additional_metadata={
                    'filter_applied': True,
                    'filtered_count': len(filtered_documents)
                }
            )
            
            response = self._generate_llm_response(enhanced_prompt)
            
            # Add filter context to response
            filter_info = self._get_filter_description(query_context)
            response = f"*Analysis based on {len(filtered_documents)} meetings{filter_info}*\n\n{response}"
            
            follow_up_questions = self._generate_comprehensive_follow_ups(query_context.query, response, context_chunks)
            
            return response, follow_up_questions, datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"[ERROR] Filtered comprehensive query failed: {e}")
            return f"I encountered an error while processing your filtered query: {str(e)}", [], datetime.now().isoformat()
    
    def _handle_targeted_query(self, query_context: QueryContext) -> Tuple[str, List[str], str]:
        """Handle targeted queries for specific documents or meetings."""
        logger.info("[TARGETED] Handling targeted query")
        
        try:
            # Get specific documents
            targeted_documents = self._apply_document_filters(query_context)
            logger.info(f"[TARGETED] Found {len(targeted_documents)} targeted documents")
            
            if not targeted_documents:
                return "I couldn't find the specific documents or meetings you're looking for. Please check your selection.", [], datetime.now().isoformat()
            
            # Get detailed context for targeted documents
            context_chunks = self._get_detailed_context(targeted_documents, query_context)
            
            # Use detailed analysis template for targeted queries
            enhanced_prompt = self.prompt_manager.generate_enhanced_prompt(
                query_context.query,
                context_chunks,
                query_type='detailed_analysis'
            )
            
            response = self._generate_llm_response(enhanced_prompt)
            
            # Generate targeted follow-up questions
            follow_up_questions = self._generate_targeted_follow_ups(query_context.query, response, context_chunks)
            
            return response, follow_up_questions, datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"[ERROR] Targeted query processing failed: {e}")
            return f"I encountered an error while processing your targeted query: {str(e)}", [], datetime.now().isoformat()
    
    def _handle_general_query(self, query_context: QueryContext) -> Tuple[str, List[str], str]:
        """Handle general queries with standard processing."""
        logger.info("[GENERAL] Handling general query")
        
        # Fall back to enhanced legacy processing for general queries
        if self.processor:
            try:
                response, context = self.processor.answer_query_with_intelligence(
                    query_context.query,
                    user_id=query_context.user_id,
                    document_ids=query_context.document_ids,
                    project_id=query_context.project_id,
                    meeting_ids=query_context.meeting_ids,
                    date_filters=query_context.date_filters,
                    folder_path=query_context.folder_path,
                    context_limit=min(query_context.context_limit, 100),  # Use enhanced context limit
                    include_context=True
                )
                
                # Generate enhanced follow-up questions
                follow_up_questions = self.processor.generate_follow_up_questions(
                    query_context.query, response, context
                )
                
                return response, follow_up_questions, datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"[ERROR] General query processing failed: {e}")
                return f"I encountered an error while processing your query: {str(e)}", [], datetime.now().isoformat()
        else:
            return "Query processing is temporarily unavailable.", [], datetime.now().isoformat()
    
    def _intelligent_document_sampling(self, documents: List[Dict], query: str) -> List[Dict]:
        """Intelligently sample documents for large-scale analysis."""
        logger.info(f"[SAMPLING] Intelligently sampling from {len(documents)} documents")
        
        # Score documents by relevance and recency
        scored_documents = []
        
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        
        for doc in documents:
            score = 0.0
            
            # Content relevance (title, summary)
            doc_text = f"{doc.get('title', '')} {doc.get('summary', '')}".lower()
            doc_words = set(re.findall(r'\b\w+\b', doc_text))
            
            if query_keywords and doc_words:
                overlap = len(query_keywords.intersection(doc_words))
                score += overlap / len(query_keywords) * 0.4
            
            # Recency score
            if doc.get('upload_date'):
                try:
                    doc_date = datetime.fromisoformat(doc['upload_date'])
                    days_old = (datetime.now() - doc_date).days
                    recency_score = max(0, 1 - days_old / 365) * 0.3
                    score += recency_score
                except:
                    pass
            
            # Size/importance score (larger documents often more important)
            content_length = len(doc.get('content', ''))
            if content_length > 1000:
                score += 0.2
            
            # Project diversity bonus (ensure coverage across projects)
            score += 0.1  # Base score for inclusion
            
            scored_documents.append((score, doc))
        
        # Sort by score and take top documents
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        
        # Take top documents, ensuring we don't exceed limit
        selected_count = min(self.MAX_DOCUMENTS_SUMMARY, len(scored_documents))
        selected_documents = [doc for _, doc in scored_documents[:selected_count]]
        
        logger.info(f"[SAMPLING] Selected {len(selected_documents)} documents using intelligent sampling")
        return selected_documents
    
    def _get_comprehensive_context(
        self, 
        documents: List[Dict], 
        query_context: QueryContext,
        max_chunks: int = 300
    ) -> List[Dict[str, Any]]:
        """Get comprehensive context from documents for large-scale analysis."""
        logger.info(f"[CONTEXT] Getting comprehensive context from {len(documents)} documents")
        
        context_chunks = []
        
        for doc in documents:
            # Get document chunks using database manager
            try:
                doc_chunks = self.db_manager.get_document_chunks(doc['document_id'])
                
                for chunk in doc_chunks:
                    context_chunks.append({
                        'text': chunk.content if hasattr(chunk, 'content') else '',
                        'document_name': doc.get('filename', 'Unknown'),
                        'document_id': doc['document_id'],
                        'chunk_id': chunk.chunk_id if hasattr(chunk, 'chunk_id') else '',
                        'timestamp': doc.get('upload_date'),
                        'metadata': {
                            'project_id': doc.get('project_id'),
                            'meeting_id': doc.get('meeting_id'),
                            'title': doc.get('title')
                        }
                    })
                    
                    if len(context_chunks) >= max_chunks:
                        break
                        
            except Exception as e:
                logger.error(f"[ERROR] Failed to get chunks for document {doc.document_id}: {e}")
                continue
            
            if len(context_chunks) >= max_chunks:
                break
        
        logger.info(f"[CONTEXT] Collected {len(context_chunks)} context chunks")
        return context_chunks
    
    def _get_detailed_context(self, documents: List[Dict], query_context: QueryContext) -> List[Dict[str, Any]]:
        """Get detailed context for targeted queries."""
        return self._get_comprehensive_context(documents, query_context, max_chunks=150)
    
    def _apply_document_filters(self, query_context: QueryContext) -> List[Dict]:
        """Apply filters to get relevant documents."""
        logger.info("[FILTER] Applying document filters")
        
        try:
            # Start with all user documents
            documents = self.db_manager.get_all_documents(query_context.user_id)
            
            # Apply project filter
            if query_context.project_id:
                documents = [d for d in documents if d.get('project_id') == query_context.project_id]
                logger.info(f"[FILTER] After project filter: {len(documents)} documents")
            
            # Apply meeting filter
            if query_context.meeting_ids:
                documents = [d for d in documents if d.get('meeting_id') in query_context.meeting_ids]
                logger.info(f"[FILTER] After meeting filter: {len(documents)} documents")
            
            # Apply document filter
            if query_context.document_ids:
                documents = [d for d in documents if d.document_id in query_context.document_ids]
                logger.info(f"[FILTER] After document filter: {len(documents)} documents")
            
            # Apply folder filter
            if query_context.folder_path:
                documents = [d for d in documents if query_context.folder_path in d.get('folder_path', '')]
                logger.info(f"[FILTER] After folder filter: {len(documents)} documents")
            
            # Apply date filters
            if query_context.date_filters:
                documents = self._apply_date_filters(documents, query_context.date_filters)
                logger.info(f"[FILTER] After date filter: {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to apply filters: {e}")
            return []
    
    def _apply_date_filters(self, documents: List[Dict], date_filters: Dict[str, Any]) -> List[Dict]:
        """Apply date-based filters to documents."""
        if not date_filters:
            return documents
        
        try:
            from datetime import datetime, timedelta
            filtered_docs = []
            
            # Extract date filter parameters
            start_date = date_filters.get('start_date')
            end_date = date_filters.get('end_date') 
            timeframe = date_filters.get('timeframe')
            
            # Convert timeframe to date range if specified
            if timeframe and not (start_date or end_date):
                from src.database.sqlite_operations import SQLiteOperations
                sqlite_ops = SQLiteOperations('meeting_documents.db')
                start_date, end_date = sqlite_ops._calculate_date_range(timeframe)
            
            # Filter documents by date range
            for doc in documents:
                doc_date = None
                
                # Extract date from document (try multiple fields)
                if 'date' in doc and doc['date']:
                    doc_date_str = doc['date']
                elif 'upload_date' in doc and doc['upload_date']:
                    doc_date_str = doc['upload_date']
                elif 'created_at' in doc and doc['created_at']:
                    doc_date_str = doc['created_at']
                else:
                    continue  # Skip documents without dates
                
                # Parse date string to datetime object
                try:
                    if isinstance(doc_date_str, str):
                        # Handle ISO format with or without timezone
                        if 'T' in doc_date_str:
                            doc_date = datetime.fromisoformat(doc_date_str.replace('Z', '+00:00'))
                        else:
                            doc_date = datetime.fromisoformat(doc_date_str)
                    elif isinstance(doc_date_str, datetime):
                        doc_date = doc_date_str
                    else:
                        continue  # Skip if date format is unknown
                except (ValueError, TypeError):
                    continue  # Skip documents with unparseable dates
                
                # Apply date range filters
                include_doc = True
                
                if start_date and doc_date < start_date:
                    include_doc = False
                    
                if end_date and doc_date > end_date:
                    include_doc = False
                
                if include_doc:
                    filtered_docs.append(doc)
            
            logger.info(f"[DATE_FILTER] Filtered {len(documents)} → {len(filtered_docs)} documents using date filters: {date_filters}")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"[ERROR] Date filtering failed: {e}")
            return documents  # Return unfiltered on error
    
    def _get_filter_description(self, query_context: QueryContext) -> str:
        """Get human-readable description of applied filters."""
        filters = []
        
        if query_context.project_id:
            filters.append(f"project {query_context.project_id}")
        
        if query_context.meeting_ids:
            if len(query_context.meeting_ids) == 1:
                filters.append(f"meeting {query_context.meeting_ids[0]}")
            else:
                filters.append(f"{len(query_context.meeting_ids)} specific meetings")
        
        if query_context.folder_path:
            filters.append(f"folder '{query_context.folder_path}'")
        
        if filters:
            return f" from {', '.join(filters)}"
        else:
            return ""
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using the global LLM instance."""
        try:
            from meeting_processor import llm
            
            if llm is None:
                logger.error("[ERROR] LLM not available")
                return "I'm sorry, but the AI system is not properly configured. Please check the system settings."
            
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content="You are a helpful AI assistant that provides detailed, conversational responses about meeting documents. Avoid structured formats with bullet points or headers unless specifically requested. Provide rich, comprehensive answers with specific details, quotes, and context. Always cite document filenames rather than chunk numbers when referencing information."),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"[ERROR] LLM response generation failed: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def _generate_comprehensive_follow_ups(
        self, 
        query: str, 
        response: str, 
        context_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate comprehensive follow-up questions for large-scale queries."""
        try:
            # Create sample context for follow-up generation
            sample_context = "\n".join([chunk['text'][:500] for chunk in context_chunks[:5]])
            
            if self.processor:
                return self.processor.generate_follow_up_questions(query, response, sample_context)
            else:
                # Generate basic follow-up questions
                return [
                    "Can you provide more details about specific meetings or topics?",
                    "What were the most important decisions made across these meetings?",
                    "Are there any action items or follow-ups I should be aware of?",
                    "How do these meetings connect to broader project goals?"
                ]
                
        except Exception as e:
            logger.error(f"[ERROR] Follow-up generation failed: {e}")
            return []
    
    def _generate_targeted_follow_ups(
        self,
        query: str,
        response: str,
        context_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate targeted follow-up questions for specific queries."""
        return self._generate_comprehensive_follow_ups(query, response, context_chunks)
    
    def _fallback_to_legacy(self, query_context: QueryContext) -> Tuple[str, List[str], str]:
        """Fallback to legacy processing if enhanced processing fails."""
        logger.info("[FALLBACK] Using legacy processing")
        
        if self.processor:
            try:
                response, context = self.processor.answer_query_with_intelligence(
                    query_context.query,
                    user_id=query_context.user_id,
                    document_ids=query_context.document_ids,
                    project_id=query_context.project_id,
                    meeting_ids=query_context.meeting_ids,
                    date_filters=query_context.date_filters,
                    folder_path=query_context.folder_path,
                    context_limit=query_context.context_limit,
                    include_context=True
                )
                
                follow_up_questions = self.processor.generate_follow_up_questions(
                    query_context.query, response, context
                )
                
                return response, follow_up_questions, datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"[ERROR] Legacy fallback failed: {e}")
                return f"I encountered an error while processing your query: {str(e)}", [], datetime.now().isoformat()
        else:
            return "Query processing is temporarily unavailable.", [], datetime.now().isoformat()