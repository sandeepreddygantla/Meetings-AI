"""
Query processing module for AI-powered responses.
Handles intelligent query processing and response generation.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from src.ai.llm_client import get_llm_client, generate_response

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Service for processing AI queries and generating intelligent responses."""
    
    def __init__(self):
        """Initialize query processor."""
        self.llm_client = None
    
    def ensure_client(self) -> bool:
        """
        Ensure LLM client is available.
        
        Returns:
            True if client is available
        """
        if self.llm_client is None:
            self.llm_client = get_llm_client()
        
        return self.llm_client is not None
    
    def detect_summary_query(self, query: str) -> bool:
        """
        Detect if a query is asking for a summary or comprehensive overview.
        
        Args:
            query: User query to analyze
            
        Returns:
            True if this appears to be a summary query
        """
        summary_keywords = [
            'summary', 'summarize', 'overview', 'recap', 'what happened',
            'key points', 'main topics', 'highlights', 'takeaways',
            'comprehensive', 'complete', 'all', 'everything', 'total'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in summary_keywords)
    
    def detect_timeframe_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Detect timeframe references in queries.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with timeframe information or None
        """
        try:
            query_lower = query.lower()
            today = datetime.now().date()
            
            timeframes = {}
            
            # Detect specific time references
            if 'today' in query_lower:
                timeframes['start_date'] = today
                timeframes['end_date'] = today
                timeframes['description'] = 'today'
            
            elif 'yesterday' in query_lower:
                yesterday = today - timedelta(days=1)
                timeframes['start_date'] = yesterday
                timeframes['end_date'] = yesterday
                timeframes['description'] = 'yesterday'
            
            elif 'this week' in query_lower:
                # Calculate start of week (Monday)
                days_since_monday = today.weekday()
                start_of_week = today - timedelta(days=days_since_monday)
                timeframes['start_date'] = start_of_week
                timeframes['end_date'] = today
                timeframes['description'] = 'this week'
            
            elif 'last week' in query_lower:
                # Calculate last week
                days_since_monday = today.weekday()
                start_of_this_week = today - timedelta(days=days_since_monday)
                end_of_last_week = start_of_this_week - timedelta(days=1)
                start_of_last_week = end_of_last_week - timedelta(days=6)
                timeframes['start_date'] = start_of_last_week
                timeframes['end_date'] = end_of_last_week
                timeframes['description'] = 'last week'
            
            elif 'this month' in query_lower:
                # Calculate start of month
                start_of_month = today.replace(day=1)
                timeframes['start_date'] = start_of_month
                timeframes['end_date'] = today
                timeframes['description'] = 'this month'
            
            # Look for specific date patterns (YYYY-MM-DD, MM/DD/YYYY, etc.)
            date_patterns = [
                r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',  # YYYY-MM-DD
                r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',  # MM/DD/YYYY
                r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b'   # MM-DD-YYYY
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, query)
                if matches:
                    # Handle different date formats
                    for match in matches:
                        try:
                            if len(match[0]) == 4:  # YYYY-MM-DD format
                                date = datetime.strptime(f"{match[0]}-{match[1]}-{match[2]}", "%Y-%m-%d").date()
                            else:  # MM/DD/YYYY or MM-DD-YYYY format
                                date = datetime.strptime(f"{match[0]}/{match[1]}/{match[2]}", "%m/%d/%Y").date()
                            
                            timeframes['start_date'] = date
                            timeframes['end_date'] = date
                            timeframes['description'] = date.strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue
                
                if 'start_date' in timeframes:
                    break
            
            return timeframes if timeframes else None
            
        except Exception as e:
            logger.error(f"Error detecting timeframe: {e}")
            return None
    
    def generate_intelligent_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        user_id: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an intelligent response based on query and context.
        
        Args:
            query: User's query
            context_chunks: Relevant document chunks
            user_id: User ID
            additional_context: Additional context information
            
        Returns:
            Generated response
        """
        try:
            if not self.ensure_client():
                return "I'm sorry, the AI service is currently unavailable."
            
            # Prepare context from chunks
            context_text = self._prepare_context_text(context_chunks)
            
            # Detect query type and adjust prompt accordingly
            is_summary = self.detect_summary_query(query)
            timeframe_info = self.detect_timeframe_query(query)
            
            # Build comprehensive prompt
            prompt = self._build_query_prompt(
                query, 
                context_text, 
                is_summary, 
                timeframe_info, 
                additional_context
            )
            
            # Generate response
            response = generate_response(prompt)
            
            if response:
                return self._post_process_response(response)
            else:
                return "I'm sorry, I couldn't generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"Error generating intelligent response: {e}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def generate_follow_up_questions(
        self,
        original_query: str,
        response: str,
        context_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate relevant follow-up questions.
        
        Args:
            original_query: Original user query
            response: Generated response
            context_chunks: Context chunks used
            
        Returns:
            List of follow-up questions
        """
        try:
            if not self.ensure_client():
                return []
            
            # Extract topics and entities from context
            topics = self._extract_topics_from_context(context_chunks)
            
            prompt = f"""
            Based on this conversation:
            
            User Question: {original_query}
            
            AI Response: {response}
            
            Available Topics: {', '.join(topics[:10])}  # Limit to 10 topics
            
            Generate 3-4 relevant follow-up questions that would help the user explore the topic deeper or discover related information. Focus on:
            1. Specific details mentioned in the response
            2. Related aspects not yet covered
            3. Actionable next steps
            4. Connections to other topics
            
            Format as a simple list, one question per line.
            """
            
            follow_up_response = generate_response(prompt)
            
            if follow_up_response:
                # Parse questions from response
                questions = self._parse_follow_up_questions(follow_up_response)
                return questions[:4]  # Limit to 4 questions
            else:
                return []
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    def _prepare_context_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from document chunks."""
        if not context_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(context_chunks[:20]):  # Limit to 20 chunks
            content = chunk.get('content', '')
            filename = chunk.get('filename', 'Unknown')
            
            # Add metadata if available
            metadata_parts = []
            if chunk.get('extracted_date'):
                metadata_parts.append(f"Date: {chunk['extracted_date']}")
            if chunk.get('speakers'):
                metadata_parts.append(f"Speakers: {', '.join(chunk['speakers'][:3])}")
            
            metadata_str = f" ({', '.join(metadata_parts)})" if metadata_parts else ""
            
            context_parts.append(f"[{i+1}] From {filename}{metadata_str}: {content}")
        
        return "\n\n".join(context_parts)
    
    def _build_query_prompt(
        self,
        query: str,
        context_text: str,
        is_summary: bool,
        timeframe_info: Optional[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build comprehensive query prompt."""
        
        # Base prompt
        base_prompt = f"""
        You are an intelligent meeting assistant analyzing meeting documents and transcripts.
        
        User Question: {query}
        
        Relevant Context:
        {context_text}
        """
        
        # Add specific instructions based on query type
        if is_summary:
            base_prompt += """
            
            This appears to be a summary request. Please provide:
            1. A comprehensive overview of the key points
            2. Important decisions made
            3. Action items identified
            4. Main topics discussed
            5. Notable quotes or insights
            
            Structure your response clearly with headers and bullet points where appropriate.
            """
        
        if timeframe_info:
            base_prompt += f"""
            
            The user is asking about information from: {timeframe_info.get('description', 'a specific time period')}
            Please focus on content from that timeframe.
            """
        
        # Add general instructions
        base_prompt += """
        
        Instructions:
        - Provide accurate, helpful responses based on the context provided
        - If information isn't available in the context, acknowledge this clearly
        - Use specific quotes and references when appropriate
        - Be concise but comprehensive
        - Use professional, friendly tone
        - Structure your response for readability
        """
        
        return base_prompt
    
    def _extract_topics_from_context(self, context_chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract topics from context chunks."""
        topics = set()
        
        for chunk in context_chunks:
            # Extract topics from metadata
            if chunk.get('topics'):
                topics.update(chunk['topics'][:5])  # Limit per chunk
            
            # Extract speakers as potential topics
            if chunk.get('speakers'):
                topics.update(chunk['speakers'][:3])  # Limit speakers
        
        return list(topics)
    
    def _parse_follow_up_questions(self, response: str) -> List[str]:
        """Parse follow-up questions from AI response."""
        try:
            lines = response.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and obvious non-questions
                if not line or len(line) < 10:
                    continue
                
                # Remove common prefixes
                prefixes = ['1. ', '2. ', '3. ', '4. ', '- ', '• ', '* ']
                for prefix in prefixes:
                    if line.startswith(prefix):
                        line = line[len(prefix):]
                
                # Only include lines that end with question marks or seem like questions
                if line.endswith('?') or any(q_word in line.lower() for q_word in ['what', 'how', 'when', 'where', 'why', 'who', 'which']):
                    questions.append(line)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing follow-up questions: {e}")
            return []
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response."""
        try:
            # Clean up common formatting issues
            response = response.strip()
            
            # Remove excessive newlines
            response = re.sub(r'\n{3,}', '\n\n', response)
            
            # Ensure proper spacing around headers
            response = re.sub(r'\n([A-Z][^:\n]*:)\n', r'\n\n\1\n', response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error post-processing response: {e}")
            return response