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
            
            # Detect if this is a multi-meeting query
            is_multi_meeting = self._detect_multi_meeting_query(query, context_chunks, additional_context)
            
            if is_multi_meeting:
                logger.info(f"Multi-meeting query detected with {len(set(chunk.get('filename', '') for chunk in context_chunks))} unique meetings")
                return self._generate_multi_meeting_response(query, context_chunks, user_id, additional_context)
            else:
                # Standard single response for regular queries
                return self._generate_standard_response(query, context_chunks, user_id, additional_context)
            
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
        """Build comprehensive query prompt with dynamic response scaling."""
        
        # Analyze context richness for dynamic scaling
        context_richness = self._analyze_context_richness(context_text)
        response_requirements = self._determine_response_requirements(query, context_richness)
        
        # Base prompt with enhanced instructions
        base_prompt = f"""
        You are an expert meeting intelligence assistant analyzing meeting documents and transcripts. Your goal is to provide detailed, comprehensive, and insightful responses that fully address user questions.
        
        User Question: {query}
        
        Relevant Context (from {context_richness['total_sources']} sources):
        {context_text}
        
        RESPONSE REQUIREMENTS:
        - Response Length: {response_requirements['target_length']}
        - Detail Level: {response_requirements['detail_level']}
        - Analysis Depth: {response_requirements['analysis_depth']}
        - Include Examples: {response_requirements['include_examples']}
        """
        
        # Add specific instructions based on query type
        if is_summary:
            base_prompt += """
            
            SUMMARY REQUEST DETECTED - Provide a comprehensive analysis including:
            
            ## Executive Summary
            - High-level overview of the meeting(s) or content
            - Key outcomes and main themes
            
            ## Detailed Analysis
            1. **Key Points & Topics**: Elaborate on main discussion areas with specific details
            2. **Decisions Made**: List all decisions with context and reasoning
            3. **Action Items**: Detailed action items with assignees, deadlines, and context
            4. **Important Discussions**: Significant conversations, debates, or insights
            5. **Participants & Roles**: Who was involved and their contributions
            6. **Timeline & Process**: Chronological flow of events or discussions
            
            ## Supporting Details
            - Relevant quotes and specific references
            - Background context and implications
            - Cross-references to related topics or meetings
            
            Use clear headers, bullet points, and structured formatting for maximum readability.
            """
        
        if timeframe_info:
            base_prompt += f"""
            
            TIMEFRAME ANALYSIS for {timeframe_info.get('description', 'specified period')}:
            - Focus specifically on content from this timeframe
            - Provide temporal context and progression
            - Compare with other periods if relevant data exists
            - Highlight time-sensitive information and trends
            """
        
        # Dynamic instructions based on context richness
        if context_richness['has_multiple_meetings'] or context_richness['total_sources'] > 10:
            base_prompt += f"""
            
            MULTI-MEETING ANALYSIS ({context_richness['total_sources']} sources detected):
            - Create a structured overview with clear sections for each meeting/document
            - Compare and contrast information across different meetings
            - Identify patterns, trends, and evolution of topics over time
            - Cross-reference related discussions and decisions
            - Highlight consistency or changes in direction
            - Use tables or bullet points to organize large amounts of information
            - Provide a comprehensive executive summary at the beginning
            - Include a detailed breakdown section with meeting-by-meeting analysis
            
            LARGE-SCALE QUERY HANDLING:
            - Structure response with clear headers and subheaders
            - Use numbered lists for sequential information
            - Create summary tables for key decisions and actions
            - Prioritize most critical information first
            - Ensure comprehensive coverage while maintaining readability
            """
        
        if context_richness['has_speakers']:
            base_prompt += """
            
            SPEAKER-SPECIFIC ANALYSIS:
            - Attribute statements and viewpoints to specific speakers
            - Analyze different perspectives and opinions
            - Highlight key contributions from different participants
            - Note areas of agreement or disagreement
            """
        
        # Special instructions for very large queries
        if context_richness['total_sources'] > 15:
            base_prompt += """
            
            LARGE-SCALE DOCUMENT ANALYSIS (15+ sources):
            
            ## RESPONSE STRUCTURE REQUIREMENTS:
            
            ### 1. Executive Summary (150-200 words)
            - High-level overview of all meetings/documents
            - Key themes and patterns across the entire dataset
            - Most critical findings and decisions
            
            ### 2. Detailed Analysis by Category
            - **Key Decisions**: Organized chronologically or by importance
            - **Action Items**: Grouped by responsible parties or deadlines
            - **Main Topics**: Themes that appear across multiple meetings
            - **Participant Insights**: Key contributions from different speakers
            
            ### 3. Meeting-by-Meeting Breakdown
            - Brief summary of each meeting's key points
            - Cross-references to related meetings
            - Evolution of topics over time
            
            ### 4. Synthesis and Insights
            - Patterns and trends across all meetings
            - Conflicts or consistency in decisions
            - Outstanding issues or follow-ups needed
            
            **IMPORTANT**: Use tables, bullet points, and clear formatting to handle this large volume of information effectively.
            """
        
        # Enhanced general instructions for detailed responses
        base_prompt += f"""
        
        COMPREHENSIVE RESPONSE GUIDELINES:
        
        **Content Requirements:**
        - Provide {response_requirements['target_length']} responses with rich detail
        - Include specific examples, quotes, and concrete references from the context
        - Explain not just WHAT happened, but WHY it matters and HOW it connects to broader themes
        - Use technical terminology appropriately and explain complex concepts
        - Provide background context to help users understand the full picture
        - For multi-meeting analysis: Create structured summaries with clear organization
        - Use tables, bullet points, and sections to handle large amounts of information
        - Prioritize most important information first, then provide comprehensive details
        
        **Structure & Formatting:**
        - Use clear headers and subheaders to organize information
        - Create bullet points and numbered lists for clarity
        - Include relevant metadata (dates, speakers, meeting names) when available
        - Use professional yet conversational tone
        - Ensure logical flow and coherent narrative
        
        **Analysis & Insights:**
        - Go beyond surface-level information to provide meaningful insights
        - Connect related concepts and show relationships between topics
        - Identify implications and potential next steps
        - Highlight important patterns or trends
        - Provide actionable information where relevant
        
        **Quality Standards:**
        - If information is incomplete, acknowledge limitations clearly
        - Distinguish between facts from the meetings and your analysis/interpretation
        - Use specific quotes and references to support key points
        - Ensure accuracy and avoid speculation beyond the available context
        - Maintain objectivity while providing helpful synthesis
        
        **For Multi-Meeting/Large-Scale Queries:**
        - Break down complex information into digestible sections
        - Use formatting (headers, bullets, tables) to organize large amounts of data
        - Provide both high-level summaries and detailed breakdowns
        - Cross-reference related information across different meetings
        - Maintain coherent narrative flow despite large volume of information
        
        Remember: Users want detailed, comprehensive answers that fully explore their questions. For large-scale queries involving many meetings, prioritize clear organization and structure while ensuring complete coverage.
        """
        
        return base_prompt
    
    def _analyze_context_richness(self, context_text: str) -> Dict[str, Any]:
        """Analyze the richness and quality of available context for dynamic response scaling."""
        try:
            # Basic metrics
            context_length = len(context_text)
            word_count = len(context_text.split())
            
            # Count sources (documents/meetings)
            source_indicators = context_text.count('[') + context_text.count('From ')
            
            # Detect speakers
            speaker_indicators = context_text.count('Speakers:') + context_text.count('Speaker:')
            has_speakers = speaker_indicators > 0 or any(indicator in context_text.lower() 
                                                       for indicator in ['said', 'mentioned', 'discussed', 'asked'])
            
            # Detect multiple meetings
            meeting_indicators = context_text.count('meeting') + context_text.count('Meeting')
            has_multiple_meetings = meeting_indicators > 2 or context_text.count('Date:') > 1
            
            # Detect dates and timestamps
            date_indicators = context_text.count('Date:') + context_text.count('2024') + context_text.count('2023')
            has_temporal_info = date_indicators > 0
            
            # Detect action items and decisions
            action_indicators = sum(1 for keyword in ['action', 'decision', 'follow-up', 'next steps', 'todo', 'assigned']
                                  if keyword in context_text.lower())
            has_actionable_content = action_indicators > 0
            
            # Calculate richness score (0-100)
            richness_score = min(100, (
                min(30, context_length // 100) +  # Length component (max 30)
                min(20, source_indicators * 5) +   # Source diversity (max 20)
                (15 if has_speakers else 0) +      # Speaker information (15)
                (15 if has_multiple_meetings else 0) +  # Multi-meeting (15)
                (10 if has_temporal_info else 0) +     # Temporal info (10)
                (10 if has_actionable_content else 0)   # Actionable content (10)
            ))
            
            return {
                'context_length': context_length,
                'word_count': word_count,
                'total_sources': max(1, source_indicators),
                'richness_score': richness_score,
                'has_speakers': has_speakers,
                'has_multiple_meetings': has_multiple_meetings,
                'has_temporal_info': has_temporal_info,
                'has_actionable_content': has_actionable_content
            }
            
        except Exception as e:
            logger.error(f"Error analyzing context richness: {e}")
            return {
                'context_length': len(context_text),
                'word_count': len(context_text.split()),
                'total_sources': 1,
                'richness_score': 50,  # Default medium richness
                'has_speakers': False,
                'has_multiple_meetings': False,
                'has_temporal_info': False,
                'has_actionable_content': False
            }
    
    def _determine_response_requirements(self, query: str, context_richness: Dict[str, Any]) -> Dict[str, str]:
        """Determine appropriate response requirements based on query and context."""
        try:
            query_lower = query.lower()
            richness_score = context_richness['richness_score']
            
            # Analyze query complexity
            complexity_indicators = [
                'explain', 'analyze', 'compare', 'detail', 'comprehensive', 'thorough',
                'what happened', 'tell me about', 'how did', 'why did', 'walk me through'
            ]
            
            is_complex_query = any(indicator in query_lower for indicator in complexity_indicators)
            is_summary_query = any(indicator in query_lower for indicator in ['summary', 'overview', 'recap'])
            
            # Special handling for large-scale queries (multiple meetings)
            multi_meeting_indicators = ['all meetings', 'every meeting', 'across meetings', 'all files', 'entire project']
            is_multi_meeting_query = any(indicator in query_lower for indicator in multi_meeting_indicators)
            
            # Check for numeric indicators (e.g., "30 meetings", "all 25 files")
            import re
            numeric_patterns = [r'\d+\s+meetings?', r'\d+\s+files?', r'all\s+\d+']
            has_numeric_scope = any(re.search(pattern, query_lower) for pattern in numeric_patterns)
            
            # Determine target length based on multiple factors
            if is_multi_meeting_query or has_numeric_scope or context_richness.get('total_sources', 1) > 10:
                target_length = "extensive multi-document analysis (1500-3000 words)"
                detail_level = "comprehensive with structured breakdown"
                analysis_depth = "systematic cross-document analysis"
            elif is_summary_query or richness_score > 70:
                target_length = "comprehensive (800-1200 words)"
                detail_level = "extensive"
                analysis_depth = "deep analytical"
            elif is_complex_query or richness_score > 50:
                target_length = "detailed (500-800 words)"
                detail_level = "thorough"
                analysis_depth = "analytical"
            elif richness_score > 30:
                target_length = "moderate (300-500 words)"
                detail_level = "moderate"
                analysis_depth = "explanatory"
            else:
                target_length = "focused (200-400 words)"
                detail_level = "clear and direct"
                analysis_depth = "informative"
            
            # Always include examples for better responses
            include_examples = "yes, with specific quotes and references"
            
            return {
                'target_length': target_length,
                'detail_level': detail_level,
                'analysis_depth': analysis_depth,
                'include_examples': include_examples
            }
            
        except Exception as e:
            logger.error(f"Error determining response requirements: {e}")
            return {
                'target_length': "detailed (400-600 words)",
                'detail_level': "thorough",
                'analysis_depth': "analytical",
                'include_examples': "yes, with specific examples"
            }
    
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
    
    def _detect_multi_meeting_query(self, query: str, context_chunks: List[Dict[str, Any]], additional_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Detect if this is a query that requires multi-meeting analysis.
        """
        query_lower = query.lower()
        
        # Check for explicit multi-meeting indicators
        multi_meeting_indicators = [
            'all meetings', 'every meeting', 'across meetings', 'all files', 'entire project',
            'overall summary', 'complete overview', 'comprehensive summary', 'project summary'
        ]
        
        if any(indicator in query_lower for indicator in multi_meeting_indicators):
            return True
        
        # Check for numeric indicators (e.g., "30 meetings", "all 25 files")
        import re
        numeric_patterns = [r'\d+\s+meetings?', r'\d+\s+files?', r'all\s+\d+']
        if any(re.search(pattern, query_lower) for pattern in numeric_patterns):
            return True
        
        # Check if we have chunks from many different meetings (10+ unique files)
        unique_meetings = len(set(chunk.get('filename', '') for chunk in context_chunks))
        if unique_meetings >= 10:
            return True
        
        # Check additional context
        if additional_context and additional_context.get('meetings_involved', 0) >= 10:
            return True
        
        return False
    
    def _generate_multi_meeting_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        user_id: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a comprehensive response for multi-meeting queries using a staged approach.
        """
        try:
            # Group chunks by meeting/file
            meetings_data = self._group_chunks_by_meeting(context_chunks)
            
            logger.info(f"Processing {len(meetings_data)} meetings for multi-meeting query")
            
            # Stage 1: Generate individual meeting summaries
            meeting_summaries = {}
            for meeting_name, chunks in meetings_data.items():
                try:
                    summary = self._generate_single_meeting_summary(meeting_name, chunks, query)
                    meeting_summaries[meeting_name] = summary
                    logger.info(f"Generated summary for meeting: {meeting_name}")
                except Exception as e:
                    logger.error(f"Error summarizing meeting {meeting_name}: {e}")
                    meeting_summaries[meeting_name] = f"Error generating summary for {meeting_name}"
            
            # Stage 2: Create comprehensive cross-meeting analysis
            comprehensive_response = self._create_comprehensive_multi_meeting_response(
                query, meeting_summaries, context_chunks, additional_context
            )
            
            return comprehensive_response
            
        except Exception as e:
            logger.error(f"Error in multi-meeting response generation: {e}")
            return self._generate_standard_response(query, context_chunks, user_id, additional_context)
    
    def _group_chunks_by_meeting(self, context_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group context chunks by meeting/filename.
        """
        meetings_data = {}
        for chunk in context_chunks:
            meeting_name = chunk.get('filename', 'Unknown Meeting')
            if meeting_name not in meetings_data:
                meetings_data[meeting_name] = []
            meetings_data[meeting_name].append(chunk)
        
        return meetings_data
    
    def _generate_single_meeting_summary(self, meeting_name: str, chunks: List[Dict[str, Any]], original_query: str) -> str:
        """
        Generate a focused summary for a single meeting.
        """
        try:
            # Prepare context for this specific meeting
            meeting_content = "\n\n".join([chunk.get('content', '') for chunk in chunks[:5]])  # Limit to 5 chunks per meeting
            
            # Extract key metadata
            speakers = set()
            decisions = []
            actions = []
            
            for chunk in chunks:
                if chunk.get('speakers'):
                    speakers.update(chunk['speakers'])
                if chunk.get('decisions'):
                    decisions.extend(chunk['decisions'])
                if chunk.get('actions'):
                    actions.extend(chunk['actions'])
            
            summary_prompt = f"""
            Create a concise but comprehensive summary for this meeting in the context of the user's query.
            
            Meeting: {meeting_name}
            User's Query: {original_query}
            
            Meeting Content:
            {meeting_content}
            
            Participants: {', '.join(list(speakers)[:5]) if speakers else 'Not specified'}
            
            Instructions:
            - Focus on information relevant to the user's query
            - Provide a 150-200 word summary
            - Include key decisions, actions, and discussion points
            - Mention important participants and their contributions
            - Use bullet points for clarity
            
            Format:
            **Meeting: {meeting_name}**
            - **Key Points**: [main discussion topics]
            - **Decisions**: [key decisions made]
            - **Actions**: [action items identified]
            - **Participants**: [key contributors]
            """
            
            response = generate_response(summary_prompt)
            return response if response else f"Unable to generate summary for {meeting_name}"
            
        except Exception as e:
            logger.error(f"Error generating single meeting summary: {e}")
            return f"Error summarizing {meeting_name}: {str(e)}"
    
    def _create_comprehensive_multi_meeting_response(
        self,
        query: str,
        meeting_summaries: Dict[str, str],
        all_chunks: List[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a comprehensive response that synthesizes all meeting summaries.
        """
        try:
            # Combine all meeting summaries
            all_summaries_text = "\n\n".join([f"{summary}" for summary in meeting_summaries.values()])
            
            # Extract overall patterns
            all_speakers = set()
            all_decisions = []
            all_actions = []
            
            for chunk in all_chunks:
                if chunk.get('speakers'):
                    all_speakers.update(chunk['speakers'])
                if chunk.get('decisions'):
                    all_decisions.extend(chunk['decisions'])
                if chunk.get('actions'):
                    all_actions.extend(chunk['actions'])
            
            synthesis_prompt = f"""
            Based on the individual meeting summaries provided, create a comprehensive response to the user's query.
            
            User's Query: {query}
            
            Number of Meetings Analyzed: {len(meeting_summaries)}
            
            Individual Meeting Summaries:
            {all_summaries_text}
            
            Overall Statistics:
            - Total Participants: {len(all_speakers)}
            - Total Decisions: {len(all_decisions)}
            - Total Actions: {len(all_actions)}
            
            Instructions:
            Create a structured, comprehensive response with the following sections:
            
            ## Executive Summary (200-300 words)
            - High-level overview addressing the user's query
            - Key themes across all meetings
            - Most important findings
            
            ## Detailed Analysis
            ### Key Themes and Patterns
            - Common topics across meetings
            - Evolution of discussions over time
            
            ### Critical Decisions
            - Most important decisions made
            - Cross-meeting decision patterns
            
            ### Action Items and Follow-ups
            - Key action items across all meetings
            - Outstanding issues or recurring themes
            
            ### Participant Insights
            - Key contributors across meetings
            - Different perspectives and viewpoints
            
            ## Meeting-by-Meeting Highlights
            [Brief key points from each meeting - 2-3 sentences each]
            
            ## Summary and Next Steps
            - Overall conclusions
            - Recommended follow-up actions
            - Outstanding questions or issues
            
            Use clear formatting with headers, bullet points, and organized structure to handle this large amount of information effectively.
            """
            
            response = generate_response(synthesis_prompt)
            return response if response else "Unable to generate comprehensive multi-meeting analysis"
            
        except Exception as e:
            logger.error(f"Error creating comprehensive response: {e}")
            return f"Error creating comprehensive analysis: {str(e)}"
    
    def _generate_standard_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        user_id: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate standard response for regular (non-multi-meeting) queries.
        """
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