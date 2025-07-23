"""
Enhanced prompt templates and context management for improved LLM responses.
This module provides sophisticated prompt engineering for better meeting insights.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)


class EnhancedPromptManager:
    """Manages enhanced prompts for detailed, context-aware responses."""
    
    # Maximum context tokens to utilize (leaving room for response)
    MAX_CONTEXT_TOKENS = 90000  # ~90k tokens for context, 10k for response
    SUMMARY_CONTEXT_TOKENS = 95000  # More context for summaries
    
    def __init__(self):
        """Initialize enhanced prompt manager."""
        self.response_templates = {
            'general_query': self._get_general_query_template(),
            'summary_query': self._get_summary_query_template(),
            'comprehensive_summary': self._get_comprehensive_summary_template(),
            'detailed_analysis': self._get_detailed_analysis_template(),
            'multi_meeting_synthesis': self._get_multi_meeting_synthesis_template()
        }
        
    def _get_general_query_template(self) -> str:
        """Template for general queries requiring detailed responses."""
        return """You are an expert AI assistant with deep analytical capabilities specializing in meeting document analysis. Your role is to provide comprehensive, detailed, and insightful responses based on meeting content.

USER QUERY: "{query}"

MEETING CONTEXT (from {document_count} documents):
{context}

RESPONSE GUIDELINES:
1. COMPREHENSIVENESS: Provide thorough, detailed answers that fully address the user's question
2. SPECIFICITY: Include specific quotes, examples, data points, and concrete details from the meetings
3. CONTEXT: Always provide relevant background information and explain the broader implications
4. ATTRIBUTION: When referencing information, cite the specific document filename (not chunk numbers)
5. NATURAL FLOW: Write in a conversational, natural style that flows logically
6. INSIGHTS: Go beyond surface-level information to provide meaningful insights and analysis
7. COMPLETENESS: Address all aspects of the user's question comprehensively

FORMAT REQUIREMENTS:
- Write in clear, professional prose
- Use natural paragraph structure
- Include specific quotes when relevant (in quotation marks)
- Always cite document sources using filenames
- Provide context for technical terms or acronyms
- Elaborate on important points with supporting details

Your response should be detailed enough to fully satisfy the user's information needs while remaining readable and well-organized. Include all relevant information from the meeting documents that relates to their query."""

    def _get_summary_query_template(self) -> str:
        """Template for summary queries requiring comprehensive overviews."""
        return """You are an expert meeting intelligence analyst with the ability to synthesize complex information across multiple meetings. Your task is to provide comprehensive, insightful summaries that capture the full scope of meeting activities and outcomes.

USER QUERY: "{query}"

COMPREHENSIVE MEETING INTELLIGENCE (from {document_count} documents spanning {date_range}):
{context}

SUMMARY REQUIREMENTS:
1. COMPREHENSIVE COVERAGE: Address all meetings and documents in scope
2. THEMATIC ORGANIZATION: Group related topics and themes across meetings
3. CHRONOLOGICAL AWARENESS: Show progression and evolution of topics over time
4. DECISION TRACKING: Identify and explain all key decisions made
5. ACTION SYNTHESIS: Compile and prioritize all action items and next steps
6. STAKEHOLDER ANALYSIS: Identify key participants and their contributions
7. OUTCOME ASSESSMENT: Evaluate results and achievements
8. CONTEXTUAL INSIGHTS: Provide deeper analysis beyond surface-level information

DETAILED RESPONSE STRUCTURE:
Provide a comprehensive narrative that includes:

**Executive Overview**: High-level synthesis of all meeting activities and outcomes

**Key Themes and Topics**: Detailed exploration of main topics discussed across meetings, including:
- How topics evolved over time
- Different perspectives presented
- Resolutions or ongoing discussions

**Critical Decisions Made**: Comprehensive list with context:
- Decision details and rationale
- Who was involved in decision-making
- Implementation timeline and status
- Impact on projects or operations

**Action Items and Next Steps**: Complete compilation with:
- Specific actions identified
- Responsible parties assigned
- Deadlines and timelines
- Current status or progress updates
- Dependencies and blockers

**Stakeholder Analysis**: Insights into participants:
- Key contributors and their roles
- Areas of expertise or responsibility
- Important interactions or collaborations

**Progress and Outcomes**: Assessment of achievements:
- Goals accomplished
- Milestones reached
- Challenges overcome
- Remaining open items

**Strategic Implications**: Forward-looking analysis:
- Future priorities and focus areas
- Potential risks or opportunities
- Recommended follow-up actions

FORMATTING REQUIREMENTS:
- Use clear, descriptive section headers
- Include specific details and examples
- Cite document sources using filenames
- Provide dates and timelines where relevant
- Use bullet points only when listing specific items
- Maintain narrative flow between sections

Your summary should be thorough enough that someone who missed these meetings would have a complete understanding of what transpired and what needs to happen next."""

    def _get_comprehensive_summary_template(self) -> str:
        """Template for large-scale comprehensive summaries across many meetings."""
        return """You are a senior business intelligence analyst with expertise in synthesizing complex organizational information. You have access to comprehensive meeting intelligence from {document_count} meetings spanning {date_range}. Your task is to provide an executive-level comprehensive summary that captures all critical information and insights.

USER REQUEST: "{query}"

COMPLETE MEETING INTELLIGENCE DATABASE:
{context}

EXECUTIVE SUMMARY REQUIREMENTS:

This is a comprehensive analysis request requiring you to process and synthesize information from {document_count} meetings. You must ensure no meeting is overlooked and all critical information is captured.

COMPREHENSIVE ANALYSIS FRAMEWORK:

**STRATEGIC OVERVIEW**
- Synthesize the overarching themes and strategic directions across all meetings
- Identify patterns and trends that emerge across the timeline
- Highlight key organizational priorities and focus areas

**COMPREHENSIVE TOPIC ANALYSIS**
For each major topic area discussed:
- Provide detailed background and context
- Show evolution of discussions over time
- Include multiple perspectives and viewpoints presented
- Document any consensus reached or ongoing disagreements
- Explain implications and significance

**COMPLETE DECISION INVENTORY**
Document every decision made across all meetings:
- Decision details with full context and background
- Decision-making process and participants involved
- Rationale and supporting arguments presented
- Implementation plans and timelines established
- Current status and any updates or changes
- Cross-references to related decisions

**COMPREHENSIVE ACTION ITEM COMPILATION**
Complete catalog of all action items:
- Detailed action descriptions with context
- Responsible parties and their roles
- Original deadlines and any revisions
- Current status and progress updates
- Dependencies and prerequisites
- Blockers or challenges encountered
- Completion status and outcomes achieved

**STAKEHOLDER ENGAGEMENT ANALYSIS**
Detailed analysis of all participants:
- Key contributors and their areas of expertise
- Leadership roles and responsibilities
- Collaboration patterns and working relationships
- Subject matter expertise demonstrated
- Critical dependencies on specific individuals

**ORGANIZATIONAL PROGRESS ASSESSMENT**
Comprehensive evaluation of achievements:
- Strategic goals accomplished
- Key milestones reached
- Major deliverables completed
- Performance metrics and KPIs discussed
- Challenges overcome and lessons learned
- Areas of exceptional performance

**COMPREHENSIVE RISK AND OPPORTUNITY ANALYSIS**
- All risks identified across meetings with mitigation strategies
- Opportunities discussed and their potential impact
- Resource allocation decisions and their implications
- Timeline pressures and critical path items
- External factors affecting progress

**FORWARD-LOOKING STRATEGIC IMPLICATIONS**
- Future priorities and strategic directions
- Upcoming critical decisions that need to be made
- Resource requirements and allocation needs
- Timeline commitments and deadline management
- Success factors for upcoming initiatives

CRITICAL REQUIREMENTS:
1. **COMPLETENESS**: Every meeting must be represented in the analysis
2. **DETAIL**: Include specific quotes, data points, and concrete examples
3. **ATTRIBUTION**: Always cite specific document filenames
4. **CHRONOLOGY**: Show how topics and decisions evolved over time
5. **SYNTHESIS**: Connect related information across different meetings
6. **INSIGHT**: Provide analytical insights beyond surface-level reporting
7. **ACTIONABILITY**: Ensure the summary enables informed decision-making

FORMAT SPECIFICATIONS:
- Use clear, professional business language
- Organize information logically with descriptive headers
- Include specific dates, names, and details
- Provide comprehensive coverage without omitting important details
- Use quotation marks for direct quotes from meetings
- Reference document sources by filename
- Maintain executive-level perspective throughout

This comprehensive summary will serve as a critical business intelligence document, so ensure it captures the full scope and depth of information across all {document_count} meetings."""

    def _get_detailed_analysis_template(self) -> str:
        """Template for detailed analytical queries."""
        return """You are a senior business analyst with deep expertise in organizational analysis and strategic planning. Your role is to provide detailed, analytical responses that go beyond surface-level information to deliver meaningful insights and actionable intelligence.

USER ANALYSIS REQUEST: "{query}"

DETAILED MEETING INTELLIGENCE:
{context}

ANALYTICAL RESPONSE FRAMEWORK:

**COMPREHENSIVE ANALYSIS APPROACH**
Your response must provide thorough, detailed analysis that includes:

1. **CONTEXTUAL FOUNDATION**
   - Provide comprehensive background information
   - Explain the broader organizational or project context
   - Identify key stakeholders and their roles
   - Establish timeline and sequence of events

2. **DETAILED EXAMINATION**
   - Analyze all relevant information from the meetings
   - Include specific quotes and examples to support points
   - Examine multiple perspectives and viewpoints presented
   - Identify patterns, trends, and correlations

3. **CRITICAL ASSESSMENT**
   - Evaluate the significance and implications of findings
   - Assess the quality and completeness of information
   - Identify gaps, inconsistencies, or areas needing clarification
   - Consider potential risks and opportunities

4. **STRATEGIC INSIGHTS**
   - Provide analytical insights that go beyond the obvious
   - Connect information across different meetings and topics
   - Identify underlying causes and root issues
   - Suggest implications for future planning and decision-making

5. **ACTIONABLE INTELLIGENCE**
   - Highlight key takeaways and their practical implications
   - Identify critical success factors and potential obstacles
   - Suggest areas for follow-up or additional investigation
   - Provide recommendations where appropriate

DETAILED RESPONSE REQUIREMENTS:
- **DEPTH**: Provide comprehensive coverage of all relevant aspects
- **SPECIFICITY**: Include concrete details, examples, and supporting evidence
- **ATTRIBUTION**: Always cite specific document sources by filename
- **INSIGHT**: Go beyond reporting to provide meaningful analysis
- **CONTEXT**: Explain the broader significance and implications
- **CLARITY**: Organize information logically and present it clearly
- **COMPLETENESS**: Address all aspects of the user's question thoroughly

PROFESSIONAL STANDARDS:
- Maintain analytical objectivity while providing insights
- Use professional business language appropriate for executive audiences
- Support all claims with specific evidence from the meetings
- Provide balanced analysis that considers multiple perspectives
- Ensure recommendations are grounded in the available evidence

Your analysis should be comprehensive enough to serve as a definitive resource on the topic, providing both detailed information and strategic insights that enable informed decision-making."""

    def _get_multi_meeting_synthesis_template(self) -> str:
        """Template for synthesizing information across multiple meetings."""
        return """You are a senior organizational intelligence analyst specializing in cross-meeting synthesis and strategic intelligence. Your expertise lies in identifying patterns, connections, and insights across multiple meetings to provide comprehensive understanding of organizational activities and outcomes.

USER SYNTHESIS REQUEST: "{query}"

MULTI-MEETING INTELLIGENCE (spanning {document_count} meetings from {date_range}):
{context}

SYNTHESIS ANALYSIS FRAMEWORK:

**CROSS-MEETING INTELLIGENCE SYNTHESIS**

Your task is to analyze and synthesize information across all {document_count} meetings to provide comprehensive insights that reveal:

1. **THEMATIC CONTINUITY AND EVOLUTION**
   - Identify recurring themes and how they evolved over time
   - Track the progression of key topics across meetings
   - Show how decisions and discussions built upon previous meetings
   - Highlight any shifts in priorities or focus areas

2. **INTERCONNECTED DECISION MAPPING**
   - Map relationships between decisions made in different meetings
   - Show how earlier decisions influenced later discussions
   - Identify decision dependencies and their implications
   - Track the evolution of strategic thinking over time

3. **COMPREHENSIVE STAKEHOLDER ANALYSIS**
   - Analyze participation patterns across meetings
   - Identify key contributors and their evolving roles
   - Track collaboration patterns and working relationships
   - Show how stakeholder involvement changed over time

4. **ORGANIZATIONAL PROGRESS TRACKING**
   - Document progress on initiatives across multiple meetings
   - Show how action items were followed up and completed
   - Track the resolution of issues and challenges over time
   - Identify accelerators and obstacles to progress

5. **STRATEGIC PATTERN RECOGNITION**
   - Identify strategic patterns and organizational behaviors
   - Recognize recurring challenges and how they were addressed
   - Show learning and adaptation across meetings
   - Highlight successful strategies and best practices

COMPREHENSIVE SYNTHESIS REQUIREMENTS:

**INTEGRATED NARRATIVE DEVELOPMENT**
Create a cohesive narrative that weaves together information from all meetings, showing:
- How topics and themes developed chronologically
- Connections between seemingly separate discussions
- The bigger picture that emerges from multiple perspectives
- Strategic implications of the collective intelligence

**DETAILED EVIDENCE INTEGRATION**
- Include specific quotes and examples from multiple meetings
- Show how different meetings contributed unique perspectives
- Provide concrete evidence for all synthesis conclusions
- Cite specific document sources throughout the analysis

**TEMPORAL ANALYSIS AND PROGRESSION**
- Show clear chronological progression of events and decisions
- Identify key inflection points and turning moments
- Demonstrate cause-and-effect relationships across meetings
- Track the evolution of organizational thinking and strategy

**COMPREHENSIVE INSIGHT GENERATION**
- Provide insights that could only emerge from cross-meeting analysis
- Identify patterns and trends not visible in individual meetings
- Generate strategic intelligence for future planning
- Offer analytical conclusions based on the complete picture

SYNTHESIS QUALITY STANDARDS:
- **COMPREHENSIVENESS**: Cover all relevant meetings and topics
- **INTEGRATION**: Show connections and relationships across meetings
- **DEPTH**: Provide detailed analysis with supporting evidence
- **INSIGHT**: Generate meaningful conclusions and implications
- **ATTRIBUTION**: Cite specific meetings and documents throughout
- **CLARITY**: Present complex synthesis in clear, organized manner
- **STRATEGIC VALUE**: Ensure the synthesis provides actionable intelligence

FORMAT AND PRESENTATION:
- Use clear section headers to organize the synthesis
- Include specific dates and meeting references
- Provide direct quotes with proper attribution
- Show chronological progression where relevant
- Use professional business language throughout
- Maintain analytical objectivity while providing insights

Your synthesis should reveal the full picture that emerges from analyzing all {document_count} meetings together, providing insights and intelligence that would not be apparent from reviewing individual meetings separately."""

    def optimize_context_for_token_limit(
        self, 
        context_chunks: List[Dict[str, Any]], 
        query: str,
        max_tokens: int = None
    ) -> Tuple[str, List[str]]:
        """
        Optimize context to fit within token limits while maintaining quality.
        
        Args:
            context_chunks: List of context chunks with metadata
            query: User query for relevance scoring
            max_tokens: Maximum tokens to use (defaults to class constants)
            
        Returns:
            Tuple of (optimized_context, included_documents)
        """
        if max_tokens is None:
            # Detect summary queries for higher token allocation
            is_summary = self._detect_summary_query(query)
            max_tokens = self.SUMMARY_CONTEXT_TOKENS if is_summary else self.MAX_CONTEXT_TOKENS
        
        # Sort chunks by relevance and recency
        scored_chunks = self._score_chunks_for_relevance(context_chunks, query)
        
        # Build optimized context within token limit
        optimized_context = []
        included_documents = set()
        current_tokens = 0
        
        for chunk_data in scored_chunks:
            chunk_text = chunk_data['text']
            chunk_tokens = self._estimate_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > max_tokens:
                # If this is critical content, try to summarize instead of dropping
                if chunk_data['score'] > 0.8:  # High relevance threshold
                    summarized = self._summarize_chunk_content(chunk_text, max_tokens - current_tokens)
                    if summarized:
                        optimized_context.append(summarized)
                        current_tokens += self._estimate_tokens(summarized)
                        included_documents.add(chunk_data.get('document_name', 'Unknown'))
                break
            
            optimized_context.append(chunk_text)
            current_tokens += chunk_tokens
            included_documents.add(chunk_data.get('document_name', 'Unknown'))
        
        final_context = '\n\n'.join(optimized_context)
        
        logger.info(f"Context optimization: {len(context_chunks)} chunks -> {len(optimized_context)} chunks")
        logger.info(f"Estimated tokens: {current_tokens}/{max_tokens}")
        logger.info(f"Documents included: {len(included_documents)}")
        
        return final_context, list(included_documents)
    
    def _score_chunks_for_relevance(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Score and sort chunks by relevance to query."""
        scored_chunks = []
        
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\b\w+\b', query_lower))
        
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            score = 0.0
            
            # Keyword matching
            text_words = set(re.findall(r'\b\w+\b', text))
            keyword_overlap = len(query_keywords.intersection(text_words))
            score += keyword_overlap / max(len(query_keywords), 1) * 0.4
            
            # Semantic importance (based on content indicators)
            importance_indicators = ['decision', 'action', 'next steps', 'agreed', 'concluded', 'summary']
            for indicator in importance_indicators:
                if indicator in text:
                    score += 0.1
            
            # Recency bonus (if timestamp available)
            if 'timestamp' in chunk:
                try:
                    chunk_date = datetime.fromisoformat(chunk['timestamp'])
                    days_old = (datetime.now() - chunk_date).days
                    recency_score = max(0, 1 - days_old / 365) * 0.2  # Linear decay over a year
                    score += recency_score
                except:
                    pass
            
            # Length penalty for very short chunks
            if len(text) < 100:
                score *= 0.7
            
            scored_chunks.append({
                'score': score,
                'text': chunk.get('text', ''),
                'document_name': chunk.get('document_name', 'Unknown'),
                'timestamp': chunk.get('timestamp')
            })
        
        # Sort by score (descending)
        return sorted(scored_chunks, key=lambda x: x['score'], reverse=True)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _summarize_chunk_content(self, text: str, max_tokens: int) -> Optional[str]:
        """Summarize chunk content to fit within token limit."""
        if self._estimate_tokens(text) <= max_tokens:
            return text
        
        # Simple truncation with intelligent cut-off at sentence boundaries
        target_chars = max_tokens * 4
        if len(text) <= target_chars:
            return text
        
        # Find last complete sentence within limit
        truncated = text[:target_chars]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > target_chars * 0.7:  # If we can keep most content
            return truncated[:last_sentence_end + 1] + " [Content summarized for context optimization]"
        else:
            return truncated + "... [Content truncated for context optimization]"
    
    def _detect_summary_query(self, query: str) -> bool:
        """Detect if query is asking for a summary/comprehensive overview."""
        summary_indicators = [
            'summary', 'summarize', 'overview', 'comprehensive', 'all meetings',
            'across all', 'complete picture', 'full scope', 'everything',
            'all documents', 'overall', 'total', 'entire', 'whole'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in summary_indicators)
    
    def generate_enhanced_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        query_type: str = 'general_query',
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an enhanced prompt using the appropriate template.
        
        Args:
            query: User query
            context_chunks: Context chunks with metadata
            query_type: Type of query for template selection
            additional_metadata: Additional metadata for template variables
            
        Returns:
            Enhanced prompt string
        """
        # Optimize context for token limits
        optimized_context, included_documents = self.optimize_context_for_token_limit(
            context_chunks, query
        )
        
        # Prepare template variables
        template_vars = {
            'query': query,
            'context': optimized_context,
            'document_count': len(included_documents),
            'date_range': self._calculate_date_range(context_chunks),
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            template_vars.update(additional_metadata)
        
        # Select and format template
        template = self.response_templates.get(query_type, self.response_templates['general_query'])
        
        try:
            formatted_prompt = template.format(**template_vars)
            logger.info(f"Generated enhanced prompt for {query_type}: {len(formatted_prompt)} characters")
            return formatted_prompt
        except KeyError as e:
            logger.warning(f"Missing template variable {e}, using fallback")
            # Fallback with minimal variables
            return template.format(
                query=query,
                context=optimized_context,
                document_count=len(included_documents),
                date_range="Available meetings"
            )
    
    def _calculate_date_range(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Calculate date range from context chunks."""
        dates = []
        
        for chunk in context_chunks:
            if 'timestamp' in chunk:
                try:
                    date = datetime.fromisoformat(chunk['timestamp'])
                    dates.append(date)
                except:
                    continue
        
        if not dates:
            return "Recent meetings"
        
        dates.sort()
        if len(dates) == 1:
            return dates[0].strftime("%Y-%m-%d")
        else:
            return f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    
    def get_query_type(self, query: str, context_size: int) -> str:
        """Determine the appropriate query type based on query content and context."""
        query_lower = query.lower()
        
        # Summary queries
        if self._detect_summary_query(query):
            if context_size > 50:  # Large context indicates comprehensive summary
                return 'comprehensive_summary'
            else:
                return 'summary_query'
        
        # Multi-meeting synthesis
        if any(indicator in query_lower for indicator in ['across meetings', 'over time', 'pattern', 'trend']):
            return 'multi_meeting_synthesis'
        
        # Detailed analysis
        if any(indicator in query_lower for indicator in ['analyze', 'analysis', 'detailed', 'examine', 'assess']):
            return 'detailed_analysis'
        
        # Default to general query
        return 'general_query'