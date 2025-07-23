"""
Enhanced LLM configuration for improved context handling and response quality.
Optimized for 100k token models with comprehensive meeting analysis capabilities.
"""

import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class EnhancedLLMConfig:
    """Enhanced LLM configuration for optimized meeting analysis."""
    
    # Enhanced context limits for 100k token models
    CONTEXT_LIMITS = {
        'general_query': 75,           # Increased from 10-50
        'summary_query': 150,          # Increased from 100  
        'comprehensive_summary': 300,   # New: For large-scale summaries
        'detailed_analysis': 100,       # New: For detailed analytical queries
        'multi_meeting_synthesis': 200, # New: For cross-meeting analysis
        'legacy_fallback': 50          # Conservative fallback
    }
    
    # Token optimization settings
    MAX_CONTEXT_TOKENS = 90000      # Reserve 10k tokens for response
    MAX_RESPONSE_TOKENS = 10000     # Maximum response length
    TOKEN_BUFFER = 5000             # Safety buffer for calculations
    
    # Response quality settings
    RESPONSE_QUALITY = {
        'min_response_length': 200,     # Minimum response length (characters)
        'detailed_response_threshold': 500,  # When to provide detailed responses
        'comprehensive_threshold': 1000,     # When to provide comprehensive responses
        'include_citations': True,           # Always include document citations
        'include_context_summary': True      # Include context summary for large queries
    }
    
    # Enhanced prompt settings
    PROMPT_ENHANCEMENTS = {
        'use_enhanced_templates': True,      # Use enhanced prompt templates
        'include_document_metadata': True,   # Include document metadata in context
        'use_intelligent_chunking': True,   # Use intelligent chunk selection
        'optimize_for_comprehensiveness': True,  # Optimize for comprehensive responses
        'enable_cross_document_synthesis': True  # Enable synthesis across documents
    }
    
    @classmethod
    def get_context_limit(cls, query_type: str, is_enhanced: bool = True) -> int:
        """
        Get appropriate context limit based on query type and enhancement status.
        
        Args:
            query_type: Type of query being processed
            is_enhanced: Whether enhanced processing is enabled
            
        Returns:
            Appropriate context limit
        """
        if not is_enhanced:
            return cls.CONTEXT_LIMITS.get('legacy_fallback', 50)
        
        return cls.CONTEXT_LIMITS.get(query_type, cls.CONTEXT_LIMITS['general_query'])
    
    @classmethod
    def should_use_detailed_response(cls, query: str, context_size: int) -> bool:
        """
        Determine if a detailed response should be generated.
        
        Args:
            query: User query
            context_size: Size of available context
            
        Returns:
            True if detailed response should be used
        """
        query_lower = query.lower()
        
        # Always use detailed responses for summary/comprehensive queries
        comprehensive_indicators = [
            'summary', 'summarize', 'comprehensive', 'detailed', 'complete',
            'all meetings', 'overview', 'analysis', 'insights'
        ]
        
        if any(indicator in query_lower for indicator in comprehensive_indicators):
            return True
        
        # Use detailed responses for large context
        if context_size > 20:  # More than 20 chunks suggests comprehensive query
            return True
        
        # Use detailed responses for analytical queries
        analytical_indicators = ['why', 'how', 'analyze', 'explain', 'compare', 'evaluate']
        if any(indicator in query_lower for indicator in analytical_indicators):
            return True
        
        return False
    
    @classmethod
    def get_response_requirements(cls, query: str, context_size: int) -> Dict[str, Any]:
        """
        Get response requirements based on query and context.
        
        Args:
            query: User query
            context_size: Size of available context
            
        Returns:
            Dictionary of response requirements
        """
        requirements = {
            'min_length': cls.RESPONSE_QUALITY['min_response_length'],
            'include_citations': cls.RESPONSE_QUALITY['include_citations'],
            'include_examples': False,
            'include_context_summary': False,
            'response_style': 'conversational',
            'detail_level': 'standard'
        }
        
        query_lower = query.lower()
        
        # Adjust based on query type
        if cls.should_use_detailed_response(query, context_size):
            requirements.update({
                'min_length': cls.RESPONSE_QUALITY['detailed_response_threshold'],
                'include_examples': True,
                'response_style': 'comprehensive',
                'detail_level': 'detailed'
            })
        
        # Comprehensive summaries need special handling
        if any(indicator in query_lower for indicator in ['comprehensive', 'complete', 'all meetings']):
            requirements.update({
                'min_length': cls.RESPONSE_QUALITY['comprehensive_threshold'],
                'include_context_summary': True,
                'response_style': 'executive_summary',
                'detail_level': 'comprehensive'
            })
        
        # Large context indicates need for synthesis
        if context_size > 50:
            requirements['include_context_summary'] = True
        
        return requirements
    
    @classmethod
    def optimize_for_model(cls, model_name: str = None) -> Dict[str, Any]:
        """
        Optimize configuration for specific model capabilities.
        
        Args:
            model_name: Name of the LLM model being used
            
        Returns:
            Optimized configuration dictionary
        """
        # Default to GPT-4 optimization
        config = {
            'max_context_tokens': cls.MAX_CONTEXT_TOKENS,
            'max_response_tokens': cls.MAX_RESPONSE_TOKENS,
            'supports_long_context': True,
            'supports_structured_output': True,
            'optimal_chunk_size': 1000,
            'context_overlap': 200
        }
        
        # Model-specific optimizations
        if model_name and 'gpt-4' in model_name.lower():
            config.update({
                'max_context_tokens': 100000,  # GPT-4 supports very long context
                'optimal_chunk_size': 1500,
                'context_overlap': 300
            })
        elif model_name and 'gpt-3.5' in model_name.lower():
            config.update({
                'max_context_tokens': 12000,   # GPT-3.5 has smaller context
                'max_response_tokens': 4000,
                'optimal_chunk_size': 800,
                'context_overlap': 150
            })
        
        return config
    
    @classmethod
    def get_enhanced_settings(cls) -> Dict[str, Any]:
        """Get all enhanced settings for the application."""
        return {
            'context_limits': cls.CONTEXT_LIMITS,
            'response_quality': cls.RESPONSE_QUALITY,
            'prompt_enhancements': cls.PROMPT_ENHANCEMENTS,
            'token_optimization': {
                'max_context_tokens': cls.MAX_CONTEXT_TOKENS,
                'max_response_tokens': cls.MAX_RESPONSE_TOKENS,
                'token_buffer': cls.TOKEN_BUFFER
            }
        }


# Global configuration instance
enhanced_config = EnhancedLLMConfig()

# Configuration validation
def validate_configuration() -> bool:
    """Validate the enhanced configuration settings."""
    try:
        # Check that context limits are reasonable
        max_limit = max(EnhancedLLMConfig.CONTEXT_LIMITS.values())
        if max_limit > 500:  # Sanity check
            logger.warning(f"Very high context limit detected: {max_limit}")
        
        # Check token settings
        total_tokens = EnhancedLLMConfig.MAX_CONTEXT_TOKENS + EnhancedLLMConfig.MAX_RESPONSE_TOKENS
        if total_tokens > 120000:  # Beyond most model limits
            logger.warning(f"Total token allocation exceeds typical model limits: {total_tokens}")
        
        logger.info("Enhanced LLM configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


# Initialize and validate configuration on import
if not validate_configuration():
    logger.warning("Using fallback configuration due to validation errors")