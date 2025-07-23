"""
Enhanced AI module for Meetings AI application.
Provides advanced context management and response generation capabilities.
"""

from .enhanced_prompts import EnhancedPromptManager
from .context_manager import EnhancedContextManager, QueryContext

__all__ = ['EnhancedPromptManager', 'EnhancedContextManager', 'QueryContext']