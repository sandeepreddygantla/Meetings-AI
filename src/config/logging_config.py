"""
Logging configuration for Meetings AI application.
Provides centralized logging with production-optimized filtering.
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, Optional
import re


class ProductionFilter(logging.Filter):
    """Filter to eliminate noise and optimize log output for production."""
    
    # Patterns to completely ignore (startup noise)
    IGNORE_PATTERNS = [
        r"Created/verified directory:",
        r"Database path resolved to:",
        r"Vector index path resolved to:",
        r"Using absolute path:",
        r"AI clients configured for lazy loading",
        r"Connection pool created successfully", 
        r"Application setup completed successfully",
        r"Loaded existing FAISS index with \d+ vectors",
        r"Flask app created with base path:",
        r"All services initialized successfully",
        r"All API routes registered",
        r"Initializing services",
        r"Processor initialized",
        r"Asset optimizer initialized",
        r"Background processor initialized",
        r"Flask-Login configured",
        r"Core routes registered",
        r"Chunk metadata rebuild not needed",
    ]
    
    # Static file patterns to ignore (HTTP noise) 
    STATIC_PATTERNS = [
        r"GET.*\.(css|js|png|jpg|ico|gif|woff|ttf)",
        r"304.*static/",
        r"favicon\.png",
        r"Optum-logo\.png",
        r"\[36m.*\[0m",  # 304 Not Modified responses
    ]
    
    # Repetitive database patterns to reduce
    DB_VERBOSE_PATTERNS = [
        r"Step \d+:",
        r"Looking for \d+ chunks in database",
        r"Database query returned \d+ rows",
        r"Retrieved \d+ chunks from SQLite",
        r"User ID filtering results:",
        r"Passed: \d+ chunks",
        r"Filtered out: \d+ chunks", 
        r"No metadata filters to apply",
        r"Sorting and limiting results",
        r"Result \d+: Chunk.*Score:",
        r"Extracted project_id from folder_path:",
        r"Including chunk .* - project_id match:",
        r"Including chunk .* in results",
        r"Total search results after filtering:",
        r"\[STATS\] Metadata filtering:",
    ]
    
    def __init__(self):
        super().__init__()
        self.ignore_regex = re.compile('|'.join(self.IGNORE_PATTERNS), re.IGNORECASE)
        self.static_regex = re.compile('|'.join(self.STATIC_PATTERNS), re.IGNORECASE)
        self.db_verbose_regex = re.compile('|'.join(self.DB_VERBOSE_PATTERNS), re.IGNORECASE)
        
        # Track repetitive patterns to avoid spam
        self.seen_patterns = {}
    
    def filter(self, record):
        try:
            message = record.getMessage()
            
            # Debug: Track None messages to identify source
            if message.strip() == "None":
                # For debugging - uncomment next line to see what's logging None  
                # print(f"DEBUG: None message from {record.name}:{record.lineno} in {record.funcName}")
                return False
            
            # Skip empty or whitespace-only messages
            if not message.strip():
                return False
            
            # Completely ignore startup noise
            if self.ignore_regex.search(message):
                return False
            
            # Ignore static file HTTP requests
            if self.static_regex.search(message):
                return False
                
            # Reduce verbose database logging
            if self.db_verbose_regex.search(message):
                return False
        except Exception as e:
            # If there's an error processing the message, allow it through
            # For debugging - uncomment next line
            # print(f"DEBUG: Filter error: {e}")
            pass
        
        # Skip repetitive session loading (keep only first per hour)
        if "User loaded successfully" in message:
            user_pattern = f"session_load_{getattr(record, 'user_id', 'unknown')}"
            current_hour = record.created // 3600  # Hour bucket
            pattern_key = f"{user_pattern}_{current_hour}"
            
            if pattern_key in self.seen_patterns:
                return False  # Skip repetitive loads
            else:
                self.seen_patterns[pattern_key] = True
                # Clean old entries to prevent memory growth
                if len(self.seen_patterns) > 1000:
                    self.seen_patterns.clear()
        
        return True


class ComponentTagFormatter(logging.Formatter):
    """Formatter that adds component tags for unified logging."""
    
    # Component mapping based on logger names
    COMPONENT_MAP = {
        'src.services.auth_service': 'AUTH',
        'src.api.auth_routes': 'AUTH',
        'src.services.chat_service': 'CHAT', 
        'src.api.chat_routes': 'CHAT',
        'src.services.document_service': 'DOCS',
        'src.services.upload_service': 'UPLOAD',
        'src.api.document_routes': 'UPLOAD',
        'src.database.manager': 'SEARCH',
        'src.database.sqlite_operations': 'DB',
        'src.database.vector_operations': 'VECTOR',
        'src.ai.query_processor': 'AI',
        'src.ai.context_manager': 'AI',
        'src.ai.enhanced_prompts': 'AI',
        'meeting_processor': 'AI',
        'werkzeug': 'HTTP',
        '__main__': 'SYSTEM',
        'flask_app': 'SYSTEM',
        'src.services.background_processor': 'SYSTEM',
    }
    
    def format(self, record):
        # Add component tag
        logger_name = record.name
        component = self.COMPONENT_MAP.get(logger_name, 'SYSTEM')
        
        # Get user context
        user_id = getattr(record, 'user_id', 'SYSTEM')
        if user_id != 'SYSTEM':
            user_part = f" | USER:{user_id}"
        else:
            user_part = ""
        
        # Create optimized format
        formatted_time = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        
        # Optimize message content
        message = record.getMessage()
        
        # Final safety check - skip None messages at formatter level
        if message.strip() in ["None", "null", ""]:
            return None  # Skip logging this message
        
        message = self._optimize_message(message, component)
        
        # If message was optimized to None, skip it
        if message is None:
            return None
        
        return f"{formatted_time} [{record.levelname}] {component}{user_part} | {message}"
    
    def _optimize_message(self, message: str, component: str) -> str:
        """Optimize message content based on component."""
        
        # Optimize database search messages
        if component == 'SEARCH' and 'FINAL ENHANCED SEARCH RESULTS' in message:
            # Extract key info from verbose search completion
            if 'Final results returned:' in message:
                results = re.search(r'Final results returned: (\d+)', message)
                if results:
                    return f"Found: {results.group(1)} results"
        
        # Optimize HTTP messages
        if component == 'HTTP':
            # Simplify HTTP logs - remove color codes and extract key info
            clean_msg = re.sub(r'\[[\d;]+m', '', message)  # Remove ANSI codes
            
            # Extract method, path, status, timing if available
            http_match = re.search(r'(GET|POST|PUT|DELETE)\s+([^\s]+).*?(\d{3})', clean_msg)
            if http_match:
                method, path, status = http_match.groups()
                # Only log API calls, skip static files (already filtered)
                if '/api/' in path:
                    return f"{method} {path} | {status}"
                else:
                    return f"{method} {path}"
            
        # Optimize authentication messages  
        if component == 'AUTH':
            if 'Loading user for session:' in message:
                return None  # Skip, will be filtered out
            if 'User authenticated successfully:' in message:
                user = message.split(':')[-1].strip()
                return f"Login successful"
            if 'User logged out:' in message:
                return f"Logout"
        
        # Optimize database messages
        if component == 'DB' and 'Calculated date range' in message:
            date_match = re.search(r"for '(\w+)':", message)
            if date_match:
                return f"Date filter: {date_match.group(1)}"
        
        return message


class LoggingConfig:
    """Configuration class for application logging setup."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = os.path.join(log_dir, "meetingsai.log")
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
    
    def setup_logging(self) -> logging.Logger:
        """Set up single unified logging for entire application."""
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()  # Remove all existing handlers
        root_logger.setLevel(self.log_level)
        
        # Create single unified handler with rotation
        handler = TimedRotatingFileHandler(
            self.log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        
        # Add production filter and formatter
        handler.addFilter(ProductionFilter())
        handler.setFormatter(ComponentTagFormatter())
        handler.setLevel(self.log_level)
        
        # Add console handler for development only
        if os.getenv('FLASK_ENV') == 'development':
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)  # Only warnings+ to console
            console_handler.setFormatter(ComponentTagFormatter())
            console_handler.addFilter(ProductionFilter())
            root_logger.addHandler(console_handler)
        
        # Add unified handler to root logger
        root_logger.addHandler(handler)
        
        # Configure specific loggers to use root handler
        self._configure_module_loggers()
        
        return root_logger
    
    def _configure_module_loggers(self):
        """Configure all module loggers to use unified system."""
        
        # List of all loggers used in the application
        logger_names = [
            '__main__', 'flask_app', 'werkzeug',
            'src.services.auth_service', 'src.api.auth_routes',
            'src.services.chat_service', 'src.api.chat_routes', 
            'src.services.document_service', 'src.services.upload_service',
            'src.api.document_routes', 'src.database.manager',
            'src.database.sqlite_operations', 'src.database.vector_operations',
            'src.ai.query_processor', 'src.ai.context_manager',
            'src.ai.enhanced_prompts', 'src.ai.embeddings',
            'meeting_processor', 'src.services.background_processor',
            'src.utils.asset_optimizer', 'src.utils.helpers',
        ]
        
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()  # Remove any existing handlers
            logger.propagate = True   # Use root logger's handlers
            logger.setLevel(self.log_level)


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Set up application logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Configured root logger
    """
    config = LoggingConfig(log_dir, log_level)
    return config.setup_logging()


# Export for easy importing
__all__ = ['LoggingConfig', 'setup_logging', 'ProductionFilter', 'ComponentTagFormatter']