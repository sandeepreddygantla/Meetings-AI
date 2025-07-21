"""
Input validation utilities for Meetings AI application.
"""
import re
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not email or not email.strip():
            return False, "Email is required"
        
        email = email.strip()
        
        # Basic email validation pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            return False, "Invalid email format"
        
        if len(email) > 255:
            return False, "Email is too long"
        
        return True, "Valid email"
        
    except Exception as e:
        logger.error(f"Email validation error: {e}")
        return False, "Email validation failed"


def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validate username format and requirements.
    
    Args:
        username: Username to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not username or not username.strip():
            return False, "Username is required"
        
        username = username.strip()
        
        # Length check
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if len(username) > 50:
            return False, "Username must be less than 50 characters"
        
        # Character validation
        if not re.match(r'^[a-zA-Z0-9_.-]+$', username):
            return False, "Username can only contain letters, numbers, dots, hyphens, and underscores"
        
        # Cannot start with special characters
        if username[0] in '._-':
            return False, "Username cannot start with special characters"
        
        return True, "Valid username"
        
    except Exception as e:
        logger.error(f"Username validation error: {e}")
        return False, "Username validation failed"


def validate_password(password: str, confirm_password: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate password strength and requirements.
    
    Args:
        password: Password to validate
        confirm_password: Optional password confirmation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not password:
            return False, "Password is required"
        
        # Length check
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        if len(password) > 128:
            return False, "Password must be less than 128 characters"
        
        # Check for confirmation match
        if confirm_password is not None and password != confirm_password:
            return False, "Passwords do not match"
        
        # Basic strength requirements
        has_letter = re.search(r'[a-zA-Z]', password)
        has_digit = re.search(r'\d', password)
        
        if not has_letter:
            return False, "Password must contain at least one letter"
        
        # Optional: Require digit for stronger passwords
        # if not has_digit:
        #     return False, "Password must contain at least one number"
        
        return True, "Valid password"
        
    except Exception as e:
        logger.error(f"Password validation error: {e}")
        return False, "Password validation failed"


def validate_full_name(full_name: str) -> Tuple[bool, str]:
    """
    Validate full name format.
    
    Args:
        full_name: Full name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not full_name or not full_name.strip():
            return False, "Full name is required"
        
        full_name = full_name.strip()
        
        # Length check
        if len(full_name) < 2:
            return False, "Full name must be at least 2 characters long"
        
        if len(full_name) > 100:
            return False, "Full name must be less than 100 characters"
        
        # Basic character validation (allow letters, spaces, common punctuation)
        if not re.match(r'^[a-zA-Z\s\'\-\.]+$', full_name):
            return False, "Full name can only contain letters, spaces, apostrophes, hyphens, and periods"
        
        return True, "Valid full name"
        
    except Exception as e:
        logger.error(f"Full name validation error: {e}")
        return False, "Full name validation failed"


def validate_project_name(project_name: str) -> Tuple[bool, str]:
    """
    Validate project name format.
    
    Args:
        project_name: Project name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not project_name or not project_name.strip():
            return False, "Project name is required"
        
        project_name = project_name.strip()
        
        # Length check
        if len(project_name) < 1:
            return False, "Project name cannot be empty"
        
        if len(project_name) > 100:
            return False, "Project name must be less than 100 characters"
        
        return True, "Valid project name"
        
    except Exception as e:
        logger.error(f"Project name validation error: {e}")
        return False, "Project name validation failed"


def validate_file_upload(files: List[Any], allowed_extensions: Optional[List[str]] = None, max_file_size: int = 100 * 1024 * 1024) -> Tuple[bool, List[str]]:
    """
    Validate file upload requirements.
    
    Args:
        files: List of uploaded files
        allowed_extensions: List of allowed file extensions
        max_file_size: Maximum file size in bytes
        
    Returns:
        Tuple of (all_valid, list_of_errors)
    """
    try:
        if not files:
            return False, ["No files provided"]
        
        if allowed_extensions is None:
            allowed_extensions = ['.txt', '.docx', '.pdf']
        
        errors = []
        
        for file in files:
            if not file or not file.filename:
                errors.append("Empty file detected")
                continue
            
            filename = file.filename
            
            # Check file extension
            valid_ext = any(filename.lower().endswith(ext.lower()) for ext in allowed_extensions)
            if not valid_ext:
                errors.append(f"File '{filename}' has unsupported format. Allowed: {', '.join(allowed_extensions)}")
            
            # Check file size if available
            if hasattr(file, 'content_length') and file.content_length:
                if file.content_length > max_file_size:
                    errors.append(f"File '{filename}' is too large. Maximum size: {max_file_size // 1024 // 1024}MB")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"File upload validation error: {e}")
        return False, ["File validation failed"]


def validate_search_query(query: str) -> Tuple[bool, str]:
    """
    Validate search query format.
    
    Args:
        query: Search query to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not query or not query.strip():
            return False, "Search query cannot be empty"
        
        query = query.strip()
        
        # Length check
        if len(query) < 2:
            return False, "Search query must be at least 2 characters long"
        
        if len(query) > 1000:
            return False, "Search query is too long"
        
        return True, "Valid search query"
        
    except Exception as e:
        logger.error(f"Search query validation error: {e}")
        return False, "Search query validation failed"


def validate_registration_data(data: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate complete registration data.
    
    Args:
        data: Registration data dictionary
        
    Returns:
        Tuple of (is_valid, error_messages_dict)
    """
    try:
        errors = {}
        
        # Validate username
        username = data.get('username', '')
        is_valid, error_msg = validate_username(username)
        if not is_valid:
            errors['username'] = error_msg
        
        # Validate email
        email = data.get('email', '')
        is_valid, error_msg = validate_email(email)
        if not is_valid:
            errors['email'] = error_msg
        
        # Validate full name
        full_name = data.get('full_name', '')
        is_valid, error_msg = validate_full_name(full_name)
        if not is_valid:
            errors['full_name'] = error_msg
        
        # Validate password
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        is_valid, error_msg = validate_password(password, confirm_password)
        if not is_valid:
            errors['password'] = error_msg
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"Registration data validation error: {e}")
        return False, {'general': 'Validation failed'}


def validate_login_data(data: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate login data.
    
    Args:
        data: Login data dictionary
        
    Returns:
        Tuple of (is_valid, error_messages_dict)
    """
    try:
        errors = {}
        
        # Validate username
        username = data.get('username', '')
        if not username or not username.strip():
            errors['username'] = 'Username is required'
        
        # Validate password
        password = data.get('password', '')
        if not password:
            errors['password'] = 'Password is required'
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"Login data validation error: {e}")
        return False, {'general': 'Validation failed'}


def sanitize_input(input_string: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input string.
    
    Args:
        input_string: String to sanitize
        max_length: Optional maximum length
        
    Returns:
        Sanitized string
    """
    try:
        if not input_string:
            return ""
        
        # Strip whitespace
        sanitized = input_string.strip()
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Limit length if specified
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length].strip()
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Input sanitization error: {e}")
        return ""