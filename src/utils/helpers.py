"""
Utility helper functions for Meetings AI application.
"""
import os
import hashlib
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: str) -> Optional[str]:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File hash as hex string or None if error
    """
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash for {file_path}: {e}")
        return None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe filesystem storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    try:
        # Use werkzeug's secure_filename as base
        safe_name = secure_filename(filename)
        
        # Additional sanitization
        safe_name = re.sub(r'[^\w\-_\.]', '_', safe_name)
        safe_name = re.sub(r'_+', '_', safe_name)  # Replace multiple underscores
        
        # Ensure it's not empty
        if not safe_name:
            safe_name = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return safe_name
        
    except Exception as e:
        logger.error(f"Error sanitizing filename {filename}: {e}")
        return f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_safe_directory_name(name: str) -> str:
    """
    Create a safe directory name from a project/user name.
    
    Args:
        name: Original name
        
    Returns:
        Safe directory name
    """
    try:
        # Replace spaces and problematic characters
        safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_name = re.sub(r'[^\w\-_]', '_', safe_name)
        safe_name = re.sub(r'_+', '_', safe_name)  # Replace multiple underscores
        safe_name = safe_name.strip('_')  # Remove leading/trailing underscores
        
        # Ensure it's not empty
        if not safe_name:
            safe_name = f"folder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return safe_name
        
    except Exception as e:
        logger.error(f"Error creating safe directory name from {name}: {e}")
        return f"folder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure a directory exists, create it if necessary.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    try:
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
        
    except Exception as e:
        logger.error(f"Error formatting file size: {e}")
        return f"{size_bytes} B"


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Validate file extension against allowed extensions.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (e.g., ['.txt', '.pdf'])
        
    Returns:
        True if extension is allowed
    """
    try:
        if not filename:
            return False
        
        # Get file extension
        _, ext = os.path.splitext(filename.lower())
        
        # Normalize allowed extensions
        normalized_allowed = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                            for ext in allowed_extensions]
        
        return ext in normalized_allowed
        
    except Exception as e:
        logger.error(f"Error validating file extension for {filename}: {e}")
        return False


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date from filename using common patterns.
    
    Args:
        filename: Filename to analyze
        
    Returns:
        Extracted datetime or None if not found
    """
    try:
        # Common date patterns in filenames
        patterns = [
            r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})',  # YYYY-MM-DD or YYYY_MM_DD
            r'(\d{1,2})[-_](\d{1,2})[-_](\d{4})',  # MM-DD-YYYY or MM_DD_YYYY
            r'(\d{4})(\d{2})(\d{2})',              # YYYYMMDD
            r'(\d{2})(\d{2})(\d{4})',              # MMDDYYYY
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            if matches:
                match = matches[0]
                try:
                    # Try different date formats
                    if len(match[0]) == 4:  # Year first
                        date = datetime.strptime(f"{match[0]}-{match[1]}-{match[2]}", "%Y-%m-%d")
                    else:  # Month/day first
                        date = datetime.strptime(f"{match[2]}-{match[0]}-{match[1]}", "%Y-%m-%d")
                    
                    return date
                except ValueError:
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting date from filename {filename}: {e}")
        return None


def clean_text_content(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    try:
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters but keep basic formatting
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
        
    except Exception as e:
        logger.error(f"Error cleaning text content: {e}")
        return text


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with metadata
    """
    try:
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + chunk_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                # Look for space within overlap distance
                for i in range(end, max(end - 100, start), -1):
                    if text[i].isspace():
                        end = i
                        break
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    'content': chunk_text,
                    'chunk_index': chunk_index,
                    'start_char': start,
                    'end_char': end,
                    'length': len(chunk_text)
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + chunk_size - chunk_overlap, end)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        return [{'content': text, 'chunk_index': 0, 'start_char': 0, 'end_char': len(text), 'length': len(text)}]


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    try:
        import uuid
        unique_part = str(uuid.uuid4()).replace('-', '')[:12]
        timestamp_part = datetime.now().strftime('%Y%m%d%H%M%S')
        
        if prefix:
            return f"{prefix}_{timestamp_part}_{unique_part}"
        else:
            return f"{timestamp_part}_{unique_part}"
            
    except Exception as e:
        logger.error(f"Error generating unique ID: {e}")
        return f"{prefix}_{int(datetime.now().timestamp())}" if prefix else str(int(datetime.now().timestamp()))