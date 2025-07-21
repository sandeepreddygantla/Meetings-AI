"""
Document-related data models for Meetings AI application.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class DocumentChunk:
    """Document chunk model with AI-enhanced metadata."""
    chunk_id: str
    document_id: str
    user_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    
    # AI-enhanced metadata
    speakers: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    decisions: Optional[List[str]] = None
    action_items: Optional[List[str]] = None
    key_points: Optional[List[str]] = None
    questions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Database fields
    created_at: Optional[datetime] = None
    vector_id: Optional[int] = None


@dataclass
class MeetingDocument:
    """Complete meeting document model with metadata."""
    document_id: str
    user_id: str
    project_id: Optional[str]
    meeting_id: Optional[str]
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    file_hash: str
    content: str
    processed_content: str
    
    # AI-enhanced metadata
    summary: Optional[str] = None
    extracted_date: Optional[datetime] = None
    speakers: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    decisions: Optional[List[str]] = None
    action_items: Optional[List[str]] = None
    key_points: Optional[List[str]] = None
    questions: Optional[List[str]] = None
    sentiment_scores: Optional[Dict[str, float]] = None
    
    # Processing metadata
    chunk_count: int = 0
    processing_status: str = "pending"
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None


@dataclass
class UploadJob:
    """File upload job tracking model."""
    job_id: str
    user_id: str
    project_id: Optional[str]
    meeting_id: Optional[str]
    total_files: int
    processed_files: int = 0
    failed_files: int = 0
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class SearchResult:
    """Search result model for document queries."""
    chunk_id: str
    document_id: str
    filename: str
    content: str
    similarity_score: float
    chunk_index: int
    
    # Document metadata
    extracted_date: Optional[datetime] = None
    speakers: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    
    # Context information
    context_before: Optional[str] = None
    context_after: Optional[str] = None