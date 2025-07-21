"""
User-related data models for Meetings AI application.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from flask_login import UserMixin


@dataclass
class User(UserMixin):
    """User model with authentication support."""
    user_id: str
    username: str
    email: str
    full_name: str
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    
    def get_id(self):
        """Return user ID for Flask-Login."""
        return self.user_id
    
    def __post_init__(self):
        """Set id attribute for Flask-Login compatibility."""
        self.id = self.user_id


@dataclass
class Project:
    """Project model for organizing meetings and documents."""
    project_id: str
    user_id: str
    project_name: str
    description: str
    created_at: datetime
    is_active: bool = True


@dataclass
class Meeting:
    """Meeting model for organizing documents."""
    meeting_id: str
    project_id: str
    user_id: str
    meeting_name: str
    meeting_date: Optional[datetime] = None
    created_at: datetime = None
    is_active: bool = True


@dataclass
class SessionUser(UserMixin):
    """Minimal user data for session storage."""
    user_id: str
    username: str
    email: str
    full_name: str
    is_active: bool = True
    is_authenticated: bool = True
    is_anonymous: bool = False
    
    def get_id(self):
        """Return user ID for Flask-Login."""
        return self.user_id
    
    def __post_init__(self):
        """Set id attribute for Flask-Login compatibility."""
        self.id = self.user_id