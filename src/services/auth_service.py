"""
Authentication service for Meetings AI application.
Handles user authentication, registration, and session management.
"""
import logging
import bcrypt
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from flask import session
from flask_login import login_user, logout_user, current_user

from src.models.user import User, SessionUser
from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class AuthService:
    """Service for handling authentication operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize authentication service.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
    
    def register_user(self, username: str, email: str, full_name: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """
        Register a new user.
        
        Args:
            username: Username for the new user
            email: Email address
            full_name: Full name of the user
            password: Plain text password
            
        Returns:
            Tuple of (success, message, user_id)
        """
        try:
            # Validate input
            if not all([username, email, full_name, password]):
                return False, 'All fields are required', None
            
            username = username.strip()
            email = email.strip()
            full_name = full_name.strip()
            
            if len(password) < 6:
                return False, 'Password must be at least 6 characters', None
            
            # Hash password
            try:
                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            except Exception as e:
                logger.error(f"Password hashing failed: {e}")
                return False, 'Password processing failed', None
            
            # Create user
            try:
                user_id = self.db_manager.create_user(username, email, full_name, password_hash)
                logger.info(f"User created with ID: {user_id}")
            except ValueError as e:
                return False, str(e), None
            except Exception as e:
                logger.error(f"User creation failed: {e}")
                return False, f'User creation failed: {str(e)}', None
            
            # Create default project
            try:
                self.db_manager.create_project(user_id, "Default Project", "Default project for meetings")
            except Exception as e:
                logger.error(f"Default project creation failed: {e}")
                # Don't fail registration if project creation fails
            
            logger.info(f"New user registered: {username} ({user_id})")
            return True, 'Registration successful! Please log in.', user_id
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False, f'Registration failed: {str(e)}', None
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str, Optional[User]]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Tuple of (success, message, user_object)
        """
        try:
            if not username or not password:
                return False, 'Username and password are required', None
            
            username = username.strip()
            
            # Get user from database
            try:
                user = self.db_manager.get_user_by_username(username)
            except Exception as e:
                logger.error(f"Database error during authentication: {e}")
                return False, 'System error - please try again later', None
            
            if not user:
                logger.warning(f"User not found: {username}")
                return False, 'Invalid username or password', None
            
            # Check password
            try:
                if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                    logger.warning(f"Invalid password for user: {username}")
                    return False, 'Invalid username or password', None
            except Exception as e:
                logger.error(f"Password verification failed: {e}")
                return False, 'Authentication failed', None
            
            logger.info(f"User authenticated successfully: {username}")
            return True, 'Login successful', user
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, f'Login failed: {str(e)}', None
    
    def login_user_session(self, user: User, remember: bool = True) -> bool:
        """
        Login user and create session.
        
        Args:
            user: User object
            remember: Whether to remember the user
            
        Returns:
            Success status
        """
        try:
            # Create Flask-Login session
            flask_user = SessionUser(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                full_name=user.full_name
            )
            
            login_user(flask_user, remember=remember)
            session.permanent = True
            
            # Update last login
            try:
                self.db_manager.update_user_last_login(user.user_id)
            except Exception as e:
                logger.error(f"Failed to update last login: {e}")
                # Don't fail login for this
            
            return True
            
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return False
    
    def logout_user_session(self) -> Tuple[bool, str]:
        """
        Logout current user and cleanup session.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            username = current_user.username if current_user.is_authenticated else "Unknown"
            logout_user()
            logger.info(f"User logged out: {username}")
            return True, 'Logged out successfully'
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False, f'Logout failed: {str(e)}'
    
    def get_current_user_data(self) -> Optional[Dict[str, Any]]:
        """
        Get current user data for API responses.
        
        Returns:
            User data dictionary or None
        """
        try:
            if not current_user.is_authenticated:
                return None
            
            return {
                'user_id': current_user.user_id,
                'username': current_user.username,
                'email': current_user.email,
                'full_name': current_user.full_name
            }
        except Exception as e:
            logger.error(f"Error getting current user data: {e}")
            return None
    
    def validate_user_access(self, user_id: str) -> bool:
        """
        Validate that current user has access to the specified user_id.
        
        Args:
            user_id: User ID to validate access for
            
        Returns:
            True if access is valid
        """
        try:
            if not current_user.is_authenticated:
                return False
            
            return current_user.user_id == user_id
        except Exception as e:
            logger.error(f"User access validation error: {e}")
            return False
    
    def load_user_for_session(self, user_id: str) -> Optional[SessionUser]:
        """
        Load user for Flask-Login user_loader.
        
        Args:
            user_id: User ID to load
            
        Returns:
            SessionUser object or None
        """
        try:
            logger.info(f"Loading user for session: {user_id}")
            
            try:
                user = self.db_manager.get_user_by_id(user_id)
                if user:
                    logger.info(f"User loaded successfully: {user.username}")
                    return SessionUser(
                        user_id=user.user_id,
                        username=user.username,
                        email=user.email,
                        full_name=user.full_name
                    )
                else:
                    logger.warning(f"User not found in database: {user_id}")
                    return None
                    
            except Exception as db_error:
                logger.error(f"Database error in user loader: {db_error}")
                # Fallback to minimal user from session ID format
                if "_" in user_id:
                    username = user_id.split("_")[-1]
                    return SessionUser(
                        user_id=user_id,
                        username=username,
                        email=f"{username}@company.com",
                        full_name=username
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error loading user {user_id}: {e}")
            return None