"""
Database configuration for Meetings AI application.
Handles SQLite session interface for IIS/WFASTCGI compatibility.
"""
import os
import sqlite3
import logging
import pickle
from datetime import datetime
from uuid import uuid4
from flask.sessions import SessionInterface, SessionMixin
from typing import Optional, Dict, Any


class SqliteSession(dict, SessionMixin):
    """Custom session class for SQLite storage."""
    pass


class SqliteSessionInterface(SessionInterface):
    """
    SQLite-based session interface for WFASTCGI compatibility.
    Provides persistent session storage for IIS deployments.
    """
    
    def __init__(self, db_path: str = 'sessions.db'):
        """
        Initialize session interface.
        
        Args:
            db_path: Path to session database file
        """
        self.db_path = db_path
        self.session_cookie_name = 'session'
        self._init_db()
    
    def _init_db(self):
        """Initialize the sessions table."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    data BLOB,
                    expiry TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
            logging.info("Session database initialized successfully")
        except Exception as e:
            logging.error(f"Session database initialization error: {e}")
    
    def open_session(self, app, request):
        """
        Load session from database.
        
        Args:
            app: Flask application instance
            request: Flask request object
            
        Returns:
            Session object
        """
        session = SqliteSession()
        
        try:
            sid = request.cookies.get(self.session_cookie_name)
            if not sid:
                logging.debug("No session ID in cookies, returning empty session")
                return session
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT data FROM sessions WHERE id = ? AND expiry > ?', 
                (sid, datetime.now())
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                try:
                    data = pickle.loads(result[0])
                    session.update(data)
                    logging.debug(f"Loaded session data for ID: {sid}")
                except Exception as pickle_error:
                    logging.error(f"Session data unpickling error: {pickle_error}")
            else:
                logging.debug(f"No valid session found for ID: {sid}")
                
        except Exception as e:
            logging.error(f"Session load error: {e}")
        
        return session
    
    def save_session(self, app, session, response):
        """
        Save session to database.
        
        Args:
            app: Flask application instance
            session: Session object
            response: Flask response object
        """
        try:
            domain = self.get_cookie_domain(app)
            path = self.get_cookie_path(app)
            
            # Handle empty or unmodified sessions
            if not session or not getattr(session, 'modified', True):
                if hasattr(session, 'modified') and session.modified:
                    response.delete_cookie(
                        self.session_cookie_name, 
                        domain=domain, 
                        path=path
                    )
                return
            
            # Get or generate session ID
            sid = None
            try:
                from flask import request as flask_request
                if flask_request:
                    sid = flask_request.cookies.get(self.session_cookie_name)
            except:
                pass
                
            if not sid:
                sid = str(uuid4())
            
            # Calculate expiry
            expiry = datetime.now() + app.permanent_session_lifetime
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            data = pickle.dumps(dict(session))
            conn.execute(
                'INSERT OR REPLACE INTO sessions (id, data, expiry) VALUES (?, ?, ?)',
                (sid, data, expiry)
            )
            conn.commit()
            conn.close()
            
            # Set cookie
            response.set_cookie(
                self.session_cookie_name, 
                sid,
                expires=expiry, 
                httponly=True,
                domain=domain, 
                path=path, 
                secure=app.config.get('SESSION_COOKIE_SECURE', False)
            )
            
            logging.debug(f"Session saved successfully with ID: {sid}")
            
        except Exception as e:
            logging.error(f"Session save error: {e}")
    
    def get_cookie_name(self, app):
        """Get the session cookie name."""
        return self.session_cookie_name


def setup_database_session(app):
    """
    Setup database-backed session interface for Flask app.
    
    Args:
        app: Flask application instance
    """
    app.session_interface = SqliteSessionInterface()
    return app.session_interface


def ensure_directories_exist():
    """Ensure required directories exist for the application."""
    directories = [
        'uploads', 'temp', 'meeting_documents', 
        'logs', 'backups', 'templates', 'static'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created/verified directory: {directory}")
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {e}")


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration settings.
    
    Returns:
        Database configuration dictionary
    """
    return {
        'session_db_path': os.environ.get('SESSION_DB_PATH', 'sessions.db'),
        'main_db_path': os.environ.get('MAIN_DB_PATH', 'meeting_documents.db'),
        'vector_index_path': os.environ.get('VECTOR_INDEX_PATH', 'vector_index.faiss'),
    }