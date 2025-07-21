"""
Configuration management for Meetings AI application.
Provides dynamic path configuration and environment-based settings.
"""
import os
import secrets
from datetime import timedelta
from typing import Optional


def get_base_path() -> str:
    """
    Get base path from environment or default to /meetingsai.
    This ensures consistent routing regardless of deployment environment.
    """
    return os.environ.get('BASE_PATH', '/meetingsai')


def get_static_url() -> str:
    """Get static URL path with base path prefix."""
    return f'{get_base_path()}/static'


def get_upload_folder() -> str:
    """Get upload folder path."""
    return os.environ.get('UPLOAD_FOLDER', 'uploads')


def get_max_file_size() -> int:
    """Get maximum file size in bytes."""
    return int(os.environ.get('MAX_FILE_SIZE', 100 * 1024 * 1024))  # 100MB default


def get_secret_key() -> str:
    """Get Flask secret key from environment or generate one."""
    return os.environ.get('SECRET_KEY', secrets.token_hex(32))


def get_database_url() -> str:
    """Get database URL."""
    return os.environ.get('DATABASE_URL', 'sqlite:///meeting_documents.db')


def get_session_lifetime() -> timedelta:
    """Get session lifetime."""
    days = int(os.environ.get('SESSION_LIFETIME_DAYS', 30))
    return timedelta(days=days)


class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = get_secret_key()
    
    # Path settings
    BASE_PATH = get_base_path()
    STATIC_URL = get_static_url()
    UPLOAD_FOLDER = get_upload_folder()
    MAX_CONTENT_LENGTH = get_max_file_size()
    
    # Database settings
    DATABASE_URL = get_database_url()
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = get_session_lifetime()
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    REMEMBER_COOKIE_DURATION = get_session_lifetime()
    REMEMBER_COOKIE_SECURE = os.environ.get('REMEMBER_COOKIE_SECURE', 'False').lower() == 'true'
    REMEMBER_COOKIE_HTTPONLY = True


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True


def get_config(config_name: Optional[str] = None) -> Config:
    """
    Get configuration class based on environment.
    
    Args:
        config_name: Configuration name ('development', 'production', or None for auto-detect)
    
    Returns:
        Configuration class instance
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
    }
    
    return config_map.get(config_name, DevelopmentConfig)()


def setup_flask_config(app, base_path: Optional[str] = None):
    """
    Setup Flask application configuration.
    
    Args:
        app: Flask application instance
        base_path: Base path for the application (optional)
    """
    config = get_config()
    
    # Override base path if provided
    if base_path:
        config.BASE_PATH = base_path
        config.STATIC_URL = f'{base_path}/static'
    
    # Apply configuration to Flask app
    app.config.from_object(config)
    
    # Set additional runtime configuration
    app.config.update({
        'BASE_PATH': config.BASE_PATH,
        'STATIC_URL': config.STATIC_URL,
    })
    
    return config