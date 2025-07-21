"""
API routes initialization for Meetings AI application.
"""
import logging
from flask import Flask

from src.api.auth_routes import create_auth_blueprint
from src.api.chat_routes import create_chat_blueprint
from src.api.document_routes import create_document_blueprint

logger = logging.getLogger(__name__)


def register_all_routes(app: Flask, base_path: str, services: dict):
    """
    Register all API route blueprints.
    
    Args:
        app: Flask application instance
        base_path: Base path for all routes (e.g., '/meetingsai')
        services: Dictionary of service instances
    """
    try:
        # Create and register auth blueprint
        auth_bp = create_auth_blueprint(base_path, services['auth_service'])
        app.register_blueprint(auth_bp)
        logger.info("Auth routes registered")
        
        # Create and register chat blueprint
        chat_bp = create_chat_blueprint(base_path, services['chat_service'])
        app.register_blueprint(chat_bp)
        logger.info("Chat routes registered")
        
        # Create and register document blueprint
        doc_bp = create_document_blueprint(
            base_path, 
            services['document_service'], 
            services['upload_service']
        )
        app.register_blueprint(doc_bp)
        logger.info("Document routes registered")
        
        logger.info(f"All API routes registered with base path: {base_path}")
        
    except Exception as e:
        logger.error(f"Error registering routes: {e}")
        raise