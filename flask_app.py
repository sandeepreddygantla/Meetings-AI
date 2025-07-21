"""
Flask application for Meetings AI - Refactored with modular architecture.
Maintains IIS compatibility while providing clean separation of concerns.
"""
from flask import Flask, render_template, redirect, session
from flask_login import LoginManager, login_required
import os
import logging

# Import configuration
from src.config.settings import setup_flask_config, get_base_path
from src.config.database import setup_database_session, ensure_directories_exist

# Import AI client initialization
from src.ai.llm_client import initialize_ai_clients

# Import database manager
from src.database.manager import DatabaseManager

# Import services
from src.services.auth_service import AuthService
from src.services.chat_service import ChatService
from src.services.document_service import DocumentService
from src.services.upload_service import UploadService

# Import API routes
from src.api import register_all_routes

# Import existing processor for backwards compatibility
try:
    from meeting_processor import EnhancedMeetingDocumentProcessor
except ImportError as e:
    logging.error(f"Failed to import meeting_processor: {e}")
    EnhancedMeetingDocumentProcessor = None

# Ensure logs directory exists
import os
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/flask_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
BASE_PATH = get_base_path()  # Dynamic base path from configuration
app = None
db_manager = None
services = {}
processor = None
_application_initialized = False


def create_flask_app():
    """Create and configure Flask application."""
    global app
    
    # Create Flask app with dynamic paths
    app = Flask(__name__, 
                static_url_path=f'{BASE_PATH}/static',
                template_folder='templates')
    
    # Setup configuration
    setup_flask_config(app, BASE_PATH)
    
    # Setup database session interface for IIS compatibility
    setup_database_session(app)
    
    logger.info(f"Flask app created with base path: {BASE_PATH}")
    return app


def initialize_services():
    """Initialize all services and dependencies."""
    global db_manager, services, processor
    
    try:
        logger.info("Initializing services...")
        
        # Initialize AI clients (following instructions.md)
        if not initialize_ai_clients():
            logger.error("Failed to initialize AI clients")
            # Continue anyway - some functionality may be limited
        
        # Initialize database manager
        db_manager = DatabaseManager()
        logger.info("Database manager initialized")
        
        # Initialize processor for backwards compatibility
        if EnhancedMeetingDocumentProcessor:
            try:
                processor = EnhancedMeetingDocumentProcessor(chunk_size=1000, chunk_overlap=200)
                logger.info("Processor initialized for backwards compatibility")
            except Exception as e:
                logger.error(f"Failed to initialize processor: {e}")
                processor = None
        
        # Initialize services
        services['auth_service'] = AuthService(db_manager)
        services['chat_service'] = ChatService(db_manager, processor)
        services['document_service'] = DocumentService(db_manager, processor)
        services['upload_service'] = UploadService(db_manager, services['document_service'])
        
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False


def setup_flask_login():
    """Setup Flask-Login configuration."""
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = f'{BASE_PATH}/login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        """Load user for Flask-Login."""
        return services['auth_service'].load_user_for_session(user_id)
    
    logger.info("Flask-Login configured")


def register_core_routes():
    """Register core application routes."""
    
    @app.route('/')
    def root():
        """Root redirect to base path."""
        return redirect(f'{BASE_PATH}/')
    
    @app.route(f'{BASE_PATH}/')
    @app.route(f'{BASE_PATH}')
    def index():
        """Main chat interface."""
        try:
            return render_template('chat.html', config={
                'basePath': BASE_PATH,
                'staticPath': f'{BASE_PATH}/static'
            })
        except Exception as e:
            logger.error(f"Error rendering chat.html: {e}")
            return f"Error loading chat interface: {str(e)}", 500
    
    @app.route(f'{BASE_PATH}/api/refresh', methods=['POST'])
    @login_required
    def refresh_system():
        """Refresh the system."""
        try:
            logger.info("System refresh requested")
            if processor:
                processor.refresh_clients()
                logger.info("System refreshed successfully")
                return {'success': True, 'message': 'System refreshed successfully'}
            else:
                logger.error("Processor not initialized for refresh")
                return {'success': False, 'error': 'System not initialized'}, 500
        except Exception as e:
            logger.error(f"Refresh error: {e}")
            return {'success': False, 'error': str(e)}, 500
    
    logger.info(f"Core routes registered with base path: {BASE_PATH}")


def setup_application():
    """Setup the complete application."""
    global _application_initialized
    
    if _application_initialized:
        logger.info("Application already initialized")
        return True
    
    try:
        logger.info("Starting application setup...")
        
        # Ensure required directories exist
        ensure_directories_exist()
        
        # Create Flask app
        create_flask_app()
        
        # Initialize services
        if not initialize_services():
            logger.error("Failed to initialize services")
            return False
        
        # Setup Flask-Login
        setup_flask_login()
        
        # Register core routes
        register_core_routes()
        
        # Register API routes
        register_all_routes(app, BASE_PATH, services)
        
        logger.info("Application setup completed successfully")
        _application_initialized = True
        return True
        
    except Exception as e:
        logger.error(f"Critical error during application setup: {e}")
        return False


def get_application():
    """Get the Flask application instance."""
    global app
    
    if not _application_initialized:
        if not setup_application():
            logger.error("Failed to setup application")
            # Create minimal app to prevent crashes
            app = Flask(__name__)
            
            @app.route('/')
            def error():
                return "Application initialization failed. Please check logs.", 500
    
    return app


# Initialize application on module load for IIS compatibility
try:
    if not _application_initialized:
        success = setup_application()
        if success:
            logger.info("Application initialized successfully on module load")
        else:
            logger.error("Application initialization failed on module load")
except Exception as e:
    logger.error(f"Critical error during module load initialization: {e}")
    # Create minimal app to prevent crashes
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return "Application initialization failed. Please check logs.", 500

# For IIS compatibility - app must be available at module level
app = get_application()

# Development server entry point
if __name__ == '__main__':
    app = get_application()
    if app:
        logger.info(f"Starting development server with base path: {BASE_PATH}")
        app.run(debug=True, host='127.0.0.1', port=2000)
    else:
        logger.error("Failed to get application instance")