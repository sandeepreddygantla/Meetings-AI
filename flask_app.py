"""
Flask application for Meetings AI - Refactored with modular architecture.
Maintains IIS compatibility while providing clean separation of concerns.
"""
from flask import Flask, render_template, redirect, session, send_from_directory, make_response
from flask_login import LoginManager, login_required
import mimetypes
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
from src.services.background_processor import get_background_processor, shutdown_background_processor

# Import API routes
from src.api import register_all_routes

# Import asset optimizer
from src.utils.asset_optimizer import AssetOptimizer

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
asset_optimizer = None
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
    global db_manager, services, processor, asset_optimizer
    
    try:
        logger.info("Initializing services...")
        
        # AI clients now use lazy loading - no initialization needed at startup
        # This significantly improves application startup time
        logger.info("AI clients configured for lazy loading - will initialize on first use")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        logger.info("Database manager initialized")
        
        # Initialize processor for backwards compatibility - share database manager
        if EnhancedMeetingDocumentProcessor:
            try:
                # Pass database manager directly to avoid creating duplicate instances
                processor = EnhancedMeetingDocumentProcessor(
                    chunk_size=1000, 
                    chunk_overlap=200, 
                    db_manager=db_manager
                )
                logger.info("Processor initialized with shared database manager")
            except Exception as e:
                logger.error(f"Failed to initialize processor: {e}")
                processor = None
        
        # Initialize services
        services['auth_service'] = AuthService(db_manager)
        services['chat_service'] = ChatService(db_manager, processor)
        services['document_service'] = DocumentService(db_manager, processor)
        services['upload_service'] = UploadService(db_manager, services['document_service'])
        
        # Initialize asset optimizer
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        asset_optimizer = AssetOptimizer(static_dir)
        logger.info("Asset optimizer initialized")
        
        # Initialize background processor
        background_processor = get_background_processor()
        logger.info("Background processor initialized")
        
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
    """Register core application routes with static file optimizations."""
    
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
    
    @app.route(f'{BASE_PATH}/static/<path:filename>')
    def optimized_static(filename):
        """Serve static files with optimization, caching headers and compression."""
        try:
            # Get the static folder path
            static_folder = os.path.join(app.root_path, 'static')
            
            # Check if file exists
            file_path = os.path.join(static_folder, filename)
            if not os.path.exists(file_path):
                return "File not found", 404
            
            # Get file extension
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Handle CSS and JS files with optimization
            if file_ext == '.css' and asset_optimizer:
                content = asset_optimizer.get_optimized_css(filename)
                if content:
                    response = make_response(content)
                    response.headers['Content-Type'] = 'text/css; charset=utf-8'
                    response.headers['Cache-Control'] = 'public, max-age=3600'
                    response.headers['ETag'] = f'"{hash(filename + str(os.path.getmtime(file_path)))}"'
                    response.headers['Vary'] = 'Accept-Encoding'
                    return response
            
            elif file_ext == '.js' and asset_optimizer:
                content = asset_optimizer.get_optimized_js(filename)
                if content:
                    response = make_response(content)
                    response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
                    response.headers['Cache-Control'] = 'public, max-age=3600'
                    response.headers['ETag'] = f'"{hash(filename + str(os.path.getmtime(file_path)))}"'
                    response.headers['Vary'] = 'Accept-Encoding'
                    return response
            
            # Fallback to regular file serving with cache headers
            response = make_response(send_from_directory(static_folder, filename))
            
            # Set cache headers based on file type
            if file_ext in ['.js', '.css']:
                # Cache JavaScript and CSS for 1 hour
                response.headers['Cache-Control'] = 'public, max-age=3600'
                response.headers['ETag'] = f'"{hash(filename + str(os.path.getmtime(file_path)))}"'
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico']:
                # Cache images for 24 hours
                response.headers['Cache-Control'] = 'public, max-age=86400'
                response.headers['ETag'] = f'"{hash(filename + str(os.path.getmtime(file_path)))}"'
            elif file_ext in ['.woff', '.woff2', '.ttf', '.eot']:
                # Cache fonts for 30 days
                response.headers['Cache-Control'] = 'public, max-age=2592000'
                response.headers['ETag'] = f'"{hash(filename + str(os.path.getmtime(file_path)))}"'
            else:
                # Default caching for other files
                response.headers['Cache-Control'] = 'public, max-age=1800'  # 30 minutes
            
            # Add compression hint
            response.headers['Vary'] = 'Accept-Encoding'
            
            return response
            
        except Exception as e:
            logger.error(f"Error serving static file {filename}: {e}")
            return "Error serving file", 500
    
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
    
    @app.route(f'{BASE_PATH}/api/background-tasks/<task_id>', methods=['GET'])
    @login_required
    def get_background_task_status(task_id):
        """Get background task status."""
        try:
            bg_processor = get_background_processor()
            task = bg_processor.get_task_status(task_id)
            
            if not task:
                return {'success': False, 'error': 'Task not found'}, 404
            
            return {
                'success': True,
                'task': {
                    'id': task.task_id,
                    'name': task.name,
                    'status': task.status.value,
                    'progress': task.progress,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'error': task.error
                }
            }
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {'success': False, 'error': str(e)}, 500
    
    @app.route(f'{BASE_PATH}/api/background-tasks', methods=['GET'])
    @login_required
    def get_background_tasks():
        """Get all background tasks."""
        try:
            bg_processor = get_background_processor()
            tasks = bg_processor.get_all_tasks()
            
            task_list = []
            for task in tasks.values():
                task_list.append({
                    'id': task.task_id,
                    'name': task.name,
                    'status': task.status.value,
                    'progress': task.progress,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'error': task.error
                })
            
            return {
                'success': True,
                'tasks': task_list,
                'stats': bg_processor.get_stats()
            }
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
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
        app.run(debug=True)
    else:
        logger.error("Failed to get application instance")