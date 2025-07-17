from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import json
import logging
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import shutil
import bcrypt
import secrets
import sqlite3
import pickle
from flask.sessions import SessionInterface, SessionMixin
from uuid import uuid4

# Ensure logs directory exists
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

# Import your existing classes
try:
    from meeting_processor import EnhancedMeetingDocumentProcessor
    logger.info("Successfully imported meeting_processor")
except ImportError as e:
    logger.error(f"Failed to import meeting_processor: {e}")
    logger.error("Make sure meeting_processor.py is in the same directory")
    exit(1)

# Create Flask app with proper static folder configuration for IIS app
app = Flask(__name__, 
           static_folder='static',  # Explicitly set static folder
           static_url_path='/meetingsai/static',  # Set static URL path for IIS app
           template_folder='templates')  # Set template folder

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# WFASTCGI FIX: Use database-backed sessions instead of memory/file sessions
app.config['SECRET_KEY'] = 'your-secret-key-here-' + secrets.token_hex(16)

# Enhanced session configuration for persistence
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)  # 30 day session
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent XSS attacks
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)  # 30 day remember me
app.config['REMEMBER_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['REMEMBER_COOKIE_HTTPONLY'] = True

# Custom session class for wfastcgi compatibility
class SqliteSession(dict, SessionMixin):
    pass

class SqliteSessionInterface(SessionInterface):
    """SQLite-based session interface for wfastcgi compatibility"""
    
    def __init__(self, db_path='sessions.db'):
        self.db_path = db_path
        self.session_cookie_name = 'session'
        self._init_db()
    
    def _init_db(self):
        """Initialize the sessions table"""
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
        """Load session from database"""
        # Always return a valid session object, never None
        session = SqliteSession()
        
        try:
            sid = request.cookies.get(self.session_cookie_name)
            if not sid:
                logging.debug("No session ID in cookies, returning empty session")
                return session
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT data FROM sessions WHERE id = ? AND expiry > ?', 
                         (sid, datetime.now()))
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
        """Save session to database"""
        try:
            domain = self.get_cookie_domain(app)
            path = self.get_cookie_path(app)
            
            # Handle empty or unmodified sessions
            if not session or not getattr(session, 'modified', True):
                if hasattr(session, 'modified') and session.modified:
                    response.delete_cookie(self.session_cookie_name, domain=domain, path=path)
                return
            
            # Get or generate session ID
            sid = None
            try:
                from flask import request as flask_request
                sid = flask_request.cookies.get(self.session_cookie_name) if flask_request else None
            except:
                pass
                
            if not sid:
                sid = str(uuid4())
            
            # Calculate expiry
            expiry = datetime.now() + app.permanent_session_lifetime
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            data = pickle.dumps(dict(session))
            conn.execute('INSERT OR REPLACE INTO sessions (id, data, expiry) VALUES (?, ?, ?)',
                        (sid, data, expiry))
            conn.commit()
            conn.close()
            
            # Set cookie
            response.set_cookie(self.session_cookie_name, sid,
                              expires=expiry, httponly=True,
                              domain=domain, path=path, secure=False)
            
            logging.debug(f"Session saved successfully with ID: {sid}")
            
        except Exception as e:
            logging.error(f"Session save error: {e}")
    
    def get_cookie_name(self, app):
        """Get the session cookie name"""
        return self.session_cookie_name

# Apply custom session interface for wfastcgi
app.session_interface = SqliteSessionInterface()

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Flask will use the route name, not the URL path
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Global processor instance
processor = None
_application_initialized = False

def initialize_processor():
    """Initialize the document processor"""
    global processor
    try:
        logger.info("Initializing Enhanced Meeting Document Processor...")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Check if we can import the processor class
        try:
            logger.info("Creating processor instance...")
            processor = EnhancedMeetingDocumentProcessor(chunk_size=1000, chunk_overlap=200)
            logger.info("Processor instance created successfully")
        except Exception as proc_error:
            logger.error(f"Failed to create processor instance: {proc_error}")
            logger.exception("Processor creation traceback:")
            return False
        
        # Ensure databases are properly initialized
        if processor:
            logger.info("Checking processor vector_db...")
            if processor.vector_db:
                logger.info("Vector DB exists, initializing database schema...")
                
                # Initialize the database schema if needed
                try:
                    # Force database initialization by creating tables
                    processor.vector_db._init_database()
                    logger.info("Database schema initialized successfully")
                except Exception as db_error:
                    logger.error(f"Database initialization error: {db_error}")
                    logger.exception("Database initialization traceback:")
                    # Continue anyway, might work
                
                # Clean up expired sessions on startup
                try:
                    cleaned_count = processor.vector_db.cleanup_expired_sessions()
                    logger.info(f"Cleaned up {cleaned_count} expired sessions on startup")
                except Exception as cleanup_error:
                    logger.error(f"Session cleanup error: {cleanup_error}")
                    logger.exception("Session cleanup traceback:")
                    # Continue anyway
            else:
                logger.error("Processor vector_db is None!")
                return False
        else:
            logger.error("Processor is None after creation!")
            return False
        
        logger.info("Processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        logger.exception("Full exception traceback:")
        return False

# Initialize directories and processor when module loads (for IIS deployment)
def setup_application():
    """Setup application directories and initialize processor"""
    global processor, _application_initialized
    
    if _application_initialized:
        logger.info("Application already initialized, skipping setup")
        return
    
    try:
        logger.info("Starting application setup...")
        
        # Ensure required directories exist
        for directory in ['uploads', 'temp', 'meeting_documents', 'logs', 'backups', 'templates', 'static']:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created/verified directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
        
        # Initialize processor if not already done
        if processor is None:
            logger.info("Initializing processor for IIS deployment...")
            try:
                if initialize_processor():
                    logger.info("Processor initialized successfully for IIS")
                else:
                    logger.error("Failed to initialize processor for IIS")
            except Exception as e:
                logger.error(f"Exception during processor initialization: {e}")
                logger.exception("Full processor initialization traceback:")
        else:
            logger.info("Processor already initialized")
            
        logger.info("Application setup completed")
        _application_initialized = True
        
    except Exception as e:
        logger.error(f"Critical error during application setup: {e}")
        logger.exception("Full application setup traceback:")

# Setup application on module load - only if not already setup
if not _application_initialized:
    setup_application()
    _application_initialized = True

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, email, full_name):
        self.id = user_id
        self.user_id = user_id
        self.username = username
        self.email = email
        self.full_name = full_name
    
    def get_id(self):
        return self.user_id

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    try:
        logger.info(f"Loading user for session: {user_id}")
        
        # Try to get user from database, but don't fail if database is unavailable
        try:
            if processor and processor.vector_db:
                user = processor.vector_db.get_user_by_id(user_id)
                if user:
                    logger.info(f"User loaded successfully: {user.username}")
                    return User(user.user_id, user.username, user.email, user.full_name)
                else:
                    logger.warning(f"User not found in database: {user_id}")
                    return None
            else:
                # Database not available, create minimal user from session ID
                logger.warning("Database not available, creating minimal user from session")
                # Extract username from user_id format: user_YYYYMMDD_HHMMSS_username
                if "_" in user_id:
                    username = user_id.split("_")[-1]
                    return User(user_id, username, f"{username}@company.com", username)
                return None
        except Exception as db_error:
            logger.error(f"Database error in user loader: {db_error}")
            # Fallback to minimal user
            if "_" in user_id:
                username = user_id.split("_")[-1]
                return User(user_id, username, f"{username}@company.com", username)
            return None
            
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {e}")
        return None

def initialize_processor():
    """Initialize the document processor"""
    global processor
    try:
        logger.info("Initializing Enhanced Meeting Document Processor...")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Check if we can import the processor class
        try:
            logger.info("Creating processor instance...")
            processor = EnhancedMeetingDocumentProcessor(chunk_size=1000, chunk_overlap=200)
            logger.info("Processor instance created successfully")
        except Exception as proc_error:
            logger.error(f"Failed to create processor instance: {proc_error}")
            logger.exception("Processor creation traceback:")
            return False
        
        # Ensure databases are properly initialized
        if processor:
            logger.info("Checking processor vector_db...")
            if processor.vector_db:
                logger.info("Vector DB exists, initializing database schema...")
                
                # Initialize the database schema if needed
                try:
                    # Force database initialization by creating tables
                    processor.vector_db._init_database()
                    logger.info("Database schema initialized successfully")
                except Exception as db_error:
                    logger.error(f"Database initialization error: {db_error}")
                    logger.exception("Database initialization traceback:")
                    # Continue anyway, might work
                
                # Clean up expired sessions on startup
                try:
                    cleaned_count = processor.vector_db.cleanup_expired_sessions()
                    logger.info(f"Cleaned up {cleaned_count} expired sessions on startup")
                except Exception as cleanup_error:
                    logger.error(f"Session cleanup error: {cleanup_error}")
                    logger.exception("Session cleanup traceback:")
                    # Continue anyway
            else:
                logger.error("Processor vector_db is None!")
                return False
        else:
            logger.error("Processor is None after creation!")
            return False
        
        logger.info("Processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        logger.exception("Full exception traceback:")
        return False

# Authentication Routes
@app.route('/meetingsai/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    logger.info(f"Registration request received - Method: {request.method}")
    
    if request.method == 'GET':
        return render_template('register.html')
    
    try:
        logger.info("Processing POST registration request")
        
        # Check if processor is available
        if not processor:
            logger.error("Processor not initialized during registration")
            return jsonify({'success': False, 'error': 'System not initialized - please try again later'}), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in registration request")
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        logger.info(f"Registration data received for user: {data.get('username', 'unknown')}")
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        full_name = data.get('full_name', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        if not all([username, email, full_name, password]):
            logger.warning("Registration failed - missing required fields")
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        if password != confirm_password:
            logger.warning("Registration failed - password mismatch")
            return jsonify({'success': False, 'error': 'Passwords do not match'}), 400
        
        if len(password) < 6:
            logger.warning("Registration failed - password too short")
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
        
        logger.info("Starting user creation process")
        
        # Hash password
        try:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            logger.info("Password hashed successfully")
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            return jsonify({'success': False, 'error': 'Password processing failed'}), 500
        
        # Create user
        try:
            logger.info("Creating user in database")
            user_id = processor.vector_db.create_user(username, email, full_name, password_hash)
            logger.info(f"User created with ID: {user_id}")
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            logger.exception("Full traceback:")
            return jsonify({'success': False, 'error': f'User creation failed: {str(e)}'}), 500
        
        # Create default project
        try:
            logger.info("Creating default project")
            project_id = processor.vector_db.create_project(user_id, "Default Project", "Default project for meetings")
            logger.info(f"Default project created with ID: {project_id}")
        except Exception as e:
            logger.error(f"Default project creation failed: {e}")
            logger.exception("Full traceback:")
            # Don't fail registration if project creation fails
            project_id = None
        
        logger.info(f"New user registered: {username} ({user_id})")
        return jsonify({
            'success': True, 
            'message': 'Registration successful! Please log in.',
            'user_id': user_id
        })
        
    except ValueError as e:
        logger.error(f"Registration validation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Registration error: {e}")
        logger.exception("Full registration error traceback:")
        return jsonify({'success': False, 'error': f'Registration failed: {str(e)}'}), 500

@app.route('/meetingsai/login', methods=['GET', 'POST'])
def login():
    """User login"""
    logger.info(f"Login request received - Method: {request.method}")
    
    if request.method == 'GET':
        logger.info(f"Login GET request - User Agent: {request.headers.get('User-Agent', 'Unknown')}")
        logger.info(f"Login GET request - Referer: {request.headers.get('Referer', 'None')}")
        return render_template('login.html')
    
    try:
        logger.info("Processing POST login request")
        
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        logger.info(f"Login attempt for user: {username}")
        
        if not username or not password:
            logger.warning("Login failed - missing credentials")
            return jsonify({'success': False, 'error': 'Username and password are required'}), 400
        
        # Check processor availability
        if not processor:
            logger.error("Processor not initialized during login")
            # Try to re-initialize
            logger.info("Attempting to re-initialize processor for login...")
            if initialize_processor():
                logger.info("Processor re-initialized successfully")
            else:
                logger.error("Failed to re-initialize processor")
                return jsonify({'success': False, 'error': 'System not initialized - database connection failed. Please contact administrator.'}), 500
        
        logger.info("Getting user from database")
        
        # Get user
        user = processor.vector_db.get_user_by_username(username)
        if not user:
            logger.warning(f"User not found: {username}")
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
        
        logger.info(f"User found: {username}, checking password")
        
        # Check password
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            logger.warning(f"Invalid password for user: {username}")
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
        
        logger.info(f"Password valid for user: {username}, creating session")
        
        # Login user with permanent session
        flask_user = User(user.user_id, user.username, user.email, user.full_name)
        login_user(flask_user, remember=True)
        session.permanent = True  # Make session permanent
        
        logger.info(f"Session created for user: {username}")
        
        # Update last login
        try:
            processor.vector_db.update_user_last_login(user.user_id)
            logger.info(f"Last login updated for user: {username}")
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            # Don't fail login for this
        
        logger.info(f"User logged in successfully: {username}")
        return jsonify({
            'success': True, 
            'message': 'Login successful',
            'user': {
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        logger.exception("Full login error traceback:")
        return jsonify({'success': False, 'error': f'Login failed: {str(e)}'}), 500

@app.route('/meetingsai/logout', methods=['POST'])
@login_required
def logout():
    """User logout"""
    username = current_user.username
    logout_user()
    logger.info(f"User logged out: {username}")
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/meetingsai/api/auth/status')
def auth_status():
    """Check authentication status and validate session"""
    try:
        logger.info("=== AUTH STATUS ENDPOINT CALLED ===")
        logger.info(f"Auth status check - authenticated: {current_user.is_authenticated}")
        
        if current_user.is_authenticated:
            logger.info(f"User authenticated: {current_user.username}")
            
            # Always return authenticated if user has valid session - don't check database
            # This prevents logout loops when database/processor has issues
            session.permanent = True
            logger.info(f"Auth status valid for user: {current_user.username}")
            
            return jsonify({
                'authenticated': True,
                'user': {
                    'user_id': current_user.user_id,
                    'username': current_user.username,
                    'email': current_user.email,
                    'full_name': current_user.full_name
                }
            })
        else:
            logger.info("User not authenticated")
            return jsonify({'authenticated': False, 'reason': 'not_logged_in'}), 401
            
    except Exception as e:
        logger.error(f"Auth status check error: {e}")
        logger.exception("Full auth status error traceback:")
        return jsonify({'authenticated': False, 'reason': 'validation_error'}), 401

@app.route('/meetingsai/')
@app.route('/meetingsai')
def index():
    """Main chat interface - authentication handled by frontend"""
    logger.info("Main index route accessed")
    # Let the frontend handle authentication check to support persistent sessions
    # This prevents immediate redirect on page refresh, allowing JS to validate session
    try:
        logger.info("Attempting to render chat.html template")
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Error rendering chat.html: {e}")
        logger.exception("Template rendering error:")
        return f"Error loading chat interface: {str(e)}", 500

@app.route('/meetingsai/api/upload', methods=['POST'])
@login_required
def upload_files():
    """Handle file uploads with detailed result tracking"""
    try:
        logger.info("Upload request received")
        
        if 'files' not in request.files:
            logger.error("No files in request")
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            logger.error("No files selected")
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        # Get project selection from form data
        project_id = request.form.get('project_id', '').strip()
        logger.info(f"Project selection: {project_id}")
        
        if not processor:
            logger.error("Processor not initialized")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        # Validate project belongs to user
        if project_id:
            user_projects = processor.vector_db.get_user_projects(current_user.user_id)
            project_exists = any(p.project_id == project_id for p in user_projects)
            if not project_exists:
                return jsonify({'success': False, 'error': 'Invalid project selection'}), 400
        
        logger.info(f"Processing {len(files)} files for project {project_id or 'default'}")
        
        results = []
        successful_uploads = 0
        
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                logger.info(f"Processing file: {filename}")
                
                file_result = {
                    'filename': filename,
                    'success': False,
                    'error': None,
                    'chunks': 0
                }
                
                try:
                    # Validate file extension
                    if not filename.lower().endswith(('.docx', '.txt', '.pdf')):
                        file_result['error'] = 'Unsupported file format'
                        results.append(file_result)
                        logger.warning(f"Unsupported file format: {filename}")
                        continue
                    
                    # Save file to temp directory
                    temp_path = os.path.join('temp', filename)
                    os.makedirs('temp', exist_ok=True)
                    file.save(temp_path)
                    logger.info(f"File saved to: {temp_path}")
                    
                    # Check file size
                    file_size = os.path.getsize(temp_path)
                    if file_size == 0:
                        file_result['error'] = 'File is empty'
                        results.append(file_result)
                        os.remove(temp_path)
                        continue
                    
                    if file_size > 50 * 1024 * 1024:  # 50MB limit
                        file_result['error'] = 'File too large (max 50MB)'
                        results.append(file_result)
                        os.remove(temp_path)
                        continue
                    
                    # Process document
                    content = processor.read_document_content(temp_path)
                    if not content or not content.strip():
                        file_result['error'] = 'No readable content found'
                        results.append(file_result)
                        os.remove(temp_path)
                        logger.warning(f"No content extracted from {filename}")
                        continue
                    
                    logger.info(f"Content extracted from {filename}, length: {len(content)}")
                    
                    # Parse and process document with user context
                    meeting_doc = processor.parse_document_content(content, filename)
                    
                    # Add user context to document
                    meeting_doc.user_id = current_user.user_id
                    
                    # Use selected project or default project
                    user_projects = processor.vector_db.get_user_projects(current_user.user_id)
                    if user_projects:
                        if project_id:
                            # Use selected project
                            selected_project = next((p for p in user_projects if p.project_id == project_id), None)
                            if selected_project:
                                meeting_doc.project_id = selected_project.project_id
                                logger.info(f"Assigned document to selected project: {selected_project.project_name}")
                        else:
                            # Use default project (first one)
                            default_project = user_projects[0]
                            meeting_doc.project_id = default_project.project_id
                            logger.info(f"Assigned document to default project: {default_project.project_name}")
                        
                        # Create a basic meeting for the document
                        if meeting_doc.project_id:
                            meeting_id = processor.vector_db.create_meeting(
                                current_user.user_id,
                                meeting_doc.project_id,
                                f"Meeting - {filename}",
                                meeting_doc.date
                            )
                            meeting_doc.meeting_id = meeting_id
                    
                    chunks = processor.chunk_document(meeting_doc)
                    
                    # Create project-based folder structure
                    project_folder_name = "Default Project"  # Default fallback
                    if meeting_doc.project_id:
                        # Get the project name for folder creation
                        selected_project = next((p for p in user_projects if p.project_id == meeting_doc.project_id), None)
                        if selected_project:
                            # Sanitize project name for folder creation
                            project_folder_name = selected_project.project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                            project_folder_name = "".join(c for c in project_folder_name if c.isalnum() or c in ("_", "-"))
                    
                    # Create project-specific folder structure
                    user_folder = f"meeting_documents/user_{current_user.username}"
                    project_folder = os.path.join(user_folder, f"project_{project_folder_name}")
                    os.makedirs(project_folder, exist_ok=True)
                    
                    # Set the folder path for the document
                    folder_path = f"user_{current_user.username}/project_{project_folder_name}"
                    meeting_doc.folder_path = folder_path
                    
                    permanent_path = os.path.join(project_folder, filename)
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_permanent_path = permanent_path
                    while os.path.exists(permanent_path):
                        name, ext = os.path.splitext(original_permanent_path)
                        permanent_path = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    shutil.move(temp_path, permanent_path)
                    
                    # Add document to database with folder path
                    processor.vector_db.add_document(meeting_doc, chunks)
                    
                    # Success!
                    file_result['success'] = True
                    file_result['chunks'] = len(chunks)
                    successful_uploads += 1
                    
                    logger.info(f"Successfully processed {filename} with {len(chunks)} chunks")
                    
                except Exception as e:
                    file_result['error'] = str(e)
                    logger.error(f"Error processing {filename}: {e}")
                    
                    # Clean up temp file if it exists
                    try:
                        temp_path = os.path.join('temp', filename)
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up temp file: {cleanup_error}")
                
                results.append(file_result)
        
        # Save vector index if any files were processed successfully
        if successful_uploads > 0:
            try:
                processor.vector_db.save_index()
                logger.info(f"Vector index saved after processing {successful_uploads} files")
            except Exception as e:
                logger.error(f"Error saving vector index: {e}")
        
        # Prepare response
        response_data = {
            'success': True,
            'results': results,
            'processed': successful_uploads,
            'total': len(results),
            'message': f'Successfully processed {successful_uploads} of {len(results)} files'
        }
        
        logger.info(f"Upload completed: {successful_uploads}/{len(results)} files processed successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Critical upload error: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'processed': 0,
            'total': 0
        }), 500

@app.route('/meetingsai/api/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        document_ids = data.get('document_ids', None)  # Document filtering
        project_id = data.get('project_id', None)  # Single project filtering (legacy)
        project_ids = data.get('project_ids', None)  # Multiple project filtering (enhanced)
        meeting_ids = data.get('meeting_ids', None)  # Meeting filtering
        date_filters = data.get('date_filters', None)  # Date filtering
        folder_path = data.get('folder_path', None)  # Folder-based filtering
        
        logger.info(f"Chat request received: {message[:100]}...")
        if document_ids:
            logger.info(f"Document filter: {document_ids}")
        if project_id:
            logger.info(f"Project filter: {project_id}")
        if project_ids:
            logger.info(f"Enhanced project filters: {project_ids}")
        if meeting_ids:
            logger.info(f"Meeting filters: {meeting_ids}")
        if date_filters:
            logger.info(f"Date filters: {date_filters}")
        if folder_path:
            logger.info(f"Folder filter: {folder_path}")
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        if not processor:
            logger.error("Processor not initialized for chat")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        # Check if documents are available
        try:
            vector_size = getattr(processor.vector_db.index, 'ntotal', 0) if processor.vector_db.index else 0
            logger.info(f"Vector database size: {vector_size}")
        except Exception as e:
            logger.error(f"Error checking vector database: {e}")
            vector_size = 0
        
        if vector_size == 0:
            response = "I don't have any documents to analyze yet. Please upload some meeting documents first! 📁"
            follow_up_questions = []
            logger.info("No documents available, sending default response")
        else:
            try:
                logger.info("Generating response using processor")
                user_id = current_user.user_id
                
                # Combine project filters (legacy and enhanced)
                combined_project_ids = []
                if project_id:
                    combined_project_ids.append(project_id)
                if project_ids:
                    combined_project_ids.extend(project_ids)
                final_project_id = combined_project_ids[0] if combined_project_ids else None
                
                response, context = processor.answer_query(
                    message, 
                    user_id=user_id, 
                    document_ids=document_ids, 
                    project_id=final_project_id,
                    meeting_ids=meeting_ids,
                    date_filters=date_filters,
                    folder_path=folder_path,
                    context_limit=50, 
                    include_context=True
                )
                logger.info(f"Response generated, length: {len(response)}")
                
                # Generate follow-up questions
                try:
                    follow_up_questions = processor.generate_follow_up_questions(message, response, context)
                    logger.info(f"Generated {len(follow_up_questions)} follow-up questions")
                except Exception as follow_up_error:
                    logger.error(f"Error generating follow-up questions: {follow_up_error}")
                    follow_up_questions = []
                    
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = f"I encountered an error while processing your question: {str(e)}"
                follow_up_questions = []
        
        return jsonify({
            'success': True,
            'response': response,
            'follow_up_questions': follow_up_questions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/documents')
@login_required
def get_documents():
    """Get list of all documents for file selection"""
    try:
        logger.info("Documents request received")
        
        if not processor:
            logger.error("Processor not initialized for documents")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        user_id = current_user.user_id
        documents = processor.vector_db.get_all_documents(user_id)
        
        return jsonify({
            'success': True,
            'documents': documents,
            'count': len(documents)
        })
        
    except Exception as e:
        logger.error(f"Documents error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Project Management Endpoints
@app.route('/meetingsai/api/projects')
@login_required
def get_projects():
    """Get all projects for the current user"""
    try:
        logger.info("Projects request received")
        
        if not processor:
            logger.error("Processor not initialized for projects")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        user_id = current_user.user_id
        projects = processor.vector_db.get_user_projects(user_id)
        
        # Convert projects to dictionaries
        project_list = []
        for project in projects:
            project_list.append({
                'project_id': project.project_id,
                'project_name': project.project_name,
                'description': project.description,
                'created_at': project.created_at.isoformat(),
                'is_active': project.is_active
            })
        
        return jsonify({
            'success': True,
            'projects': project_list,
            'count': len(project_list)
        })
        
    except Exception as e:
        logger.error(f"Projects error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/projects', methods=['POST'])
@login_required
def create_project():
    """Create a new project"""
    try:
        data = request.get_json()
        project_name = data.get('project_name', '').strip()
        description = data.get('description', '').strip()
        
        if not project_name:
            return jsonify({'success': False, 'error': 'Project name is required'}), 400
        
        if not processor:
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        user_id = current_user.user_id
        project_id = processor.vector_db.create_project(user_id, project_name, description)
        
        logger.info(f"New project created: {project_name} ({project_id}) for user {current_user.username}")
        return jsonify({
            'success': True,
            'message': 'Project created successfully',
            'project_id': project_id
        })
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Create project error: {e}")
        return jsonify({'success': False, 'error': 'Failed to create project'}), 500

@app.route('/meetingsai/api/meetings')
@login_required
def get_meetings():
    """Get all meetings for the current user"""
    try:
        logger.info("Meetings request received")
        
        if not processor:
            logger.error("Processor not initialized for meetings")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        user_id = current_user.user_id
        meetings = processor.vector_db.get_user_meetings(user_id)
        
        # Convert meetings to dictionaries
        meeting_list = []
        for meeting in meetings:
            meeting_list.append({
                'meeting_id': meeting.meeting_id,
                'title': meeting.meeting_name,  # Use meeting_name from the dataclass
                'date': meeting.meeting_date.isoformat() if meeting.meeting_date else None,
                'participants': '',  # This will be populated from documents later
                'project_id': meeting.project_id,
                'created_at': meeting.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'meetings': meeting_list,
            'count': len(meeting_list)
        })
        
    except Exception as e:
        logger.error(f"Meetings error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/stats')
@login_required
def get_stats():
    """Get system statistics"""
    try:
        logger.info("Stats request received")
        
        if not processor:
            logger.error("Processor not initialized for stats")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        stats = processor.get_meeting_statistics()
        
        if "error" in stats:
            logger.error(f"Error in stats: {stats['error']}")
            return jsonify({'success': False, 'error': stats['error']}), 500
        
        logger.info("Stats generated successfully")
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/refresh', methods=['POST'])
@login_required
def refresh_system():
    """Refresh the system"""
    try:
        logger.info("System refresh requested")
        
        if processor:
            processor.refresh_clients()
            logger.info("System refreshed successfully")
            return jsonify({'success': True, 'message': 'System refreshed successfully'})
        else:
            logger.error("Processor not initialized for refresh")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/meetingsai/api/setup', methods=['POST'])
def setup_database():
    """Manual database setup endpoint"""
    try:
        global processor
        logger.info("Manual database setup requested")
        
        setup_results = {
            'directories_created': [],
            'processor_initialized': False,
            'database_initialized': False,
            'errors': []
        }
        
        # Create directories
        for directory in ['uploads', 'temp', 'meeting_documents', 'logs', 'backups']:
            try:
                os.makedirs(directory, exist_ok=True)
                setup_results['directories_created'].append(directory)
                logger.info(f"Created/verified directory: {directory}")
            except Exception as e:
                setup_results['errors'].append(f"Directory {directory}: {str(e)}")
        
        # Initialize processor
        try:
            if processor is None:
                logger.info("Initializing processor...")
                if initialize_processor():
                    setup_results['processor_initialized'] = True
                    logger.info("Processor initialized successfully")
                else:
                    setup_results['errors'].append("Processor initialization failed")
            else:
                setup_results['processor_initialized'] = True
                logger.info("Processor already initialized")
        except Exception as e:
            setup_results['errors'].append(f"Processor error: {str(e)}")
        
        # Initialize database
        try:
            if processor and processor.vector_db:
                logger.info("Initializing database schema...")
                processor.vector_db._init_database()
                setup_results['database_initialized'] = True
                logger.info("Database schema initialized successfully")
            else:
                setup_results['errors'].append("No processor or vector_db available")
        except Exception as e:
            setup_results['errors'].append(f"Database initialization error: {str(e)}")
        
        # Test database
        try:
            if processor and processor.vector_db:
                # Try to query users to test database
                test_users = processor.vector_db.get_all_users()
                setup_results['database_test'] = f"Database accessible, {len(test_users) if test_users else 0} users found"
        except Exception as e:
            setup_results['errors'].append(f"Database test error: {str(e)}")
        
        success = setup_results['processor_initialized'] and setup_results['database_initialized']
        return jsonify({
            'success': success,
            'message': 'Database setup completed' if success else 'Database setup had errors',
            'results': setup_results
        })
        
    except Exception as e:
        logger.error(f"Manual setup error: {e}")
        logger.exception("Full setup error traceback:")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/debug')
def debug_endpoint():
    """Debug endpoint to check system status"""
    try:
        debug_info = {
            'flask_app_running': True,
            'processor_status': processor is not None,
            'current_directory': os.getcwd(),
            'python_path': os.environ.get('PYTHONPATH', 'Not set'),
            'request_method': request.method,
            'request_url': request.url,
        }
        
        if processor:
            debug_info['processor_type'] = type(processor).__name__
            debug_info['vector_db_status'] = processor.vector_db is not None
            
            # Check database files
            try:
                import glob
                debug_info['database_files'] = {
                    'sqlite_files': glob.glob('*.db'),
                    'faiss_files': glob.glob('*.faiss*'),
                    'vector_files': glob.glob('vector_*')
                }
            except Exception as e:
                debug_info['file_check_error'] = str(e)
        
        return jsonify({'success': True, 'debug': debug_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'debug': {'error': str(e)}})

@app.route('/meetingsai/api/test')
def test_system():
    """Test endpoint to check if system is working"""
    try:
        status = {
            'processor_initialized': processor is not None,
            'vector_db_available': False,
            'vector_size': 0,
            'working_directory': os.getcwd(),
            'app_initialized': True
        }
        
        if processor:
            try:
                status['vector_db_available'] = processor.vector_db is not None
                if processor.vector_db and processor.vector_db.index:
                    status['vector_size'] = getattr(processor.vector_db.index, 'ntotal', 0)
            except Exception as e:
                logger.error(f"Error checking vector DB: {e}")
                status['vector_db_error'] = str(e)
        else:
            logger.warning("Processor is None - attempting re-initialization")
            # Try to re-initialize
            if initialize_processor():
                status['processor_initialized'] = True
                status['reinitialized'] = True
            else:
                status['initialization_failed'] = True
        
        logger.info(f"System test status: {status}")
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/test_registration')
def test_registration_readiness():
    """Test if system is ready for user registration"""
    try:
        readiness_check = {
            'processor_available': processor is not None,
            'vector_db_available': False,
            'database_accessible': False,
            'can_create_user': False,
            'errors': []
        }
        
        if processor:
            readiness_check['vector_db_available'] = processor.vector_db is not None
            
            if processor.vector_db:
                try:
                    # Test database access
                    test_users = processor.vector_db.get_all_users()
                    readiness_check['database_accessible'] = True
                    readiness_check['existing_users_count'] = len(test_users) if test_users else 0
                except Exception as e:
                    readiness_check['errors'].append(f"Database access error: {str(e)}")
                
                try:
                    # Test if we can create a test user (dry run)
                    import bcrypt
                    test_hash = bcrypt.hashpw('test123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    readiness_check['can_hash_password'] = True
                    readiness_check['can_create_user'] = True
                except Exception as e:
                    readiness_check['errors'].append(f"User creation test error: {str(e)}")
            else:
                readiness_check['errors'].append("Vector DB not available")
        else:
            readiness_check['errors'].append("Processor not initialized")
        
        overall_ready = (readiness_check['processor_available'] and 
                        readiness_check['vector_db_available'] and 
                        readiness_check['database_accessible'])
        
        return jsonify({
            'success': True,
            'ready_for_registration': overall_ready,
            'readiness_check': readiness_check
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/test_folder_fix')
@login_required
def test_folder_fix():
    """Test if our folder fix is loaded"""
    try:
        user_id = current_user.user_id
        documents = processor.vector_db.get_all_documents(user_id)
        
        has_folder_path = any('folder_path' in doc for doc in documents)
        has_project_name = any('project_name' in doc for doc in documents)
        
        return jsonify({
            'success': True, 
            'folder_fix_loaded': has_folder_path and has_project_name,
            'documents_count': len(documents),
            'sample_doc': documents[0] if documents else None,
            'all_docs': documents
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/test_auth_endpoint')
def test_auth_endpoint():
    """Simple test endpoint to verify API is working"""
    logger.info("=== AUTH TEST ENDPOINT CALLED ===")
    return jsonify({'success': True, 'message': 'Auth endpoint is reachable', 'timestamp': datetime.now().isoformat()})

@app.route('/meetingsai/api/test_db_init')
def test_db_init():
    """Test database initialization separately"""
    try:
        logger.info("Testing database initialization...")
        
        # Test just the VectorDatabase creation
        from meeting_processor import VectorDatabase
        test_db = VectorDatabase()
        
        result = {
            'db_created': test_db is not None,
            'db_path': test_db.db_path if test_db else None,
            'index_path': test_db.index_path if test_db else None,
            'working_directory': os.getcwd()
        }
        
        # Test database connection
        try:
            import sqlite3
            conn = sqlite3.connect(test_db.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            result['tables'] = [table[0] for table in tables]
            result['db_accessible'] = True
        except Exception as db_error:
            result['db_accessible'] = False
            result['db_error'] = str(db_error)
        
        logger.info(f"Database test result: {result}")
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"Database test error: {e}")
        logger.exception("Database test traceback:")
        return jsonify({'success': False, 'error': str(e), 'working_directory': os.getcwd()})

if __name__ == '__main__':
    # This block only runs when script is executed directly (development mode)
    # For IIS deployment, setup_application() already handles initialization
    
    # Check if required files exist (for development)
    required_files = {
        'templates/chat.html': 'HTML template',
        'static/styles.css': 'CSS stylesheet', 
        'static/script.js': 'JavaScript file'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_path} ({description})")
    
    if missing_files:
        print("Missing required files:")
        for missing in missing_files:
            print(f"   - {missing}")
        exit(1)
    
    # Run the Flask development server
    app.run(debug=True)