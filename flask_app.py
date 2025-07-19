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
        else:
            logger.info("Processor already initialized")
            
        logger.info("Application setup completed")
        _application_initialized = True
        
    except Exception as e:
        logger.error(f"Critical error during application setup: {e}")

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

# Authentication Routes
@app.route('/meetingsai/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    
    if request.method == 'GET':
        return render_template('register.html')
    
    try:
        
        # Check if processor is available
        if not processor:
            logger.error("Processor not initialized during registration")
            return jsonify({'success': False, 'error': 'System not initialized - please try again later'}), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in registration request")
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        
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
        
        
        # Hash password
        try:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            return jsonify({'success': False, 'error': 'Password processing failed'}), 500
        
        # Create user
        try:
            user_id = processor.vector_db.create_user(username, email, full_name, password_hash)
            logger.info(f"User created with ID: {user_id}")
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return jsonify({'success': False, 'error': f'User creation failed: {str(e)}'}), 500
        
        # Create default project
        try:
            project_id = processor.vector_db.create_project(user_id, "Default Project", "Default project for meetings")
        except Exception as e:
            logger.error(f"Default project creation failed: {e}")
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
        return jsonify({'success': False, 'error': f'Registration failed: {str(e)}'}), 500

@app.route('/meetingsai/login', methods=['GET', 'POST'])
def login():
    """User login"""
    
    if request.method == 'GET':
        return render_template('login.html')
    
    try:
        
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        
        if not username or not password:
            logger.warning("Login failed - missing credentials")
            return jsonify({'success': False, 'error': 'Username and password are required'}), 400
        
        # Check processor availability
        if not processor:
            logger.error("Processor not initialized during login")
            # Try to re-initialize
            if initialize_processor():
                logger.info("Processor re-initialized successfully")
            else:
                logger.error("Failed to re-initialize processor")
                return jsonify({'success': False, 'error': 'System not initialized - database connection failed. Please contact administrator.'}), 500
        
        
        # Get user
        user = processor.vector_db.get_user_by_username(username)
        if not user:
            logger.warning(f"User not found: {username}")
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
        
        
        # Check password
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            logger.warning(f"Invalid password for user: {username}")
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
        
        
        # Login user with permanent session
        flask_user = User(user.user_id, user.username, user.email, user.full_name)
        login_user(flask_user, remember=True)
        session.permanent = True  # Make session permanent
        
        
        # Update last login
        try:
            processor.vector_db.update_user_last_login(user.user_id)
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
        
        if current_user.is_authenticated:
            
            # Always return authenticated if user has valid session - don't check database
            # This prevents logout loops when database/processor has issues
            session.permanent = True
            
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
            return jsonify({'authenticated': False, 'reason': 'not_logged_in'}), 401
            
    except Exception as e:
        logger.error(f"Auth status check error: {e}")
        return jsonify({'authenticated': False, 'reason': 'validation_error'}), 401

@app.route('/meetingsai/')
@app.route('/meetingsai')
def index():
    """Main chat interface - authentication handled by frontend"""
    # Let the frontend handle authentication check to support persistent sessions
    # This prevents immediate redirect on page refresh, allowing JS to validate session
    try:
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Error rendering chat.html: {e}")
        return f"Error loading chat interface: {str(e)}", 500

@app.route('/meetingsai/api/upload', methods=['POST'])
@login_required
def upload_files():
    """Handle file uploads with asynchronous processing and deduplication"""
    try:
        if 'files' not in request.files:
            logger.error("No files in request")
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            logger.error("No files selected")
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        # Get project selection from form data
        project_id = request.form.get('project_id', '').strip()
        meeting_id = request.form.get('meeting_id', '').strip()
        
        if not processor:
            logger.error("Processor not initialized")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        # Validate project belongs to user
        if project_id:
            user_projects = processor.vector_db.get_user_projects(current_user.user_id)
            project_exists = any(p.project_id == project_id for p in user_projects)
            if not project_exists:
                return jsonify({'success': False, 'error': 'Invalid project selection'}), 400
        
        # Validate meeting belongs to user and project
        if meeting_id:
            user_meetings = processor.vector_db.get_user_meetings(current_user.user_id, project_id)
            meeting_exists = any(m.meeting_id == meeting_id for m in user_meetings)
            if not meeting_exists:
                return jsonify({'success': False, 'error': 'Invalid meeting selection'}), 400
        
        # Prepare files for processing
        file_list = []
        validation_errors = []
        duplicates = []
        
        # Create user-specific directory structure
        user_folder = f"meeting_documents/user_{current_user.username}"
        if project_id:
            project_folder_name = "default"
            if project_id:
                user_projects = processor.vector_db.get_user_projects(current_user.user_id)
                selected_project = next((p for p in user_projects if p.project_id == project_id), None)
                if selected_project:
                    project_folder_name = selected_project.project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    project_folder_name = "".join(c for c in project_folder_name if c.isalnum() or c in ("_", "-"))
            
            upload_folder = os.path.join(user_folder, f"project_{project_folder_name}")
        else:
            upload_folder = user_folder
        
        os.makedirs(upload_folder, exist_ok=True)
        
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                
                # Validate file extension
                if not filename.lower().endswith(('.docx', '.txt', '.pdf')):
                    validation_errors.append({
                        'filename': filename,
                        'error': 'Unsupported file format'
                    })
                    continue
                
                # Save file to permanent location
                file_path = os.path.join(upload_folder, filename)
                
                # Handle duplicate filenames in filesystem
                counter = 1
                original_file_path = file_path
                while os.path.exists(file_path):
                    name, ext = os.path.splitext(original_file_path)
                    file_path = f"{name}_{counter}{ext}"
                    filename = os.path.basename(file_path)
                    counter += 1
                
                # Save file
                file.save(file_path)
                
                # Check for content duplicates
                try:
                    file_hash = processor.vector_db.calculate_file_hash(file_path)
                    duplicate_info = processor.vector_db.is_file_duplicate(file_hash, filename, current_user.user_id)
                    
                    if duplicate_info:
                        duplicates.append({
                            'filename': filename,
                            'original_filename': duplicate_info['original_filename'],
                            'created_at': duplicate_info['created_at']
                        })
                        os.remove(file_path)  # Remove the duplicate file
                        continue
                        
                except Exception as e:
                    logger.error(f"Error checking duplicate for {filename}: {e}")
                    validation_errors.append({
                        'filename': filename,
                        'error': f'Error processing file: {str(e)}'
                    })
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    continue
                
                file_list.append({
                    'path': file_path,
                    'filename': filename
                })
        
        # If no valid files to process
        if not file_list:
            return jsonify({
                'success': False,
                'error': 'No valid files to process',
                'validation_errors': validation_errors,
                'duplicates': duplicates
            }), 400
        
        # Start asynchronous processing
        try:
            import threading
            
            # Capture user_id before starting thread (current_user not available in thread)
            user_id = current_user.user_id
            
            # Create job ID first for both frontend and background processing
            job_id = processor.vector_db.create_upload_job(
                user_id,
                len(file_list),
                project_id,
                meeting_id
            )
            
            def process_in_background():
                """Background processing function"""
                try:
                    processor.process_files_batch_async(
                        file_list,
                        user_id,
                        project_id,
                        meeting_id,
                        max_workers=2,  # Limit concurrent processing
                        job_id=job_id  # Pass existing job_id
                    )
                except Exception as e:
                    logger.error(f"Background processing error: {e}")
            
            # Start background processing
            thread = threading.Thread(target=process_in_background)
            thread.daemon = True
            thread.start()
            
            # Return immediate response with job ID
            return jsonify({
                'success': True,
                'job_id': job_id,
                'total_files': len(file_list),
                'validation_errors': validation_errors,
                'duplicates': duplicates,
                'message': f'Upload started for {len(file_list)} files. Use job ID to track progress.'
            })
            
        except Exception as e:
            logger.error(f"Error starting background processing: {e}")
            # Clean up uploaded files
            for file_info in file_list:
                try:
                    if os.path.exists(file_info['path']):
                        os.remove(file_info['path'])
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up {file_info['path']}: {cleanup_error}")
            
            return jsonify({
                'success': False,
                'error': f'Error starting file processing: {str(e)}'
            }), 500
    
    except Exception as e:
        logger.error(f"Critical upload error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/meetingsai/api/job_status/<job_id>')
@login_required
def get_job_status(job_id):
    """Get the status of an upload job"""
    try:
        if not processor:
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        job_status = processor.vector_db.get_job_status(job_id)
        
        if not job_status:
            return jsonify({'success': False, 'error': 'Job not found'}), 404
        
        # Check if job belongs to current user
        if job_status['user_id'] != current_user.user_id:
            return jsonify({'success': False, 'error': 'Access denied'}), 403
        
        return jsonify({
            'success': True,
            'job_status': job_status
        })
        
    except Exception as e:
        logger.error(f"Job status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/meetingsai/api/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        document_ids = data.get('document_ids', None)
        project_id = data.get('project_id', None)
        project_ids = data.get('project_ids', None)
        meeting_ids = data.get('meeting_ids', None)
        date_filters = data.get('date_filters', None)
        folder_path = data.get('folder_path', None)
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        if not processor:
            logger.error("Processor not initialized for chat")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        # Check if documents are available
        try:
            vector_size = getattr(processor.vector_db.index, 'ntotal', 0) if processor.vector_db.index else 0
        except Exception as e:
            logger.error(f"Error checking vector database: {e}")
            vector_size = 0
        
        if vector_size == 0:
            response = "I don't have any documents to analyze yet. Please upload some meeting documents first! 📁"
            follow_up_questions = []
        else:
            try:
                user_id = current_user.user_id
                
                # Combine project filters
                combined_project_ids = []
                if project_id:
                    combined_project_ids.append(project_id)
                if project_ids:
                    combined_project_ids.extend(project_ids)
                final_project_id = combined_project_ids[0] if combined_project_ids else None
                
                response, context = processor.answer_query_with_intelligence(
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
                
                # Generate follow-up questions
                try:
                    follow_up_questions = processor.generate_follow_up_questions(message, response, context)
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

@app.route('/meetingsai/api/projects')
@login_required
def get_projects():
    """Get all projects for the current user"""
    try:
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

@app.route('/meetingsai/api/documents')
@login_required
def get_documents():
    """Get list of all documents for file selection"""
    try:
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

@app.route('/meetingsai/api/meetings')
@login_required
def get_meetings():
    """Get all meetings for the current user"""
    try:
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
                'title': meeting.meeting_name,
                'date': meeting.meeting_date.isoformat() if meeting.meeting_date else None,
                'participants': '',
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
        if not processor:
            logger.error("Processor not initialized for stats")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        stats = processor.get_meeting_statistics()
        
        if "error" in stats:
            logger.error(f"Error in stats: {stats['error']}")
            return jsonify({'success': False, 'error': stats['error']}), 500
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
