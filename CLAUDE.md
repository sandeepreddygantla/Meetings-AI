# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meetings AI is a Flask-based document analysis and chat application that processes meeting documents using OpenAI/Azure OpenAI LLM technologies. The application features a modular architecture with AI-powered document processing, semantic search, and conversational interfaces.

## Development Commands

### Running the Application
```bash
# Main application (current entry point)
python flask_app.py

# Runs on http://127.0.0.1:5000 by default
# Visit: http://127.0.0.1:5000/meetingsai/
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file with required variables:
OPENAI_API_KEY=your_openai_api_key_here
BASE_PATH=/meetingsai                    # Optional, defaults to /meetingsai
SECRET_KEY=your-secure-random-key        # Required for Flask sessions
# OR for Azure:
# AZURE_CLIENT_ID=your_azure_client_id
# AZURE_CLIENT_SECRET=your_azure_client_secret  
# AZURE_PROJECT_ID=your_azure_project_id

# Create tiktoken cache directory (performance optimization)
mkdir tiktoken_cache
```

### Database Operations
The application uses SQLite + FAISS hybrid storage. Databases are created automatically on first run:
- `meeting_documents.db` - SQLite for metadata, users, projects
- `vector_index.faiss` - FAISS for semantic embeddings
- `sessions.db` - Session storage for IIS compatibility

```bash
# Reset vector database (useful for troubleshooting)
rm vector_index.faiss
# Application will rebuild automatically on next start

# Backup databases
cp meeting_documents.db backups/meeting_documents_$(date +%Y%m%d_%H%M%S).db
cp sessions.db backups/sessions_$(date +%Y%m%d_%H%M%S).db
```

## Architecture Overview

### Dual Architecture Pattern
The codebase maintains **two parallel implementations**:

1. **Legacy Monolithic** (`meeting_processor.py`): Original implementation with all functionality in one file
2. **Modular Architecture** (`src/` directory): Clean separation of concerns following industry standards

Both implementations access the same database and provide identical functionality.

### Core Components

**Entry Point:**
- `flask_app.py` - Main Flask application with modular architecture integration

**Global AI Client Management:**
- `meeting_processor.py` - Defines global variables: `access_token`, `embedding_model`, `llm`
- All other modules import these globals for AI operations

**Modular Architecture (`src/` directory):**
- **Config Layer** (`src/config/`) - Flask configuration and database session management
- **Database Layer** (`src/database/`) - DatabaseManager unifies SQLiteOperations and VectorOperations (FAISS)
- **Service Layer** (`src/services/`) - Business logic: AuthService, ChatService, DocumentService, UploadService
- **API Layer** (`src/api/`) - Flask blueprints for route organization
- **AI Layer** (`src/ai/`) - LLM client wrappers, embedding operations, query processing
- **Models** (`src/models/`) - Data classes for User, Document entities

### Key Design Patterns

**Global Variable Pattern:** All AI operations use global `llm`, `embedding_model`, `access_token` variables initialized once at startup.

**Service Composition:** Flask app initializes services with shared DatabaseManager instance, then passes services to API routes via blueprints.

**Environment Switching:** Application supports both OpenAI and Azure OpenAI by modifying the initialization functions in `meeting_processor.py`.

## Critical Development Rules

### LLM Integration Requirements (from instructions.md)
**MANDATORY:** Always use these global variables for AI operations:
```python
from meeting_processor import access_token, embedding_model, llm
```

**NEVER instantiate directly:**
```python
# FORBIDDEN - violates instructions.md
ChatOpenAI(...)
OpenAIEmbeddings(...)
AzureChatOpenAI(...)
```

**Correct pattern:**
```python
# REQUIRED - use global variables
# Always check for None before using (globals may be None if API keys missing)
if llm is not None:
    response = llm.invoke(prompt)
else:
    logger.error("LLM not available - check API key configuration")
    
if embedding_model is not None:
    embeddings = embedding_model.embed_documents(texts)
```

### Environment Switching Protocol
To switch between OpenAI and Azure environments, modify only these functions in `meeting_processor.py`:
- `get_access_token()` - Return None for OpenAI, Azure token for Azure
- `get_llm()` - Return ChatOpenAI or AzureChatOpenAI  
- `get_embedding_model()` - Return OpenAIEmbeddings or AzureOpenAIEmbeddings

All other code remains unchanged due to global variable pattern.

### Database Access Pattern
Always use DatabaseManager for database operations:
```python
# Through services (preferred)
chat_service = ChatService(db_manager, processor)

# Direct access (if needed)
db_manager = DatabaseManager()
documents = db_manager.get_all_documents(user_id)
```

## IIS Deployment & Performance Optimization

### IIS/WFASTCGI Configuration
**Critical web.config pattern for IIS deployment:**

```xml
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="FastCgiModule"
           scriptProcessor="C:\Python39\python.exe|C:\Python39\Scripts\wfastcgi.py"
           resourceType="Unspecified" />
    </handlers>
    <rewrite>
      <rules>
        <rule name="Static Files" stopProcessing="true">
          <match url="^meetingsai/static/(.*)$" />
          <action type="Rewrite" url="/static/{R:1}" />
        </rule>
      </rules>
    </rewrite>
  </system.webServer>
  <appSettings>
    <add key="WSGI_HANDLER" value="flask_app.app" />
    <add key="PYTHONPATH" value="C:\inetpub\wwwroot\meetingsai" />
    <add key="WSGI_LOG" value="C:\inetpub\logs\wfastcgi.log" />
  </appSettings>
</configuration>
```

**Entry Point Requirements:**
- **WSGI Handler:** `flask_app.app` (must be importable)
- **Python Path:** Must include application root directory
- **Base Path:** All routes support `/meetingsai` prefix via `BASE_PATH` configuration
- **Static Assets:** IIS handles static files directly for performance

### Memory Management & Performance Patterns

**Vector Operations Optimization:**
```python
# In src/database/vector_operations.py
class VectorOperations:
    def __init__(self, index_path: str, dimension: int):
        self.batch_size = 100  # Process vectors in batches for memory efficiency
        self.cache_size = 1000  # In-memory cache for frequent searches
        
    def batch_add_vectors(self, embeddings: List[np.ndarray], metadata: List[Dict]):
        """Add vectors in batches to prevent memory overflow"""
        for i in range(0, len(embeddings), self.batch_size):
            batch_embeddings = embeddings[i:i + self.batch_size]
            batch_metadata = metadata[i:i + self.batch_size]
            
            # Add batch to FAISS index
            start_id = self.index.ntotal
            self.index.add(np.array(batch_embeddings))
            
            # Update metadata mapping
            for j, meta in enumerate(batch_metadata):
                self.id_to_metadata[start_id + j] = meta
```

**Tiktoken Cache Strategy:**
```python
# Performance-critical caching for IIS memory constraints
import os
os.environ['TIKTOKEN_CACHE_DIR'] = 'tiktoken_cache'  # Required before tiktoken import

# In meeting_processor.py
def initialize_tiktoken_cache():
    """Ensure tiktoken cache exists and is writable"""
    cache_dir = os.getenv('TIKTOKEN_CACHE_DIR', 'tiktoken_cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Created tiktoken cache directory: {cache_dir}")
    
    # Verify write permissions
    test_file = os.path.join(cache_dir, 'test_write.tmp')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info("Tiktoken cache directory is writable")
    except Exception as e:
        logger.error(f"Tiktoken cache directory not writable: {e}")
```

### Session Management for IIS Compatibility

**Custom SQLite Session Backend:**
```python
# In src/config/database.py
class SQLiteSessionInterface(SessionInterface):
    """Custom session interface for IIS/WFASTCGI compatibility"""
    
    def __init__(self, db_path: str = 'sessions.db'):
        self.db_path = db_path
        self._init_session_table()
        
    def _init_session_table(self):
        """Create session table if not exists"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                data BLOB,
                expiry DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        
    def open_session(self, app, request):
        """Load session from SQLite database"""
        session_id = request.cookies.get(app.session_cookie_name)
        if not session_id:
            return self.session_class()
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clean expired sessions
        cursor.execute("DELETE FROM sessions WHERE expiry < ?", (datetime.now(),))
        
        # Load active session
        cursor.execute(
            "SELECT data FROM sessions WHERE session_id = ? AND expiry > ?",
            (session_id, datetime.now())
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                return self.session_class(pickle.loads(row[0]))
            except:
                return self.session_class()
        
        return self.session_class()
```

**Background Process Management:**
```python
# In src/config/settings.py - IIS-compatible background processing
import threading
from concurrent.futures import ThreadPoolExecutor

class IISCompatibleProcessManager:
    """Background processing that works within IIS constraints"""
    
    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_jobs = {}
        
    def submit_file_processing(self, file_data, user_id, project_id):
        """Submit file processing job with IIS-safe threading"""
        job_id = str(uuid.uuid4())
        
        # Use daemon threads to prevent IIS app pool hanging
        future = self.executor.submit(
            self._process_file_safe, file_data, user_id, project_id
        )
        
        self.active_jobs[job_id] = {
            'future': future,
            'status': 'processing',
            'created_at': datetime.now()
        }
        
        return job_id
        
    def _process_file_safe(self, file_data, user_id, project_id):
        """File processing with IIS-specific error handling"""
        try:
            # Set thread as daemon to prevent IIS hanging
            threading.current_thread().daemon = True
            
            # Process file with timeout for IIS constraints
            with timeout_handler(300):  # 5-minute timeout
                return self._process_file_content(file_data, user_id, project_id)
                
        except TimeoutError:
            logger.error("File processing timed out - IIS constraint")
            raise
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise
```

### Static Asset Optimization for IIS

**Asset Bundling Strategy:**
```xml
<!-- In web.config - Enable compression and caching -->
<system.webServer>
    <httpCompression>
        <dynamicTypes>
            <add mimeType="application/json" enabled="true" />
            <add mimeType="text/css" enabled="true" />
            <add mimeType="application/javascript" enabled="true" />
        </dynamicTypes>
        <staticTypes>
            <add mimeType="text/css" enabled="true" />
            <add mimeType="application/javascript" enabled="true" />
        </staticTypes>
    </httpCompression>
    
    <staticContent>
        <clientCache cacheControlMode="UseMaxAge" cacheControlMaxAge="30.00:00:00" />
    </staticContent>
</system.webServer>
```

**Dynamic Asset Path Configuration:**
```javascript
// In static/js/config.js - IIS-compatible asset loading
window.APP_CONFIG = {
    BASE_PATH: '{{ base_path }}',  // Dynamically set by Flask template
    STATIC_URL: '{{ base_path }}/static',
    API_BASE: '{{ base_path }}/api'
};

// Ensure all AJAX calls use dynamic base path
function makeAPICall(endpoint, options = {}) {
    const url = `${window.APP_CONFIG.API_BASE}${endpoint}`;
    return fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        }
    });
}
```

### Database Performance Optimization

**Connection Pooling for SQLite:**
```python
# In src/database/sqlite_operations.py
import sqlite3
from contextlib import contextmanager
import threading

class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool for IIS"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        self.lock = threading.Lock()
        
        # Pre-create connections
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
            conn.execute("PRAGMA synchronous = NORMAL")  # Performance balance
            self.connections.append(conn)
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic return"""
        with self.lock:
            if self.connections:
                conn = self.connections.pop()
            else:
                # Create temporary connection if pool exhausted
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                
        try:
            yield conn
        finally:
            with self.lock:
                if len(self.connections) < self.pool_size:
                    self.connections.append(conn)
                else:
                    conn.close()
```

**Index Optimization:**
```sql
-- In database initialization - Performance-critical indexes
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_project_id ON documents(project_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_project ON documents(user_id, project_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expiry);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_documents_user_date ON documents(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_documents_project_date ON documents(project_id, created_at);
```

### Application Warm-up for IIS

**IIS Application Initialization:**
```python
# In flask_app.py - Application warm-up procedure
def warm_up_application():
    """Warm up critical application components for IIS"""
    logger.info("Starting application warm-up...")
    
    try:
        # 1. Initialize AI clients
        from meeting_processor import access_token, embedding_model, llm
        if not all([embedding_model, llm]):
            logger.warning("AI clients not fully initialized during warm-up")
        
        # 2. Verify database connections
        db_manager = DatabaseManager()
        db_manager.sqlite_ops.verify_connection()
        
        # 3. Pre-load tiktoken cache
        initialize_tiktoken_cache()
        
        # 4. Verify vector index
        if os.path.exists('vector_index.faiss'):
            db_manager.vector_ops.verify_index()
            
        # 5. Clean up old sessions
        db_manager.sqlite_ops.cleanup_expired_sessions()
        
        logger.info("Application warm-up completed successfully")
        
    except Exception as e:
        logger.error(f"Application warm-up failed: {e}")

# Call warm-up during application initialization
if __name__ == "__main__":
    warm_up_application()
    app.run(host="127.0.0.1", port=5000, debug=False)
```

## Frontend Architecture

### Advanced Mention System
Located in `static/js/modules/mentions.js` with sophisticated parsing:
- `@project:name` - Filter by project
- `@meeting:name` - Filter by meeting
- `@date:today|yesterday|YYYY-MM-DD` - Date filtering
- `#folder` - Folder navigation
- `#folder>` - Show folder contents

### Configuration Management
- Backend: Dynamic `BASE_PATH` via `src/config/settings.py`
- Frontend: `static/js/config.js` matches backend configuration

## File Processing Pipeline
1. **Upload & Validation:** File type validation, deduplication via SHA-256 hashing
2. **Content Extraction:** Supports .docx, .pdf, .txt with fallback handling
3. **AI Analysis:** LLM-powered metadata extraction (topics, participants, decisions)
4. **Chunking & Embedding:** RecursiveCharacterTextSplitter + text-embedding-3-large
5. **Storage:** SQLite metadata + FAISS vector storage with bidirectional linking
6. **Background Processing:** ThreadPoolExecutor with job tracking and status updates

## Environment Variables
```bash
# AI Configuration (choose one)
OPENAI_API_KEY=sk-...                    # For OpenAI
AZURE_CLIENT_ID=...                     # For Azure OpenAI
AZURE_CLIENT_SECRET=...                 # For Azure OpenAI  
AZURE_PROJECT_ID=...                    # For Azure OpenAI

# Application Configuration
BASE_PATH=/meetingsai                   # Route prefix (default)
SECRET_KEY=your-flask-secret-key        # Flask sessions
TIKTOKEN_CACHE_DIR=tiktoken_cache       # Token caching (performance)
```

## Troubleshooting Common Issues

### Vector Database Sync Problems
If queries return "no relevant information" after file transfer between systems:
```bash
# Delete FAISS index to force rebuild from SQLite
rm vector_index.faiss
# Restart application - auto-rebuilds from database
```

### Enhanced Search Fallback
If enhanced search returns 0 results but basic search works:
- Check user_id filtering in `src/database/manager.py`
- Verify folder_path filtering isn't too restrictive
- Monitor logs for "Enhanced search returned 0 results"

### LLM Initialization Failures
- Verify environment variables are set correctly
- Check `logs/flask_app.log` for initialization errors
- Ensure tiktoken cache directory exists and is writable

### Testing
Currently, no automated tests are configured. The application relies on manual testing and logging for validation.

**Recommended setup for future development:**
```bash
# Install testing dependencies
pip install pytest pytest-flask

# Create tests/ directory structure
mkdir tests
mkdir tests/unit tests/integration
```

### Logging
The application uses comprehensive logging:
- **Main app logs**: `logs/flask_app.log`
- **Processor logs**: `logs/meeting_processor.log`
- **Console output**: Both file and console logging enabled

**Log levels**: INFO (default), configurable via logging configuration in both entry points.

### Configuration Management
**Dynamic Base Path**: Set `BASE_PATH` environment variable to change route prefix:
```bash
export BASE_PATH=/custom-path  # Changes all routes to /custom-path/*
```

**Frontend-Backend Sync**: `static/js/config.js` automatically syncs with backend configuration.

## Advanced Development Commands

### Database Management & Inspection
```bash
# Inspect database schema and contents
sqlite3 meeting_documents.db ".schema"
sqlite3 meeting_documents.db "SELECT COUNT(*) FROM documents;"
sqlite3 meeting_documents.db "SELECT COUNT(*) FROM chunks;"

# Check vector index statistics  
python -c "
import faiss
index = faiss.read_index('vector_index.faiss')
print(f'Vector index size: {index.ntotal} vectors')
print(f'Vector dimension: {index.d}')
"

# Rebuild vector index from database (if sync issues)
rm vector_index.faiss
python -c "
from src.database.manager import DatabaseManager
db = DatabaseManager()
print('Vector index rebuilt from database')
"

# Clean up sessions database
sqlite3 sessions.db "DELETE FROM sessions WHERE expiry < datetime('now');"
```

### Performance Monitoring & Optimization
```bash
# Monitor database performance
sqlite3 meeting_documents.db "EXPLAIN QUERY PLAN SELECT * FROM documents WHERE user_id = 'user123';"

# Check tiktoken cache effectiveness
du -sh tiktoken_cache/
ls -la tiktoken_cache/

# Monitor memory usage during vector operations
python -c "
import psutil, os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Performance profiling for slow queries
tail -f logs/flask_app.log | grep -E "(slow|error|timeout)"
```

### Development Environment Management
```bash
# Switch between OpenAI and Azure (modify meeting_processor.py)
# For OpenAI:
export OPENAI_API_KEY=sk-...
unset AZURE_CLIENT_ID AZURE_CLIENT_SECRET AZURE_PROJECT_ID

# For Azure:
export AZURE_CLIENT_ID=... AZURE_CLIENT_SECRET=... AZURE_PROJECT_ID=...
unset OPENAI_API_KEY

# Validate environment setup
python -c "
from meeting_processor import access_token, embedding_model, llm
print(f'Access token: {\"✓\" if access_token else \"✗\"}')
print(f'Embedding model: {\"✓\" if embedding_model else \"✗\"}')  
print(f'LLM: {\"✓\" if llm else \"✗\"}')
"
```

## Advanced Architecture Patterns

### Service Composition & Dependency Injection
The application uses a sophisticated dependency injection pattern centered around `DatabaseManager`:

```python
# In flask_app.py - Services share the same DatabaseManager instance
db_manager = DatabaseManager()
services = {
    'auth': AuthService(db_manager),
    'chat': ChatService(db_manager, processor),
    'document': DocumentService(db_manager),
    'upload': UploadService(db_manager, processor)
}

# Services are passed to API routes via blueprints
app.register_blueprint(chat_routes, url_prefix=BASE_PATH)
```

**Critical Pattern**: All services share the same `DatabaseManager` instance to ensure consistent database state and connection pooling.

### Dual Processing Strategy (Enhanced vs Legacy)
The `ChatService` implements a sophisticated dual processing approach:

```python
# In src/services/chat_service.py
class ChatService:
    def __init__(self, db_manager, processor):
        self.use_enhanced_processing = True  # Feature flag
        self.enhanced_summary_threshold = 10  # Document count threshold
        
    def process_chat_query(self, message, user_id, **filters):
        # Decision logic for processing strategy
        if self.use_enhanced_processing and estimated_docs >= self.enhanced_summary_threshold:
            return self._process_with_enhanced_context(message, context)
        else:
            return self._process_with_legacy_processor(message, context)
```

**When Enhanced Processing is Used**:
- Queries expecting 10+ documents
- Complex multi-meeting summaries  
- Date range queries spanning multiple meetings
- Project-wide analysis requests

**When Legacy Processing is Used**:
- Simple document lookups
- Single meeting questions
- Backwards compatibility scenarios

### Global Variable Pattern for AI Clients
**Critical Requirement**: All AI operations MUST use globals from `meeting_processor.py`:

```python
# CORRECT - Always import globals
from meeting_processor import access_token, embedding_model, llm

# REQUIRED - Always check for None (API keys may be missing)
async def generate_embedding(text: str):
    if embedding_model is None:
        logger.error("Embedding model not available - check API keys")
        raise ValueError("AI services not initialized")
    
    try:
        embedding = embedding_model.embed_documents([text])
        return embedding[0]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

# CORRECT - Fallback handling for missing AI services
async def chat_with_llm(prompt: str):
    if llm is None:
        return "AI services are currently unavailable. Please check configuration."
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        return "I encountered an error processing your request."
```

### Context Management Architecture
The `EnhancedContextManager` provides sophisticated document context optimization:

```python
# In src/ai/context_manager.py
class EnhancedContextManager:
    def __init__(self, db_manager, processor):
        self.chunk_manager = ChunkManager(db_manager)
        self.query_processor = QueryProcessor()
        
    def build_context(self, query: str, user_id: str, **filters) -> QueryContext:
        # 1. Query analysis and intent detection
        query_analysis = self.query_processor.analyze_query(query)
        
        # 2. Smart document filtering based on intent
        relevant_docs = self._filter_documents_by_intent(query_analysis, **filters)
        
        # 3. Context optimization (chunk selection, relevance scoring)
        optimized_chunks = self.chunk_manager.get_optimized_chunks(
            relevant_docs, query, max_tokens=8000
        )
        
        return QueryContext(chunks=optimized_chunks, metadata=query_analysis)
```

## File Processing Pipeline Deep Dive

The document processing follows a sophisticated 6-stage pipeline with job tracking:

### Stage 1: Upload & Validation
```python
# In src/services/upload_service.py
async def process_files(self, files, project_id, user_id):
    for file in files:
        # SHA-256 deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()
        if self.db_manager.document_exists_by_hash(file_hash):
            continue  # Skip duplicate
            
        # File type validation
        if not self._validate_file_type(file.filename):
            raise ValueError(f"Unsupported file type: {file.filename}")
```

### Stage 2: Content Extraction with Fallback
```python
def extract_content(self, file_path: str) -> Tuple[str, Dict]:
    extractors = [
        self._extract_docx,    # Primary: python-docx
        self._extract_pdf,     # Primary: PyPDF2  
        self._extract_txt,     # Fallback: plain text
        self._extract_generic  # Last resort: binary detection
    ]
    
    for extractor in extractors:
        try:
            content, metadata = extractor(file_path)
            if content and len(content.strip()) > 0:
                return content, metadata
        except Exception as e:
            logger.warning(f"Extractor {extractor.__name__} failed: {e}")
    
    raise ValueError("All content extractors failed")
```

### Stage 3: AI-Powered Metadata Extraction
```python
def extract_meeting_metadata(self, content: str) -> Dict:
    prompt = f"""
    Analyze this meeting document and extract:
    1. Meeting date and time
    2. Participants (names and roles)
    3. Key topics discussed
    4. Decisions made
    5. Action items with owners
    
    Content: {content[:2000]}...
    """
    
    if llm is not None:
        response = llm.invoke(prompt)
        return self._parse_metadata_response(response.content)
    else:
        return self._extract_metadata_fallback(content)
```

### Stage 4: Chunking & Embedding Strategy
```python
def process_document_chunks(self, content: str, doc_id: str):
    # RecursiveCharacterTextSplitter with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    
    chunks = splitter.split_text(content)
    
    # Batch embedding generation for efficiency
    if embedding_model is not None:
        embeddings = embedding_model.embed_documents(chunks)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.db_manager.store_chunk(chunk_id, chunk, embedding, doc_id)
```

### Stage 5: Hybrid Database Storage
```python
class DatabaseManager:
    def store_document_complete(self, doc_data: Dict, chunks: List[Tuple]):
        """Store document with atomic transaction across SQLite + FAISS"""
        try:
            # 1. SQLite transaction for metadata
            with self.sqlite_ops.get_connection() as conn:
                doc_id = self.sqlite_ops.store_document(conn, doc_data)
                
                # 2. FAISS vector storage
                for chunk_text, embedding in chunks:
                    chunk_id = self.vector_ops.add_vector(embedding, {
                        'document_id': doc_id,
                        'content': chunk_text
                    })
                    
                    # 3. Bidirectional linking
                    self.sqlite_ops.store_chunk_metadata(conn, chunk_id, doc_id)
                    
            conn.commit()
            logger.info(f"Document {doc_id} stored successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Document storage failed: {e}")
            raise
```

### Stage 6: Background Job Tracking
```python
# In src/services/upload_service.py using ThreadPoolExecutor
def process_files_async(self, files, project_id, user_id):
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = {}
        
        for file in files:
            job_id = str(uuid.uuid4())
            future = executor.submit(self._process_single_file, file, project_id, user_id)
            jobs[job_id] = {
                'future': future,
                'filename': file.filename,
                'status': 'processing',
                'progress': 0
            }
            
        # Real-time job status updates
        for job_id, job_data in jobs.items():
            try:
                result = job_data['future'].result(timeout=300)  # 5-minute timeout
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['result'] = result
            except Exception as e:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = str(e)
```

## Frontend Architecture Patterns

### Advanced Mention System Implementation
The mention system supports sophisticated query filtering:

```javascript
// In static/js/modules/mentions.js
class MentionHandler {
    parseMessageForMentions(message) {
        const mentions = [];
        
        // @project:ProjectName - Filter by specific project
        const projectRegex = /@project:([^@\s]+)/gi;
        
        // @meeting:MeetingTitle - Filter by specific meeting
        const meetingRegex = /@meeting:([^@\s]+)/gi;
        
        // @date:today|yesterday|2024-01-15 - Date filtering
        const dateRegex = /@date:(today|yesterday|\d{4}-\d{2}-\d{2})/gi;
        
        // #folder - Navigate to folder contents
        const folderRegex = /#([^#\s>]+)(>?)/g;
        
        return this._extractMentions(message, [
            projectRegex, meetingRegex, dateRegex, folderRegex
        ]);
    }
}
```

**Frontend-Backend Integration**:
```javascript
// Real-time mention processing
async function processMentionQuery(message) {
    const mentions = mentionHandler.parseMessageForMentions(message);
    const cleanQuery = mentionHandler.cleanMessageFromMentions(message);
    
    const response = await fetch('/meetingsai/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: cleanQuery,
            mentions: mentions,  // Structured mention data
            user_id: getCurrentUserId()
        })
    });
}
```

### Performance Optimization Patterns
```javascript
// Loading state management to prevent API loops
let isLoadingDocuments = false;
let isLoadingProjects = false;

async function loadDocuments() {
    if (isLoadingDocuments) {
        console.log('Documents already loading, skipping...');
        return;
    }
    
    isLoadingDocuments = true;
    try {
        const response = await fetch('/meetingsai/api/documents');
        // Process response...
    } finally {
        isLoadingDocuments = false;
    }
}

// Coordinated initialization to prevent race conditions
Promise.all([
    loadDocuments(),
    loadProjects(), 
    loadMeetings()
]).then(() => {
    loadFolders();  // Load after dependencies ready
});
```

## Critical Implementation Patterns & Error Handling

### Comprehensive Error Handling Strategy

**Service Layer Exception Management:**
```python
# In src/services/base_service.py - Common error handling patterns
class ServiceException(Exception):
    """Base service exception with error categorization"""
    def __init__(self, message: str, error_type: str = "GENERAL", details: Dict = None):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)

class ChatService:
    def process_chat_query(self, message: str, user_id: str, **filters) -> Tuple[str, List[str], str]:
        """Process chat query with comprehensive error handling"""
        try:
            # Validate inputs
            if not message or not message.strip():
                raise ServiceException("Empty message provided", "VALIDATION_ERROR")
            
            if not user_id:
                raise ServiceException("User ID required", "AUTHENTICATION_ERROR")
            
            # Check AI service availability
            from meeting_processor import llm, embedding_model
            if llm is None:
                return self._handle_ai_unavailable("LLM service not available")
            
            # Process with timeout and fallback
            try:
                with timeout_handler(30):  # 30-second timeout
                    return self._process_query_enhanced(message, user_id, **filters)
                    
            except TimeoutError:
                logger.warning(f"Query processing timeout for user {user_id}")
                return self._process_query_fallback(message, user_id, **filters)
                
        except ServiceException:
            raise  # Re-raise service exceptions
        except Exception as e:
            logger.error(f"Unexpected error in chat processing: {e}", exc_info=True)
            raise ServiceException(
                "An unexpected error occurred during query processing",
                "SYSTEM_ERROR",
                {"original_error": str(e)}
            )
    
    def _handle_ai_unavailable(self, error_msg: str) -> Tuple[str, List[str], str]:
        """Graceful degradation when AI services unavailable"""
        fallback_response = """
        I'm currently unable to process your request due to AI service unavailability. 
        Please check the system logs and ensure:
        1. API keys are correctly configured
        2. Network connectivity is available
        3. Service quotas haven't been exceeded
        
        Try again in a few moments.
        """
        return fallback_response.strip(), [], "AI_UNAVAILABLE"
```

### Vector Database Synchronization Handling

**FAISS Index Consistency Management:**
```python
# In src/database/vector_operations.py - Vector sync error handling
class VectorOperations:
    def verify_index_consistency(self) -> Dict[str, Any]:
        """Verify SQLite-FAISS consistency and report issues"""
        consistency_report = {
            'status': 'UNKNOWN',
            'issues': [],
            'sqlite_chunks': 0,
            'faiss_vectors': 0,
            'orphaned_vectors': [],
            'missing_vectors': []
        }
        
        try:
            # Get chunk count from SQLite
            sqlite_chunks = self.sqlite_ops.get_chunk_count()
            consistency_report['sqlite_chunks'] = sqlite_chunks
            
            # Get vector count from FAISS
            faiss_vectors = self.index.ntotal if hasattr(self, 'index') else 0
            consistency_report['faiss_vectors'] = faiss_vectors
            
            # Check for mismatches and repair if needed
            if sqlite_chunks != faiss_vectors:
                consistency_report['issues'].append({
                    'type': 'COUNT_MISMATCH',
                    'message': f"SQLite chunks ({sqlite_chunks}) != FAISS vectors ({faiss_vectors})"
                })
                
            return consistency_report
            
        except Exception as e:
            consistency_report['status'] = 'ERROR'
            consistency_report['issues'].append({
                'type': 'VERIFICATION_ERROR',
                'message': f"Failed to verify consistency: {e}"
            })
            return consistency_report
```

### User Session & Security Patterns

**Session Security & User Isolation:**
```python
# In src/services/auth_service.py - Security-focused session management
class AuthService:
    def authenticate_user(self, username: str, password: str, request_ip: str) -> Tuple[bool, str, Optional[str]]:
        """Secure user authentication with rate limiting"""
        try:
            # Check for account lockout
            if self._is_account_locked(username, request_ip):
                return False, "Account temporarily locked due to failed attempts", None
            
            # Validate credentials
            user = self.db_manager.get_user_by_username(username)
            if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                self._record_failed_attempt(username, request_ip)
                return False, "Invalid credentials", None
            
            # Generate secure session token
            session_token = self._generate_session_token(user.user_id)
            logger.info(f"User {username} authenticated successfully from {request_ip}")
            
            return True, "Authentication successful", session_token
            
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}")
            return False, "Authentication system error", None
    
    def ensure_user_isolation(self, user_id: str, requested_resource: str, resource_owner: str) -> bool:
        """Ensure users can only access their own resources"""
        if user_id != resource_owner:
            logger.warning(f"User {user_id} attempted to access resource owned by {resource_owner}")
            return False
        return True
```

## Advanced Integration & Security Patterns

### Real-time Query Processing Integration

**Frontend-Backend Query Coordination:**
```javascript
// In static/script.js - Advanced query processing with mention integration
async function processAdvancedQuery(message, mentionContext = null) {
    try {
        // Parse mentions from message
        const mentions = window.mentionHandler.parseMessageForMentions(message);
        const cleanQuery = window.mentionHandler.cleanMessageFromMentions(message);
        
        // Prepare query payload with context
        const queryPayload = {
            message: cleanQuery,
            mentions: mentions,
            user_id: getCurrentUserId(),
            context: {
                current_project: getCurrentProjectId(),
                active_filters: getActiveFilters(),
                session_context: getSessionContext()
            }
        };
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send query with retry logic
        const response = await fetchWithRetry('/meetingsai/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(queryPayload)
        }, 3);
        
        if (!response.ok) {
            throw new Error(`Query failed: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Handle different response types
        if (result.error) {
            displayErrorMessage(result.error, result.error_type);
        } else {
            displayQueryResult(result.response, result.sources, result.context);
        }
        
    } catch (error) {
        console.error('Query processing failed:', error);
        displayErrorMessage('Failed to process your query. Please try again.', 'NETWORK_ERROR');
    } finally {
        hideTypingIndicator();
    }
}

// Retry logic for network resilience
async function fetchWithRetry(url, options, maxRetries) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await fetch(url, options);
            if (response.ok || response.status < 500) {
                return response;  // Don't retry client errors
            }
            throw new Error(`HTTP ${response.status}`);
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
        }
    }
}
```

### Configuration Management Security

**Environment-Specific Configuration Patterns:**
```python
# In src/config/settings.py - Secure configuration management
import os
from typing import Dict, Any

class SecureConfigManager:
    """Secure configuration management with environment separation"""
    
    def __init__(self):
        self.config = self._load_configuration()
        self._validate_critical_settings()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration with security validation"""
        config = {
            # Core application settings
            'BASE_PATH': os.getenv('BASE_PATH', '/meetingsai'),
            'SECRET_KEY': os.getenv('SECRET_KEY'),
            'DEBUG': os.getenv('DEBUG', 'False').lower() == 'true',
            
            # Database settings
            'DB_PATH': os.getenv('DB_PATH', 'meeting_documents.db'),
            'VECTOR_INDEX_PATH': os.getenv('VECTOR_INDEX_PATH', 'vector_index.faiss'),
            'SESSION_DB_PATH': os.getenv('SESSION_DB_PATH', 'sessions.db'),
            
            # AI service configuration
            'AI_PROVIDER': os.getenv('AI_PROVIDER', 'openai'),  # 'openai' or 'azure'
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'AZURE_CLIENT_ID': os.getenv('AZURE_CLIENT_ID'),
            'AZURE_CLIENT_SECRET': os.getenv('AZURE_CLIENT_SECRET'),
            'AZURE_PROJECT_ID': os.getenv('AZURE_PROJECT_ID'),
            
            # Performance settings
            'TIKTOKEN_CACHE_DIR': os.getenv('TIKTOKEN_CACHE_DIR', 'tiktoken_cache'),
            'MAX_FILE_SIZE': int(os.getenv('MAX_FILE_SIZE', '52428800')),  # 50MB
            'MAX_WORKERS': int(os.getenv('MAX_WORKERS', '3')),
            
            # Security settings
            'SESSION_TIMEOUT': int(os.getenv('SESSION_TIMEOUT', '28800')),  # 8 hours
            'MAX_LOGIN_ATTEMPTS': int(os.getenv('MAX_LOGIN_ATTEMPTS', '5')),
            'ENABLE_AUDIT_LOGGING': os.getenv('ENABLE_AUDIT_LOGGING', 'True').lower() == 'true'
        }
        
        return config
    
    def _validate_critical_settings(self):
        """Validate critical configuration settings"""
        critical_missing = []
        
        # Check for required settings
        if not self.config['SECRET_KEY']:
            critical_missing.append('SECRET_KEY')
        
        # Check AI provider configuration
        if self.config['AI_PROVIDER'] == 'openai' and not self.config['OPENAI_API_KEY']:
            critical_missing.append('OPENAI_API_KEY')
        elif self.config['AI_PROVIDER'] == 'azure':
            azure_required = ['AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET', 'AZURE_PROJECT_ID']
            for setting in azure_required:
                if not self.config[setting]:
                    critical_missing.append(setting)
        
        if critical_missing:
            raise ValueError(f"Critical configuration missing: {', '.join(critical_missing)}")
        
        # Validate security settings
        if len(self.config['SECRET_KEY']) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
    
    def get_masked_config(self) -> Dict[str, Any]:
        """Return configuration with sensitive values masked for logging"""
        masked_config = self.config.copy()
        sensitive_keys = ['SECRET_KEY', 'OPENAI_API_KEY', 'AZURE_CLIENT_SECRET']
        
        for key in sensitive_keys:
            if masked_config.get(key):
                masked_config[key] = f"{masked_config[key][:8]}***"
        
        return masked_config
```

### Enhanced Logging & Monitoring

**Comprehensive Audit Trail:**
```python
# In src/utils/audit_logger.py - Security and performance audit logging
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

class AuditLogger:
    """Comprehensive audit logging for security and performance monitoring"""
    
    def __init__(self, log_file: str = 'logs/audit.log'):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_user_action(self, user_id: str, action: str, resource: str, 
                       details: Optional[Dict] = None, request_ip: str = None):
        """Log user actions for security audit"""
        audit_entry = {
            'type': 'USER_ACTION',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'request_ip': request_ip,
            'details': details or {}
        }
        self.logger.info(json.dumps(audit_entry))
    
    def log_performance_metric(self, operation: str, duration: float, 
                             resource_usage: Dict, context: Dict = None):
        """Log performance metrics for optimization"""
        performance_entry = {
            'type': 'PERFORMANCE',
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'resource_usage': resource_usage,
            'context': context or {}
        }
        self.logger.info(json.dumps(performance_entry))
    
    def log_security_event(self, event_type: str, severity: str, 
                          details: Dict, user_id: str = None, request_ip: str = None):
        """Log security events for incident response"""
        security_entry = {
            'type': 'SECURITY_EVENT',
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'user_id': user_id,
            'request_ip': request_ip,
            'details': details
        }
        self.logger.warning(json.dumps(security_entry))

# Usage examples throughout the application:
# audit_logger.log_user_action(user_id, "DOCUMENT_UPLOAD", doc_id, {"file_size": size})
# audit_logger.log_performance_metric("VECTOR_SEARCH", duration, {"vectors_searched": count})
# audit_logger.log_security_event("FAILED_LOGIN", "MEDIUM", {"username": username, "attempts": count})
```