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

## IIS Deployment Constraints
- **Entry point:** `flask_app.py` (defined in web.config)
- **WSGI handler:** `flask_app.app`
- **Base path:** All routes must support `/meetingsai` prefix
- **Session storage:** Custom SQLite backend for WFASTCGI compatibility
- **Tiktoken cache:** Required in `tiktoken_cache/` directory for performance

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