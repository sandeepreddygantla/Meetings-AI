# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meetings AI is a Flask-based document analysis and chat application that processes meeting documents using OpenAI/Azure OpenAI LLM technologies. Features modular architecture with AI-powered document processing, semantic search, and conversational interfaces.

## Development Commands

### Running the Application
```bash
python flask_app.py
# Visit: http://127.0.0.1:5000/meetingsai/
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file:
OPENAI_API_KEY=your_api_key                 # For OpenAI
SECRET_KEY=your-secure-random-key           # Required for Flask sessions
BASE_PATH=/meetingsai                       # Optional, defaults to /meetingsai

# OR for Azure:
# AZURE_CLIENT_ID=your_azure_client_id
# AZURE_CLIENT_SECRET=your_azure_client_secret  
# AZURE_PROJECT_ID=your_azure_project_id

# Create tiktoken cache directory
mkdir tiktoken_cache
```

### Database Operations
Uses SQLite + FAISS hybrid storage:
- `meeting_documents.db` - SQLite for metadata, users, projects
- `vector_index.faiss` - FAISS for semantic embeddings
- `sessions.db` - Session storage

```bash
# Reset vector database
rm vector_index.faiss  # Auto-rebuilds on restart
```

## Architecture Overview

### Dual Architecture Pattern
Two parallel implementations:
1. **Legacy Monolithic** (`meeting_processor.py`): Original implementation
2. **Modular Architecture** (`src/` directory): Clean separation of concerns

### Core Components
- `flask_app.py` - Main Flask application entry point
- `meeting_processor.py` - Global AI client variables: `access_token`, `embedding_model`, `llm`
- `src/database/` - DatabaseManager (SQLite + FAISS)
- `src/services/` - Business logic: AuthService, ChatService, DocumentService, UploadService
- `src/api/` - Flask blueprints for routes
- `src/ai/` - LLM operations and query processing

### Key Design Patterns
- **Global Variable Pattern:** All AI operations use globals from `meeting_processor.py`
- **Service Composition:** Shared DatabaseManager instance across services
- **Environment Switching:** Modify initialization functions for OpenAI/Azure switching

## Critical Development Rules

### Default Project Upload Confirmation
Confirmation dialog for "Default Project" uploads:
- Shows modal for Default Project or empty selection
- "Continue Upload" button uses color `#FF612B` (orange theme)
- Modal in `templates/chat.html`, closable via ESC/outside click

### LLM Integration Requirements
**MANDATORY:** Always use global variables for AI operations:
```python
from meeting_processor import access_token, embedding_model, llm

# Always check for None before using
if llm is not None:
    response = llm.invoke(prompt)
else:
    logger.error("LLM not available - check API key configuration")
```

**NEVER instantiate directly:** `ChatOpenAI()`, `OpenAIEmbeddings()`, `AzureChatOpenAI()`

### Environment Switching Protocol
Modify only these functions in `meeting_processor.py`:
- `get_access_token()` - Return None for OpenAI, Azure token for Azure
- `get_llm()` - Return ChatOpenAI or AzureChatOpenAI  
- `get_embedding_model()` - Return OpenAIEmbeddings or AzureOpenAIEmbeddings

### Database Access Pattern
Always use DatabaseManager:
```python
db_manager = DatabaseManager()
documents = db_manager.get_all_documents(user_id)
```

## IIS Deployment & Performance

### IIS Configuration
**web.config requirements:**
- **WSGI Handler:** `flask_app.app`
- **Python Path:** Application root directory
- **Base Path:** Routes support `/meetingsai` prefix via `BASE_PATH`

### Performance Patterns
- **Vector Operations:** Batch processing (100 vectors/batch) for memory efficiency
- **Tiktoken Cache:** Directory at `tiktoken_cache/` for token caching
- **Connection Pooling:** SQLite connections with WAL mode
- **Session Management:** Custom SQLite session backend for IIS compatibility

## Frontend Architecture

### Mention System
Located in `static/js/modules/mentions.js`:
- `@project:name` - Filter by project
- `@meeting:name` - Filter by meeting  
- `@date:today|yesterday|YYYY-MM-DD` - Date filtering
- `#folder` - Folder navigation
- `#folder>` - Show folder contents

### Upload Modal
- Event listeners set when modal opens (not page load)
- Supports click, drag/drop, file selection
- Prevents duplicate event listeners

## File Processing Pipeline
1. **Upload & Validation:** File type validation, SHA-256 deduplication
2. **Content Extraction:** .docx, .pdf, .txt support with fallbacks
3. **AI Analysis:** LLM metadata extraction (topics, participants, decisions)
4. **Chunking & Embedding:** RecursiveCharacterTextSplitter + text-embedding-3-large
5. **Storage:** SQLite metadata + FAISS vector storage
6. **Background Processing:** ThreadPoolExecutor with job tracking

## Environment Variables
```bash
# AI Configuration
OPENAI_API_KEY=sk-...                    # For OpenAI
# OR for Azure:
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
AZURE_PROJECT_ID=...

# Application Configuration  
BASE_PATH=/meetingsai                   # Route prefix
SECRET_KEY=your-flask-secret-key        # Flask sessions
TIKTOKEN_CACHE_DIR=tiktoken_cache       # Token caching
```

## Troubleshooting Common Issues

### Vector Database Sync Problems
If queries return "no relevant information":
```bash
rm vector_index.faiss  # Force rebuild from SQLite on restart
```

### Enhanced Search Issues
- Check user_id filtering in `src/database/manager.py`
- Monitor logs for "Enhanced search returned 0 results"
- Ensure enhanced processing for document-specific queries

### LLM Initialization Failures  
- Verify environment variables are set correctly
- Check `logs/flask_app.log` for initialization errors
- Ensure tiktoken cache directory exists and is writable

### Logging
- **Main app logs**: `logs/flask_app.log`
- **Processor logs**: `logs/meeting_processor.log`
- **Log levels**: INFO (default)

## Development Patterns

### Notification System
Manual-close notifications in `static/script.js`:
- No auto-dismiss, stacking with proper positioning  
- ESC key support, close button with hover effects

### Service Composition
Services share DatabaseManager instance:
```python
db_manager = DatabaseManager()
services = {
    'auth': AuthService(db_manager),
    'chat': ChatService(db_manager, processor),
    'document': DocumentService(db_manager),
    'upload': UploadService(db_manager, processor)
}
```

## Development Commands

### Database Inspection
```bash
# Basic database checks
sqlite3 meeting_documents.db "SELECT COUNT(*) FROM documents;"
python -c "import faiss; print(f'Vectors: {faiss.read_index('vector_index.faiss').ntotal}')"

# Validate environment setup
python -c "
from meeting_processor import access_token, embedding_model, llm
print(f'LLM: {\"✓\" if llm else \"✗\"}')
print(f'Embedding: {\"✓\" if embedding_model else \"✗\"}')"
```

## Processing Strategy

### Dual Processing (Enhanced vs Legacy)
`ChatService` uses enhanced processing for:
- Queries expecting 10+ documents
- Complex multi-meeting summaries
- Date range queries spanning multiple meetings
- Project-wide analysis requests

Legacy processing for:
- Simple document lookups
- Single meeting questions

