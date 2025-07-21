# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meetings AI is a Flask-based document analysis and chat application that processes meeting documents using AI/LLM technologies. The application has undergone a major refactoring from a monolithic structure to a modular, industry-standard architecture while maintaining IIS compatibility.

## Development Commands

### Running the Application
```bash
# Development server (refactored version)
python flask_app_refactored.py

# Original version (legacy)
python flask_app.py

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
```

### Database Operations
The application uses a hybrid approach with FAISS for vector search and SQLite for metadata. Database initialization is automatic on first run.

## Architecture Overview

### Core Components

**Entry Points:**
- `flask_app.py` - Original monolithic version (kept for IIS compatibility)
- `flask_app_refactored.py` - New modular entry point
- `meeting_processor.py` - Legacy processor (maintained for backwards compatibility)

**Modular Architecture (`src/` directory):**
- **Config Layer** (`src/config/`) - Dynamic configuration management with environment-based settings
- **Database Layer** (`src/database/`) - Split into VectorOperations (FAISS), SQLiteOperations, and unified DatabaseManager
- **Service Layer** (`src/services/`) - Business logic for Auth, Chat, Document, and Upload operations
- **API Layer** (`src/api/`) - Flask blueprint-based route organization
- **AI Layer** (`src/ai/`) - LLM client management, embeddings, and query processing
- **Models** (`src/models/`) - Data classes for User and Document entities
- **Utils** (`src/utils/`) - Shared utilities and validation

### Key Design Patterns

**Service-Oriented Architecture:** Each major feature area has a dedicated service class that encapsulates business logic and coordinates between database and API layers.

**Dependency Injection:** Services are initialized with database manager instances and passed to API routes, enabling clean separation and testability.

**Dynamic Path Configuration:** All routes and static paths are configurable via `BASE_PATH` environment variable (defaults to `/meetingsai`), supporting both development and production deployments.

## Critical Development Rules

### LLM Integration Requirements (from instructions.md)
**NEVER** instantiate LLM or embedding models directly. Always use these global variables:
```python
from src.ai.llm_client import access_token, embedding_model, llm
```

These are initialized once at application startup via:
```python
access_token = get_access_token()
embedding_model = get_embedding_model(access_token)
llm = get_llm(access_token)
```

**Forbidden patterns:**
```python
# DON'T DO THIS
ChatOpenAI(...)
OpenAIEmbeddings(...)
AzureChatOpenAI(...)
```

**Correct usage:**
```python
# DO THIS
response = llm.invoke(...)
embeddings = embedding_model.embed_documents(...)
```

### IIS Deployment Constraints
- Entry point MUST be `flask_app.py` for IIS compatibility
- WSGI handler is `flask_app.app` (defined in web.config)
- All routes must work with `/meetingsai` base path prefix
- Session management uses custom SQLite backend for WFASTCGI compatibility

## Frontend Architecture

### Mention System (@/# functionality)
The application features an advanced mention system extracted to `static/js/modules/mentions.js`:
- `@project:name` - Filter by specific project
- `@meeting:name` - Filter by specific meeting  
- `@date:today|yesterday|YYYY-MM-DD` - Date-based filtering
- `#folder` - Navigate to folder
- `#folder>` - Show files in folder

The MentionHandler class manages dropdown UI, API interactions, and message parsing.

### Configuration Management
Frontend configuration is centralized in `static/js/config.js` with dynamic base path support matching the backend configuration.

## Database Schema
- **SQLite:** User management, projects, meetings, document metadata, file hashes, upload jobs, sessions
- **FAISS:** Vector embeddings for semantic document search
- **Unified Access:** DatabaseManager class provides single interface to both systems

## File Processing Pipeline
1. File upload validation and deduplication
2. Content extraction (supports .docx, .pdf, .txt)
3. AI-powered content analysis and metadata extraction
4. Text chunking and embedding generation
5. Vector storage with metadata linking
6. Background processing with job tracking

## Environment Variables
- `BASE_PATH` - Application base path (default: `/meetingsai`)
- `OPENAI_API_KEY` - OpenAI API key for LLM/embeddings
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint (if using Azure)
- `SECRET_KEY` - Flask session secret
- `UPLOAD_FOLDER` - File upload directory (default: `uploads`)
- `MAX_FILE_SIZE` - Maximum upload size (default: 100MB)

## Migration Notes
The codebase supports both legacy (`flask_app.py`) and refactored (`flask_app_refactored.py`) entry points. The refactored version provides better maintainability while the original ensures deployment compatibility. Both versions access the same database and provide identical functionality.