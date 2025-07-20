# UHG Meeting Document AI - Technical Overview

## Executive Summary

The UHG Meeting Document AI is a sophisticated web-based application designed to process, store, and intelligently query meeting documents. It leverages advanced AI technologies including OpenAI's GPT-4 and embeddings, combined with vector databases to provide contextual search and analysis of meeting content.

## System Architecture

### High-Level Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Web Browser       │────▶│   Flask Web App     │────▶│   Azure OpenAI      │
│   (Frontend)        │◀────│   (Backend)         │◀────│   (GPT-4 & Embed)   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │   Databases   │
                            ├───────────────┤
                            │  SQLite (Meta)│
                            │  FAISS (Vector)│
                            └───────────────┘
```

### Core Components

1. **Frontend Layer**
   - HTML/CSS/JavaScript interface
   - Real-time chat functionality
   - Document upload management
   - Project and meeting organization

2. **Backend Layer (Flask)**
   - RESTful API endpoints
   - Authentication and session management
   - Document processing pipeline
   - Query orchestration

3. **AI Layer**
   - OpenAI GPT-4 for natural language understanding
   - OpenAI text-embedding-3-large for vector embeddings
   - FAISS for efficient vector similarity search

4. **Data Layer**
   - SQLite for structured metadata
   - FAISS for vector embeddings
   - File system for document storage

## Database Schema

### SQLite Tables

```sql
1. users
   - user_id (PRIMARY KEY)
   - username (UNIQUE)
   - email (UNIQUE)
   - password_hash
   - created_at
   - last_login

2. projects
   - project_id (PRIMARY KEY)
   - user_id (FOREIGN KEY)
   - project_name
   - description
   - created_at
   - updated_at

3. meetings
   - meeting_id (PRIMARY KEY)
   - user_id (FOREIGN KEY)
   - project_id (FOREIGN KEY)
   - meeting_name
   - date
   - created_at

4. documents
   - document_id (PRIMARY KEY)
   - filename
   - date
   - folder_path
   - content_summary
   - topics
   - participants
   - user_id (FOREIGN KEY)
   - project_id (FOREIGN KEY)
   - meeting_id (FOREIGN KEY)
   - created_at
   - updated_at
   - chunk_count

5. chunks
   - chunk_id (PRIMARY KEY)
   - document_id (FOREIGN KEY)
   - filename
   - chunk_index
   - content
   - embedding_id
   - metadata
   - user_id (FOREIGN KEY)
   - project_id (FOREIGN KEY)

6. file_hashes
   - hash_id (PRIMARY KEY)
   - filename
   - original_filename
   - sha256_hash
   - user_id (FOREIGN KEY)
   - project_id (FOREIGN KEY)
   - created_at

7. upload_jobs
   - job_id (PRIMARY KEY)
   - user_id (FOREIGN KEY)
   - project_id
   - total_files
   - processed_files
   - failed_files
   - status
   - error_message
   - created_at
   - updated_at

8. file_processing_status
   - status_id (PRIMARY KEY)
   - job_id (FOREIGN KEY)
   - filename
   - status
   - error_message
   - created_at
   - updated_at
```

## Workflow & Data Flow

### 1. Document Upload Flow (Based on Sample Meeting)

Using the sample meeting from July 14, 2025, let's trace the complete workflow:

```
Step 1: User Authentication
├── User logs in (e.g., Sandeep Reddy)
└── Session created with user_id

Step 2: Project Selection
├── User creates/selects project (e.g., "Print Migration")
└── Project metadata stored in SQLite

Step 3: Document Upload
├── User uploads meeting recording (e.g., "Document Fulfillment AIML-20250714_153021-Meeting Recording.docx")
├── File validation (format, size, duplicates)
├── SHA-256 hash generated for deduplication
└── Background job created for processing

Step 4: Document Processing
├── Extract text from document
├── Extract metadata:
│   ├── Date: July 14, 2025
│   ├── Participants: Sandeep Reddy, Joseph Mize, Kevin Vautrinot, Adrian Smithee, Ganesh K K
│   └── Topics: Login credential setup, Server procurement, OpenAI APIs, Vector embeddings
├── Split into chunks (2000 character limit)
├── Generate embeddings for each chunk using OpenAI
└── Store in databases:
    ├── Metadata → SQLite
    └── Embeddings → FAISS vector index

Step 5: Query Processing
├── User asks: "Who is responsible for server procurement?"
├── Query Analysis:
│   ├── Extract intent and keywords
│   ├── Identify context (project, date range, participants)
│   └── Generate query embedding
├── Retrieval:
│   ├── SQL search for keyword matches
│   ├── Vector search for semantic similarity
│   └── Combine and rank results
└── Response Generation:
    ├── Pass relevant chunks to GPT-4
    └── Generate contextual answer: "Michael W Gwin is responsible for server procurement"
```

### 2. Query Processing Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Processor    │
├─────────────────────┤
│ • Parse @mentions   │
│ • Extract filters   │
│ • Generate embedding│
└─────────────────────┘
    │
    ├────────────┐
    ▼            ▼
┌──────────┐  ┌──────────┐
│SQL Search│  │Vector    │
│(Keywords)│  │Search    │
└──────────┘  └──────────┘
    │            │
    └──────┬─────┘
           ▼
    ┌─────────────┐
    │  Reranking  │
    │  & Merging  │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   GPT-4     │
    │  Response   │
    └─────────────┘
```

### 3. Real-World Example Flow

Based on the sample meeting transcript:

**Upload Example:**
```
1. File: "Document Fulfillment AIML-20250714_153021-Meeting Recording.docx"
2. Extracted Metadata:
   - Date: 2025-07-14
   - Duration: 20m 42s
   - Participants: 5 people
   - Key Topics: AI tool deployment, server procurement, OpenAI integration

3. Processing:
   - 281 lines of text → ~17 chunks
   - Each chunk gets a 3072-dimensional embedding
   - Stored with metadata for retrieval
```

**Query Examples:**

1. **Date-based Query**: "What happened in the July 14 meeting?"
   ```
   SQL: SELECT * FROM documents WHERE date = '2025-07-14'
   Vector: Search for meeting summary embeddings
   Result: Complete meeting summary with participants and key decisions
   ```

2. **Person-based Query**: "What did Sandeep discuss?"
   ```
   SQL: SELECT * FROM chunks WHERE content LIKE '%Sandeep%'
   Vector: Semantic search for Sandeep's contributions
   Result: Sandeep discussed local deployment, server requirements, and gave demo
   ```

3. **Topic-based Query**: "Tell me about the OpenAI integration"
   ```
   SQL: SELECT * FROM chunks WHERE content LIKE '%OpenAI%' OR '%API%'
   Vector: Semantic search for API and integration concepts
   Result: Details about GPT-4, embeddings, token usage, and Azure OpenAI setup
   ```

## Technical Implementation Details

### Key Technologies

1. **Backend**
   - Flask 3.1.0 (Web framework)
   - SQLite3 (Metadata storage)
   - FAISS (Facebook AI Similarity Search)
   - Python 3.12

2. **AI/ML**
   - OpenAI GPT-4 (128k context window)
   - text-embedding-3-large model
   - LangChain for orchestration

3. **Frontend**
   - Vanilla JavaScript (ES6+)
   - Markdown rendering
   - Real-time updates

4. **Document Processing**
   - python-docx (Word documents)
   - PyPDF2 (PDF documents)
   - Custom text extraction

### Security Features

1. **Authentication**
   - Bcrypt password hashing
   - Session-based authentication
   - User isolation

2. **Data Security**
   - User-specific data isolation
   - Project-based access control
   - File deduplication

### Performance Optimizations

1. **Batch Processing**
   - Parallel document processing
   - Background job queues
   - Chunked embeddings generation

2. **Caching**
   - FAISS index caching
   - Session caching
   - Query result caching

3. **Efficient Retrieval**
   - Hybrid search (SQL + Vector)
   - Intelligent chunk sizing
   - Relevance ranking

## Deployment Architecture

### Current Setup (Local Development)
```
localhost:5000
├── Flask Development Server
├── SQLite Database
├── FAISS Index (local file)
└── Document Storage (local filesystem)
```

### Production Setup (Windows Server/IIS)
```
IIS Web Server
├── FastCGI Python Handler
├── Flask Application
├── SQL Database
├── FAISS Index (network storage)
└── Document Storage (network share)
```

### Azure Integration
- Azure OpenAI Service (via Optum AI Studio)
- Azure Active Directory (future)
- Azure Storage (future)

## API Endpoints

### Authentication
- `POST /meetingsai/register` - User registration
- `POST /meetingsai/login` - User login
- `POST /meetingsai/logout` - User logout

### Document Management
- `POST /meetingsai/api/upload` - Upload documents
- `GET /meetingsai/api/documents` - List documents
- `GET /meetingsai/api/projects` - List projects
- `POST /meetingsai/api/projects` - Create project

### Chat/Query
- `POST /meetingsai/api/chat` - Process queries
- `GET /meetingsai/api/conversations` - Get chat history

### Statistics
- `GET /meetingsai/api/stats` - System statistics
- `GET /meetingsai/api/job/<job_id>` - Upload job status

## Future Enhancements

1. **Automation**
   - Direct Teams integration
   - Automatic meeting recording ingestion
   - Scheduled processing

2. **Advanced Features**
   - Multi-language support
   - Voice query interface
   - Export capabilities

3. **Scalability**
   - Distributed FAISS
   - PostgreSQL migration
   - Kubernetes deployment

## Conclusion

The UHG Meeting Document AI represents a sophisticated integration of traditional database technologies with cutting-edge AI capabilities. By combining structured metadata storage with vector embeddings, the system provides both precise keyword matching and semantic understanding of meeting content. The architecture is designed to scale from local development to enterprise deployment while maintaining security and performance.