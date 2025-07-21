"""
SQLite database operations for meeting document storage and management.
This module handles all SQLite-related operations extracted from the VectorDatabase class.
"""

import os
import sqlite3
import logging
import hashlib
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Import models and global variables from the main module  
from meeting_processor import (
    access_token, embedding_model, llm,
    DocumentChunk, User, Project, Meeting, MeetingDocument
)

logger = logging.getLogger(__name__)


class SQLiteOperations:
    """Handles all SQLite database operations"""
    
    def __init__(self, db_path: str = "meeting_documents.db"):
        """
        Initialize SQLite operations
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _safe_json_loads(self, json_str: str) -> list:
        """Safely parse JSON string, return empty list if parsing fails"""
        if not json_str:
            return []
        try:
            result = json.loads(json_str)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse JSON: {json_str}, returning empty list")
            return []
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                role TEXT DEFAULT 'user'
            )
        ''')
        
        # Check if we need to migrate existing tables
        self._migrate_existing_tables(cursor)
        
        # Create projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create meetings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meetings (
                meeting_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                meeting_name TEXT NOT NULL,
                meeting_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')
        
        # Create user sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create documents table (updated with user context)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                date TIMESTAMP NOT NULL,
                title TEXT,
                content_summary TEXT,
                main_topics TEXT,
                past_events TEXT,
                future_actions TEXT,
                participants TEXT,
                chunk_count INTEGER,
                file_size INTEGER,
                user_id TEXT,
                meeting_id TEXT,
                project_id TEXT,
                folder_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id),
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')
        
        # Create chunks table (updated with user context)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                user_id TEXT,
                meeting_id TEXT,
                project_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (document_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id),
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')
        
        # Create file_hashes table for deduplication
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_hashes (
                hash_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                user_id TEXT NOT NULL,
                project_id TEXT,
                meeting_id TEXT,
                document_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (project_id) REFERENCES projects (project_id),
                FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id),
                FOREIGN KEY (document_id) REFERENCES documents (document_id),
                UNIQUE(file_hash, user_id)
            )
        ''')
        
        # Create upload_jobs table for tracking background processing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS upload_jobs (
                job_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT,
                meeting_id TEXT,
                total_files INTEGER NOT NULL,
                processed_files INTEGER DEFAULT 0,
                failed_files INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (project_id) REFERENCES projects (project_id),
                FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
            )
        ''')
        
        # Create file_processing_status table for individual file tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_processing_status (
                status_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                document_id TEXT,
                chunks_created INTEGER DEFAULT 0,
                processing_started_at TIMESTAMP,
                processing_completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES upload_jobs (job_id),
                FOREIGN KEY (document_id) REFERENCES documents (document_id)
            )
        ''')
        
        # Create indexes for faster searches
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_date ON documents(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_filename ON documents(filename)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_user ON documents(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_project ON documents(project_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_meeting ON documents(meeting_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_document ON chunks(document_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_user ON chunks(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_meetings_user ON meetings(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_meetings_project ON meetings(project_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)')
        
        conn.commit()
        conn.close()
    
    def _migrate_existing_tables(self, cursor):
        """Migrate existing tables to support multi-user structure and enhanced intelligence"""
        try:
            # Check if documents table exists and needs migration
            cursor.execute("PRAGMA table_info(documents)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'user_id' not in columns:
                logger.info("Migrating documents table to support multi-user...")
                cursor.execute('ALTER TABLE documents ADD COLUMN user_id TEXT')
                cursor.execute('ALTER TABLE documents ADD COLUMN meeting_id TEXT')
                cursor.execute('ALTER TABLE documents ADD COLUMN project_id TEXT')
                logger.info("Documents table migrated successfully")
            
            # Check if documents table needs folder_path column
            if 'folder_path' not in columns:
                logger.info("Adding folder_path column to documents table...")
                cursor.execute('ALTER TABLE documents ADD COLUMN folder_path TEXT')
                logger.info("folder_path column added successfully")
            
            # Check if chunks table exists and needs migration
            cursor.execute("PRAGMA table_info(chunks)")
            chunk_columns = [column[1] for column in cursor.fetchall()]
            
            if 'user_id' not in chunk_columns:
                logger.info("Migrating chunks table to support multi-user...")
                cursor.execute('ALTER TABLE chunks ADD COLUMN user_id TEXT')
                cursor.execute('ALTER TABLE chunks ADD COLUMN meeting_id TEXT')
                cursor.execute('ALTER TABLE chunks ADD COLUMN project_id TEXT')
                logger.info("Chunks table migrated successfully")
            
            # Enhanced intelligence metadata migration
            intelligence_columns = [
                'enhanced_content', 'chunk_type', 'speakers', 'speaker_contributions',
                'topics', 'decisions', 'actions', 'questions', 'context_before',
                'context_after', 'key_phrases', 'importance_score'
            ]
            
            missing_intelligence_columns = [col for col in intelligence_columns if col not in chunk_columns]
            
            if missing_intelligence_columns:
                logger.info("Adding enhanced intelligence metadata columns to chunks table...")
                for column in missing_intelligence_columns:
                    if column == 'importance_score':
                        cursor.execute(f'ALTER TABLE chunks ADD COLUMN {column} REAL DEFAULT 0.5')
                    else:
                        cursor.execute(f'ALTER TABLE chunks ADD COLUMN {column} TEXT')
                logger.info(f"Added {len(missing_intelligence_columns)} intelligence metadata columns")
                logger.info("Enhanced intelligence migration completed successfully")
                
        except sqlite3.OperationalError as e:
            # Tables might not exist yet, that's okay
            logger.info(f"Migration check: {e}")
            pass
    
    # Document Operations
    def store_document_metadata(self, filename: str, content: str, user_id: str, 
                              project_id: str = None, meeting_id: str = None) -> str:
        """Store document metadata and return document_id"""
        document_id = str(uuid.uuid4())
        
        try:
            from meeting_processor import EnhancedMeetingDocumentProcessor
            processor = EnhancedMeetingDocumentProcessor()
            
            # Extract date from filename first
            try:
                doc_date = processor.extract_date_from_filename(filename, content)
            except:
                doc_date = datetime.now()
            
            # Create summary of the content
            content_summary = processor.create_content_summary(content)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO documents (document_id, filename, date, content_summary, 
                                     user_id, project_id, meeting_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (document_id, filename, doc_date, content_summary, 
                  user_id, project_id, meeting_id, datetime.now()))
            
            conn.commit()
            conn.close()
            
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document metadata: {e}")
            raise
    
    def add_document_and_chunks(self, document, chunks: List):
        """Add document and its chunks to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert document metadata
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (document_id, filename, date, title, content_summary, main_topics, 
                 past_events, future_actions, participants, chunk_count, file_size,
                 user_id, meeting_id, project_id, folder_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document.document_id, document.filename, document.date, document.title,
                document.content_summary, json.dumps(document.main_topics) if document.main_topics else '[]', 
                json.dumps(document.past_events) if document.past_events else '[]',
                json.dumps(document.future_actions) if document.future_actions else '[]', 
                json.dumps(document.participants) if document.participants else '[]', len(chunks), 
                len(document.content), document.user_id, document.meeting_id,
                document.project_id, document.folder_path, datetime.now()
            ))
            
            # Prepare chunk data for bulk insert
            chunk_data_with_context = []
            for chunk in chunks:
                chunk_data = (
                    chunk.chunk_id, chunk.document_id, chunk.filename, chunk.chunk_index,
                    chunk.enhanced_content or chunk.content,
                    chunk.start_char, chunk.end_char,
                    chunk.user_id, chunk.meeting_id, chunk.project_id,
                    datetime.now()
                )
                chunk_data_with_context.append(chunk_data)
            
            # Bulk insert chunks
            cursor.executemany('''
                INSERT OR REPLACE INTO chunks 
                (chunk_id, document_id, filename, chunk_index, content, 
                 start_char, end_char, user_id, meeting_id, project_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', chunk_data_with_context)
            
            conn.commit()
            logger.info(f"Added document {document.filename} with {len(chunks)} chunks")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding document {document.filename}: {e}")
            raise
        finally:
            conn.close()
    
    def get_chunks_by_ids(self, chunk_ids: List[str]):
        """Retrieve chunks by their IDs with enhanced intelligence metadata"""
        if not chunk_ids:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get chunks data
            placeholders = ','.join(['?' for _ in chunk_ids])
            cursor.execute(f'''
                SELECT c.chunk_id, c.document_id, c.filename, c.chunk_index, c.content,
                       c.start_char, c.end_char, c.user_id, c.meeting_id, c.project_id,
                       d.date, d.title, d.content_summary, d.main_topics, d.past_events,
                       d.future_actions, d.participants
                FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.document_id
                WHERE c.chunk_id IN ({placeholders})
                ORDER BY c.document_id, c.chunk_index
            ''', chunk_ids)
            
            results = cursor.fetchall()
            conn.close()
            
            # Convert to DocumentChunk objects
            chunks = []
# DocumentChunk already imported at the top
            
            for row in results:
                chunk = DocumentChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    filename=row[2],
                    chunk_index=row[3],
                    content=row[4],
                    start_char=row[5],
                    end_char=row[6],
                    user_id=row[7],
                    meeting_id=row[8],
                    project_id=row[9],
                    # Document metadata
                    date=datetime.fromisoformat(row[10]) if row[10] else None,
                    document_title=row[11],
                    content_summary=row[12],
                    main_topics=self._safe_json_loads(row[13]),
                    past_events=self._safe_json_loads(row[14]),
                    future_actions=self._safe_json_loads(row[15]),
                    participants=self._safe_json_loads(row[16])
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def get_documents_by_timeframe(self, timeframe: str, user_id: str = None):
        """Get documents filtered by intelligent timeframe calculation"""
        try:
            from meeting_processor import EnhancedMeetingDocumentProcessor
            processor = EnhancedMeetingDocumentProcessor()
            start_date, end_date = processor._calculate_date_range(timeframe)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query_params = []
            base_query = '''
                SELECT document_id, filename, date, title, content_summary, 
                       main_topics, past_events, future_actions, participants,
                       user_id, meeting_id, project_id, folder_path
                FROM documents
                WHERE 1=1
            '''
            
            if user_id:
                base_query += " AND user_id = ?"
                query_params.append(user_id)
            
            if start_date:
                base_query += " AND date >= ?"
                query_params.append(start_date.isoformat())
            
            if end_date:
                base_query += " AND date <= ?"
                query_params.append(end_date.isoformat())
            
            base_query += " ORDER BY date DESC"
            
            cursor.execute(base_query, query_params)
            results = cursor.fetchall()
            conn.close()
            
            # Convert to MeetingDocument objects
            documents = []
# MeetingDocument already imported at the top
            
            for row in results:
                doc = MeetingDocument(
                    document_id=row[0],
                    filename=row[1],
                    date=datetime.fromisoformat(row[2]),
                    title=row[3],
                    content_summary=row[4],
                    main_topics=self._safe_json_loads(row[5]),
                    past_events=self._safe_json_loads(row[6]),
                    future_actions=self._safe_json_loads(row[7]),
                    participants=self._safe_json_loads(row[8]),
                    user_id=row[9],
                    meeting_id=row[10],
                    project_id=row[11],
                    folder_path=row[12],
                    content=""  # Content not loaded for performance
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents by timeframe: {e}")
            return []
    
    def keyword_search_chunks(self, keywords: List[str], limit: int = 50) -> List[str]:
        """Perform keyword search on chunk content"""
        if not keywords:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build search query for keywords
            keyword_conditions = []
            query_params = []
            
            for keyword in keywords:
                keyword_conditions.append("content LIKE ?")
                query_params.append(f"%{keyword}%")
            
            query = f'''
                SELECT chunk_id FROM chunks
                WHERE {' OR '.join(keyword_conditions)}
                ORDER BY chunk_index
                LIMIT ?
            '''
            query_params.append(limit)
            
            cursor.execute(query, query_params)
            results = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in results]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def get_all_documents(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get all documents with metadata for document selection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if user_id:
                cursor.execute('''
                    SELECT document_id, filename, date, title, content_summary,
                           chunk_count, file_size, user_id, project_id, meeting_id, folder_path
                    FROM documents 
                    WHERE user_id = ?
                    ORDER BY date DESC, filename
                ''', (user_id,))
            else:
                cursor.execute('''
                    SELECT document_id, filename, date, title, content_summary,
                           chunk_count, file_size, user_id, project_id, meeting_id, folder_path
                    FROM documents 
                    ORDER BY date DESC, filename
                ''')
            
            results = cursor.fetchall()
            conn.close()
            
            documents = []
            for row in results:
                doc = {
                    'document_id': row[0],
                    'filename': row[1],
                    'date': row[2],
                    'title': row[3],
                    'content_summary': row[4],
                    'chunk_count': row[5] or 0,
                    'file_size': row[6] or 0,
                    'user_id': row[7],
                    'project_id': row[8],
                    'meeting_id': row[9],
                    'folder_path': row[10]
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []
    
    # User Management Operations
    def create_user(self, username: str, email: str, full_name: str, password_hash: str) -> str:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (user_id, username, email, full_name, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, email, full_name, password_hash, datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created new user: {username}")
            return user_id
            
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed - duplicate username or email: {e}")
            raise ValueError("Username or email already exists")
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    def get_user_by_username(self, username: str):
        """Get user by username"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, username, email, full_name, password_hash, 
                       created_at, last_login, is_active, role
                FROM users 
                WHERE username = ? AND is_active = TRUE
            ''', (username,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
# User already imported at the top
                return User(
                    user_id=result[0],
                    username=result[1],
                    email=result[2],
                    full_name=result[3],
                    password_hash=result[4],
                    created_at=datetime.fromisoformat(result[5]) if result[5] else None,
                    last_login=datetime.fromisoformat(result[6]) if result[6] else None,
                    is_active=bool(result[7]),
                    role=result[8]
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def get_user_by_id(self, user_id: str):
        """Get user by user_id"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, username, email, full_name, password_hash, 
                       created_at, last_login, is_active, role
                FROM users 
                WHERE user_id = ? AND is_active = TRUE
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
# User already imported at the top
                return User(
                    user_id=result[0],
                    username=result[1],
                    email=result[2],
                    full_name=result[3],
                    password_hash=result[4],
                    created_at=datetime.fromisoformat(result[5]) if result[5] else None,
                    last_login=datetime.fromisoformat(result[6]) if result[6] else None,
                    is_active=bool(result[7]),
                    role=result[8]
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def update_user_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users SET last_login = ? WHERE user_id = ?
            ''', (datetime.now(), user_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating user last login: {e}")
    
    # Project Management Operations
    def create_project(self, user_id: str, project_name: str, description: str = "") -> str:
        """Create a new project for a user"""
        project_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO projects (project_id, user_id, project_name, description, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (project_id, user_id, project_name, description, datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created new project: {project_name} for user {user_id}")
            return project_id
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise
    
    def get_user_projects(self, user_id: str):
        """Get all projects for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT project_id, user_id, project_name, description, created_at, is_active
                FROM projects 
                WHERE user_id = ? AND is_active = TRUE
                ORDER BY created_at DESC
            ''', (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            projects = []
# Project already imported at the top
            
            for row in results:
                project = Project(
                    project_id=row[0],
                    user_id=row[1],
                    project_name=row[2],
                    description=row[3],
                    created_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    is_active=bool(row[5])
                )
                projects.append(project)
            
            return projects
            
        except Exception as e:
            logger.error(f"Error getting user projects: {e}")
            return []
    
    def get_project_by_id(self, project_id: str):
        """Get a specific project by its ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT project_id, user_id, project_name, description, created_at, is_active
                FROM projects 
                WHERE project_id = ? AND is_active = TRUE
            ''', (project_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                project = Project(
                    project_id=row[0],
                    user_id=row[1],
                    project_name=row[2],
                    description=row[3],
                    created_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    is_active=bool(row[5])
                )
                return project
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting project by ID {project_id}: {e}")
            return None
    
    # Meeting Management Operations
    def create_meeting(self, user_id: str, project_id: str, meeting_name: str, meeting_date: datetime) -> str:
        """Create a new meeting"""
        meeting_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO meetings (meeting_id, user_id, project_id, meeting_name, meeting_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (meeting_id, user_id, project_id, meeting_name, meeting_date, datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created new meeting: {meeting_name}")
            return meeting_id
            
        except Exception as e:
            logger.error(f"Error creating meeting: {e}")
            raise
    
    def get_user_meetings(self, user_id: str, project_id: str = None):
        """Get meetings for a user, optionally filtered by project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if project_id:
                cursor.execute('''
                    SELECT meeting_id, user_id, project_id, meeting_name, meeting_date, created_at
                    FROM meetings 
                    WHERE user_id = ? AND project_id = ?
                    ORDER BY meeting_date DESC, created_at DESC
                ''', (user_id, project_id))
            else:
                cursor.execute('''
                    SELECT meeting_id, user_id, project_id, meeting_name, meeting_date, created_at
                    FROM meetings 
                    WHERE user_id = ?
                    ORDER BY meeting_date DESC, created_at DESC
                ''', (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            meetings = []
# Meeting already imported at the top
            
            for row in results:
                meeting = Meeting(
                    meeting_id=row[0],
                    user_id=row[1],
                    project_id=row[2],
                    meeting_name=row[3],
                    meeting_date=datetime.fromisoformat(row[4]) if row[4] else None,
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None
                )
                meetings.append(meeting)
            
            return meetings
            
        except Exception as e:
            logger.error(f"Error getting user meetings: {e}")
            return []
    
    # Session Management Operations
    def create_session(self, user_id: str, session_id: str, expires_at: datetime) -> bool:
        """Create a new user session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_sessions (session_id, user_id, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, TRUE)
            ''', (session_id, user_id, datetime.now(), expires_at))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate a session and return user_id if valid"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id FROM user_sessions 
                WHERE session_id = ? AND is_active = TRUE AND expires_at > ?
            ''', (session_id, datetime.now()))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions SET is_active = FALSE WHERE session_id = ?
            ''', (session_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating session: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions 
                SET is_active = FALSE 
                WHERE expires_at <= ? AND is_active = TRUE
            ''', (datetime.now(),))
            
            cleaned_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired sessions")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    # File Management Operations
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def is_file_duplicate(self, file_hash: str, filename: str, user_id: str) -> Optional[Dict]:
        """Check if file is a duplicate based on hash and return original file info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT fh.original_filename, fh.created_at, d.document_id, d.filename as stored_filename
                FROM file_hashes fh
                LEFT JOIN documents d ON fh.document_id = d.document_id
                WHERE fh.file_hash = ? AND fh.user_id = ?
                ORDER BY fh.created_at DESC
                LIMIT 1
            ''', (file_hash, user_id))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'original_filename': result[0],
                    'upload_date': result[1],
                    'document_id': result[2],
                    'stored_filename': result[3]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error checking for duplicate file: {e}")
            return None
    
    def store_file_hash(self, file_hash: str, filename: str, original_filename: str, 
                       file_size: int, user_id: str, project_id: str = None, 
                       meeting_id: str = None, document_id: str = None) -> str:
        """Store file hash information for deduplication"""
        hash_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO file_hashes 
                (hash_id, filename, original_filename, file_size, file_hash, 
                 user_id, project_id, meeting_id, document_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (hash_id, filename, original_filename, file_size, file_hash,
                  user_id, project_id, meeting_id, document_id, datetime.now()))
            
            conn.commit()
            conn.close()
            
            return hash_id
            
        except sqlite3.IntegrityError:
            # Hash already exists for this user, return existing hash_id
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT hash_id FROM file_hashes 
                WHERE file_hash = ? AND user_id = ?
            ''', (file_hash, user_id))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else hash_id
        except Exception as e:
            logger.error(f"Error storing file hash: {e}")
            raise
    
    # Job Management Operations
    def create_upload_job(self, user_id: str, total_files: int, project_id: str = None, 
                         meeting_id: str = None) -> str:
        """Create a new upload job for tracking batch processing"""
        job_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO upload_jobs 
                (job_id, user_id, project_id, meeting_id, total_files, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (job_id, user_id, project_id, meeting_id, total_files, datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created upload job {job_id} for {total_files} files")
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating upload job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status and progress"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT job_id, user_id, project_id, meeting_id, total_files, 
                       processed_files, failed_files, status, error_message,
                       created_at, started_at, completed_at
                FROM upload_jobs 
                WHERE job_id = ?
            ''', (job_id,))
            
            result = cursor.fetchone()
            
            if result:
                # Get detailed file processing status
                cursor.execute('''
                    SELECT filename, status, error_message, chunks_created
                    FROM file_processing_status 
                    WHERE job_id = ?
                    ORDER BY created_at
                ''', (job_id,))
                
                files_status = cursor.fetchall()
                conn.close()
                
                return {
                    'job_id': result[0],
                    'user_id': result[1],
                    'project_id': result[2],
                    'meeting_id': result[3],
                    'total_files': result[4],
                    'processed_files': result[5],
                    'failed_files': result[6],
                    'status': result[7],
                    'error_message': result[8],
                    'created_at': result[9],
                    'started_at': result[10],
                    'completed_at': result[11],
                    'files_status': [
                        {
                            'filename': f[0],
                            'status': f[1],
                            'error_message': f[2],
                            'chunks_created': f[3]
                        } for f in files_status
                    ]
                }
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def update_job_status(self, job_id: str, status: str, processed_files: int = None, 
                         failed_files: int = None, error_message: str = None):
        """Update job status and progress"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build dynamic update query
            update_fields = []
            update_values = []
            
            update_fields.append("status = ?")
            update_values.append(status)
            
            if processed_files is not None:
                update_fields.append("processed_files = ?")
                update_values.append(processed_files)
            
            if failed_files is not None:
                update_fields.append("failed_files = ?")
                update_values.append(failed_files)
            
            if error_message is not None:
                update_fields.append("error_message = ?")
                update_values.append(error_message)
            
            if status == 'processing' and processed_files == 0:
                update_fields.append("started_at = ?")
                update_values.append(datetime.now())
            elif status in ['completed', 'failed']:
                update_fields.append("completed_at = ?")
                update_values.append(datetime.now())
            
            update_values.append(job_id)
            
            cursor.execute(f'''
                UPDATE upload_jobs 
                SET {', '.join(update_fields)}
                WHERE job_id = ?
            ''', update_values)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get counts from all tables
            stats = {}
            
            tables = ['users', 'projects', 'meetings', 'documents', 'chunks', 
                     'user_sessions', 'file_hashes', 'upload_jobs']
            
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Get active sessions count
            cursor.execute('SELECT COUNT(*) FROM user_sessions WHERE is_active = TRUE')
            stats['active_sessions'] = cursor.fetchone()[0]
            
            # Get recent activity (documents uploaded in last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM documents 
                WHERE created_at > datetime('now', '-1 day')
            ''')
            stats['documents_last_24h'] = cursor.fetchone()[0]
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}