"""
Document service for Meetings AI application.
Handles document management, processing, and metadata operations.
"""
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from werkzeug.utils import secure_filename

from src.database.manager import DatabaseManager
from src.models.document import MeetingDocument, UploadJob

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for handling document operations."""
    
    def __init__(self, db_manager: DatabaseManager, processor=None):
        """
        Initialize document service.
        
        Args:
            db_manager: Database manager instance
            processor: Document processor instance (optional for backwards compatibility)
        """
        self.db_manager = db_manager
        self.processor = processor  # For backwards compatibility with existing processor methods
    
    def get_user_documents(self, user_id: str) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Get all documents for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success, documents_list, message)
        """
        try:
            documents = self.db_manager.get_all_documents(user_id)
            return True, documents, f"Retrieved {len(documents)} documents"
        except Exception as e:
            logger.error(f"Error getting user documents: {e}")
            return False, [], f"Error retrieving documents: {str(e)}"
    
    def get_user_projects(self, user_id: str) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Get all projects for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success, projects_list, message)
        """
        try:
            projects = self.db_manager.get_user_projects(user_id)
            
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
            
            return True, project_list, f"Retrieved {len(project_list)} projects"
            
        except Exception as e:
            logger.error(f"Error getting user projects: {e}")
            return False, [], f"Error retrieving projects: {str(e)}"
    
    def create_project(self, user_id: str, project_name: str, description: str = "") -> Tuple[bool, str, Optional[str]]:
        """
        Create a new project for a user.
        
        Args:
            user_id: User ID
            project_name: Name of the project
            description: Project description
            
        Returns:
            Tuple of (success, message, project_id)
        """
        try:
            if not project_name or not project_name.strip():
                return False, 'Project name is required', None
            
            project_id = self.db_manager.create_project(user_id, project_name.strip(), description.strip())
            logger.info(f"New project created: {project_name} ({project_id}) for user {user_id}")
            
            return True, 'Project created successfully', project_id
            
        except ValueError as e:
            return False, str(e), None
        except Exception as e:
            logger.error(f"Create project error: {e}")
            return False, 'Failed to create project', None
    
    def get_user_meetings(self, user_id: str, project_id: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Get all meetings for a user, optionally filtered by project.
        
        Args:
            user_id: User ID
            project_id: Optional project ID filter
            
        Returns:
            Tuple of (success, meetings_list, message)
        """
        try:
            meetings = self.db_manager.get_user_meetings(user_id, project_id)
            
            # Convert meetings to dictionaries
            meeting_list = []
            for meeting in meetings:
                meeting_list.append({
                    'meeting_id': meeting.meeting_id,
                    'title': meeting.meeting_name,
                    'date': meeting.meeting_date.isoformat() if meeting.meeting_date else None,
                    'participants': '',  # Placeholder for future enhancement
                    'project_id': meeting.project_id,
                    'created_at': meeting.created_at.isoformat()
                })
            
            return True, meeting_list, f"Retrieved {len(meeting_list)} meetings"
            
        except Exception as e:
            logger.error(f"Error getting user meetings: {e}")
            return False, [], f"Error retrieving meetings: {str(e)}"
    
    def validate_file_upload(self, files: List[Any], project_id: Optional[str], meeting_id: Optional[str], user_id: str) -> Tuple[bool, str]:
        """
        Validate file upload parameters.
        
        Args:
            files: List of uploaded files
            project_id: Project ID (optional)
            meeting_id: Meeting ID (optional)
            user_id: User ID
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check files
            if not files or all(f.filename == '' for f in files):
                return False, 'No files selected'
            
            # Validate project belongs to user
            if project_id:
                user_projects = self.db_manager.get_user_projects(user_id)
                project_exists = any(p.project_id == project_id for p in user_projects)
                if not project_exists:
                    return False, 'Invalid project selection'
            
            # Validate meeting belongs to user and project
            if meeting_id:
                user_meetings = self.db_manager.get_user_meetings(user_id, project_id)
                meeting_exists = any(m.meeting_id == meeting_id for m in user_meetings)
                if not meeting_exists:
                    return False, 'Invalid meeting selection'
            
            return True, 'Valid'
            
        except Exception as e:
            logger.error(f"File upload validation error: {e}")
            return False, f'Validation error: {str(e)}'
    
    def prepare_upload_directory(self, user_id: str, username: str, project_id: Optional[str]) -> Tuple[bool, str, str]:
        """
        Prepare upload directory structure for user.
        
        Args:
            user_id: User ID
            username: Username
            project_id: Project ID (optional)
            
        Returns:
            Tuple of (success, upload_folder_path, error_message)
        """
        try:
            # Create user-specific directory structure
            user_folder = f"meeting_documents/user_{username}"
            
            if project_id:
                project_folder_name = "default"
                if project_id:
                    user_projects = self.db_manager.get_user_projects(user_id)
                    selected_project = next((p for p in user_projects if p.project_id == project_id), None)
                    if selected_project:
                        project_folder_name = selected_project.project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                        project_folder_name = "".join(c for c in project_folder_name if c.isalnum() or c in ("_", "-"))
                
                upload_folder = os.path.join(user_folder, f"project_{project_folder_name}")
            else:
                upload_folder = user_folder
            
            os.makedirs(upload_folder, exist_ok=True)
            return True, upload_folder, "Directory created successfully"
            
        except Exception as e:
            logger.error(f"Error preparing upload directory: {e}")
            return False, "", f"Error preparing directory: {str(e)}"
    
    def process_file_validation(self, files: List[Any], upload_folder: str, user_id: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Process and validate uploaded files.
        
        Args:
            files: List of uploaded files
            upload_folder: Upload directory path
            user_id: User ID
            
        Returns:
            Tuple of (valid_files, validation_errors, duplicates)
        """
        file_list = []
        validation_errors = []
        duplicates = []
        
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
                try:
                    file.save(file_path)
                except Exception as e:
                    validation_errors.append({
                        'filename': filename,
                        'error': f'File save error: {str(e)}'
                    })
                    continue
                
                # Check for content duplicates
                try:
                    file_hash = self.db_manager.calculate_file_hash(file_path)
                    duplicate_info = self.db_manager.is_file_duplicate(file_hash, filename, user_id)
                    
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
        
        return file_list, validation_errors, duplicates
    
    def start_background_processing(self, file_list: List[Dict[str, str]], user_id: str, project_id: Optional[str], meeting_id: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        """
        Start background processing for uploaded files.
        
        Args:
            file_list: List of validated files
            user_id: User ID
            project_id: Project ID (optional)
            meeting_id: Meeting ID (optional)
            
        Returns:
            Tuple of (success, job_id, error_message)
        """
        try:
            if not self.processor:
                return False, None, "Document processor not available"
            
            # Create job ID first
            job_id = self.db_manager.create_upload_job(
                user_id,
                len(file_list),
                project_id,
                meeting_id
            )
            
            # Start background processing using existing processor
            import threading
            
            def process_in_background():
                """Background processing function"""
                try:
                    self.processor.process_files_batch_async(
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
            
            return True, job_id, "Background processing started"
            
        except Exception as e:
            logger.error(f"Error starting background processing: {e}")
            # Clean up uploaded files on error
            for file_info in file_list:
                try:
                    if os.path.exists(file_info['path']):
                        os.remove(file_info['path'])
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up {file_info['path']}: {cleanup_error}")
            
            return False, None, f"Error starting file processing: {str(e)}"
    
    def get_upload_job_status(self, job_id: str, user_id: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Get the status of an upload job.
        
        Args:
            job_id: Job ID
            user_id: User ID (for access validation)
            
        Returns:
            Tuple of (success, job_status, message)
        """
        try:
            job_status = self.db_manager.get_job_status(job_id)
            
            if not job_status:
                return False, None, 'Job not found'
            
            # Check if job belongs to current user
            if job_status['user_id'] != user_id:
                return False, None, 'Access denied'
            
            return True, job_status, 'Job status retrieved'
            
        except Exception as e:
            logger.error(f"Job status error: {e}")
            return False, None, f"Error getting job status: {str(e)}"