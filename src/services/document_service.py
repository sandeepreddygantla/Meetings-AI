"""
Document service for Meetings AI application.
Handles document management, processing, and metadata operations.
"""
import os
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from werkzeug.utils import secure_filename

from src.database.manager import DatabaseManager
from src.models.document import MeetingDocument, UploadJob

logger = logging.getLogger(__name__)

# Simple tracking for active operations
_active_uploads = set()
_upload_lock = threading.Lock()


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
    
    def _get_worker_count(self):
        """Get optimal worker count based on current load"""
        with _upload_lock:
            active_count = len(_active_uploads)
            
        if active_count == 0:
            return 4  # Full speed for single user
        elif active_count == 1:
            return 3  # Balanced for two users
        else:
            return 2  # Conservative for multiple users
    
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
    
    def create_project(self, user_id: str, project_name: str, description: str = "") -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Create a new project for a user.
        
        Args:
            user_id: User ID
            project_name: Name of the project
            description: Project description
            
        Returns:
            Tuple of (success, message, project_id, error_code)
        """
        try:
            if not project_name or not project_name.strip():
                return False, 'Project name is required', None, 'EMPTY_NAME'
            
            project_id = self.db_manager.create_project(user_id, project_name.strip(), description.strip())
            logger.info(f"New project created: {project_name} ({project_id}) for user {user_id}")
            
            return True, 'Project created successfully', project_id, None
            
        except ValueError as e:
            error_message = str(e)
            
            if "DUPLICATE_PROJECT_NAME" in error_message:
                return False, f'A project named "{project_name}" already exists. Please choose a different name.', None, 'DUPLICATE_NAME'
            elif "DATABASE_ERROR" in error_message:
                logger.error(f"Database error creating project: {error_message}")
                return False, 'Database error occurred. Please try again.', None, 'DATABASE_ERROR'
            elif "GENERAL_ERROR" in error_message:
                logger.error(f"General error creating project: {error_message}")
                return False, 'An unexpected error occurred. Please try again.', None, 'GENERAL_ERROR'
            else:
                return False, error_message, None, 'UNKNOWN_ERROR'
                
        except Exception as e:
            logger.error(f"Create project error: {e}")
            return False, 'Failed to create project', None, 'EXCEPTION_ERROR'
    
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
                        duplicate_type = duplicate_info.get('duplicate_type', 'active')
                        
                        if duplicate_type == 'active':
                            # Regular duplicate - block upload
                            duplicates.append({
                                'filename': filename,
                                'original_filename': duplicate_info['original_filename'],
                                'created_at': duplicate_info['created_at'],
                                'action': 'blocked'
                            })
                            os.remove(file_path)  # Remove the duplicate file
                            continue
                        
                        elif duplicate_type == 'soft_deleted_restorable':
                            # Smart restore - automatically restore the soft-deleted document
                            logger.info(f"Restoring soft-deleted document {duplicate_info['document_id']} for re-uploaded file {filename}")
                            
                            restore_success = self.db_manager.undelete_document(
                                duplicate_info['document_id'], user_id
                            )
                            
                            if restore_success['success']:
                                duplicates.append({
                                    'filename': filename,
                                    'original_filename': duplicate_info['original_filename'], 
                                    'created_at': duplicate_info['created_at'],
                                    'deleted_at': duplicate_info.get('deleted_at'),
                                    'action': 'restored',
                                    'document_id': duplicate_info['document_id']
                                })
                                os.remove(file_path)  # Remove the new file since we restored the old one
                                continue
                            else:
                                # Restore failed - treat as regular duplicate
                                logger.warning(f"Failed to restore document {duplicate_info['document_id']}, treating as regular duplicate")
                                duplicates.append({
                                    'filename': filename,
                                    'original_filename': duplicate_info['original_filename'],
                                    'created_at': duplicate_info['created_at'],
                                    'action': 'blocked_restore_failed'
                                })
                                os.remove(file_path)
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
                    # Track this upload and get optimal worker count
                    with _upload_lock:
                        _active_uploads.add(user_id)
                    
                    worker_count = self._get_worker_count()
                    logger.info(f"Starting upload with {worker_count} workers for user {user_id}")
                    
                    self.processor.process_files_batch_async(
                        file_list,
                        user_id,
                        project_id,
                        meeting_id,
                        max_workers=worker_count,
                        job_id=job_id  # Pass existing job_id
                    )
                except Exception as e:
                    logger.error(f"Background processing error: {e}")
                finally:
                    # Clean up tracking
                    with _upload_lock:
                        _active_uploads.discard(user_id)
            
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
    
    # Document Deletion Operations
    def delete_document(self, document_id: str, user_id: str) -> Tuple[bool, str]:
        """
        Soft delete a single document (safer than hard deletion).
        
        Args:
            document_id: Document ID to delete
            user_id: User ID for verification
            
        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info(f"Soft-deleting document {document_id} for user {user_id}")
            
            # Perform soft deletion (marks as deleted, keeps FAISS vectors intact)
            result = self.db_manager.soft_delete_document(document_id, user_id, user_id)
            
            if result['success']:
                message = f"Document {document_id} moved to trash successfully"
                
                # Note about soft deletion
                if result.get('faiss_intact'):
                    message += " (document hidden from search, can be restored)"
                
                logger.info(message)
                return True, message
            else:
                error_msg = f"Failed to delete document {document_id}"
                if result.get('error'):
                    error_msg += f": {result['error']}"
                
                logger.error(error_msg)
                return False, error_msg
            
        except Exception as e:
            error_msg = f"Error deleting document {document_id}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def delete_multiple_documents(self, document_ids: List[str], user_id: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Soft delete multiple documents with detailed results.
        
        Args:
            document_ids: List of document IDs to delete
            user_id: User ID for verification
            
        Returns:
            Tuple of (success, detailed_results, message)
        """
        try:
            if not document_ids:
                return False, {}, "No documents provided for deletion"
            
            logger.info(f"Soft-deleting {len(document_ids)} documents for user {user_id}")
            
            # Perform batch soft deletion
            results = {
                'success': True,
                'total_requested': len(document_ids),
                'successful_deletions': [],
                'failed_deletions': [],
                'detailed_results': {}
            }
            
            for doc_id in document_ids:
                try:
                    result = self.db_manager.soft_delete_document(doc_id, user_id, user_id)
                    results['detailed_results'][doc_id] = result
                    
                    if result['success']:
                        results['successful_deletions'].append(doc_id)
                    else:
                        results['failed_deletions'].append(doc_id)
                        
                except Exception as e:
                    logger.error(f"Error soft-deleting document {doc_id}: {e}")
                    results['failed_deletions'].append(doc_id)
                    results['detailed_results'][doc_id] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Calculate summary
            results['summary'] = {
                'total_requested': results['total_requested'],
                'successful': len(results['successful_deletions']),
                'failed': len(results['failed_deletions']),
                'success_rate': len(results['successful_deletions']) / max(1, results['total_requested'])
            }
            
            summary = results.get('summary', {})
            successful_count = summary.get('successful', 0)
            failed_count = summary.get('failed', 0)
            total_count = summary.get('total_requested', len(document_ids))
            
            if successful_count > 0:
                if failed_count == 0:
                    message = f"Successfully moved all {successful_count} documents to trash"
                    success = True
                else:
                    message = f"Moved {successful_count} of {total_count} documents to trash ({failed_count} failed)"
                    success = True  # Partial success
            else:
                message = f"Failed to delete any of the {total_count} documents"
                success = False
            
            logger.info(f"Batch deletion completed: {message}")
            return success, results, message
            
        except Exception as e:
            error_msg = f"Error in batch document deletion: {str(e)}"
            logger.error(error_msg)
            return False, {}, error_msg
    
    def get_deletable_documents(self, user_id: str, filters: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[Dict], str]:
        """
        Get list of documents that can be deleted by the user.
        
        Args:
            user_id: User ID
            filters: Optional filters (project_id, date_range, min_size, etc.)
            
        Returns:
            Tuple of (success, documents_list, message)
        """
        try:
            # Get deletable documents with filters
            documents = self.db_manager.get_deletable_documents(user_id, filters)
            
            # Enhance with deletion-relevant metadata
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = dict(doc)  # Copy document data
                
                # Add deletion-specific metadata
                enhanced_doc['can_delete'] = True  # All returned docs are deletable
                enhanced_doc['deletion_impact'] = self._assess_deletion_impact(doc)
                
                enhanced_docs.append(enhanced_doc)
            
            message = f"Found {len(enhanced_docs)} deletable documents"
            if filters:
                message += " (with filters applied)"
            
            return True, enhanced_docs, message
            
        except Exception as e:
            error_msg = f"Error getting deletable documents: {str(e)}"
            logger.error(error_msg)
            return False, [], error_msg
    
    def _assess_deletion_impact(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the impact of deleting a document.
        
        Args:
            document: Document metadata
            
        Returns:
            Impact assessment dictionary
        """
        try:
            impact = {
                'size_saved': document.get('file_size', 0),
                'chunks_removed': document.get('chunk_count', 0),
                'warnings': []
            }
            
            # Add warnings for potentially important documents
            if document.get('chunk_count', 0) > 50:
                impact['warnings'].append('Large document with many sections')
            
            file_size = document.get('file_size', 0)
            if file_size > 10 * 1024 * 1024:  # > 10MB
                impact['warnings'].append('Large file (>10MB)')
            
            # Check if document is recent
            created_at = document.get('created_at')
            if created_at:
                try:
                    created_date = datetime.fromisoformat(created_at)
                    days_old = (datetime.now() - created_date).days
                    if days_old < 7:
                        impact['warnings'].append('Recently uploaded (less than 7 days ago)')
                except Exception:
                    pass
            
            return impact
            
        except Exception as e:
            logger.error(f"Error assessing deletion impact: {e}")
            return {'size_saved': 0, 'chunks_removed': 0, 'warnings': ['Unable to assess impact']}
    
    def get_storage_statistics(self, user_id: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Get storage usage statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success, statistics, message)
        """
        try:
            # Get comprehensive deletion statistics
            deletion_stats = self.db_manager.get_deletion_statistics()
            
            # Get user-specific document count and sizes
            user_documents = self.db_manager.get_all_documents(user_id)
            
            user_stats = {
                'document_count': len(user_documents),
                'total_size': sum(doc.get('file_size', 0) for doc in user_documents),
                'total_chunks': sum(doc.get('chunk_count', 0) for doc in user_documents)
            }
            
            # Combine with system-wide stats
            combined_stats = {
                'user_stats': user_stats,
                'system_stats': deletion_stats,
                'recommendations': self._generate_cleanup_recommendations(user_documents)
            }
            
            return True, combined_stats, "Storage statistics retrieved"
            
        except Exception as e:
            error_msg = f"Error getting storage statistics: {str(e)}"
            logger.error(error_msg)
            return False, {}, error_msg
    
    def _generate_cleanup_recommendations(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Generate cleanup recommendations based on user's documents.
        
        Args:
            documents: List of user documents
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            if not documents:
                return recommendations
            
            # Large files recommendation
            large_files = [doc for doc in documents if doc.get('file_size', 0) > 5 * 1024 * 1024]
            if large_files:
                recommendations.append(f"Consider reviewing {len(large_files)} large files (>5MB) for potential cleanup")
            
            # Old files recommendation
            old_files = []
            for doc in documents:
                created_at = doc.get('created_at')
                if created_at:
                    try:
                        created_date = datetime.fromisoformat(created_at)
                        days_old = (datetime.now() - created_date).days
                        if days_old > 90:
                            old_files.append(doc)
                    except Exception:
                        pass
            
            if old_files:
                recommendations.append(f"Consider archiving or deleting {len(old_files)} files older than 90 days")
            
            # Project-based recommendations
            project_counts = {}
            for doc in documents:
                project_id = doc.get('project_id', 'no_project')
                project_counts[project_id] = project_counts.get(project_id, 0) + 1
            
            if len(project_counts) > 5:
                recommendations.append(f"You have documents across {len(project_counts)} projects - consider consolidating")
            
            if not recommendations:
                recommendations.append("Your document storage looks well organized!")
            
        except Exception as e:
            logger.error(f"Error generating cleanup recommendations: {e}")
            recommendations = ["Unable to generate cleanup recommendations"]
        
        return recommendations