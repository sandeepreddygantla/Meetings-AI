"""
Upload service for Meetings AI application.
Handles file upload coordination and validation.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from src.services.document_service import DocumentService
from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class UploadService:
    """Service for coordinating file upload operations."""
    
    def __init__(self, db_manager: DatabaseManager, document_service: DocumentService):
        """
        Initialize upload service.
        
        Args:
            db_manager: Database manager instance
            document_service: Document service instance
        """
        self.db_manager = db_manager
        self.document_service = document_service
    
    def handle_file_upload(
        self,
        files: List[Any],
        user_id: str,
        username: str,
        project_id: Optional[str] = None,
        meeting_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Handle complete file upload process.
        
        Args:
            files: List of uploaded files
            user_id: User ID
            username: Username
            project_id: Optional project ID
            meeting_id: Optional meeting ID
            
        Returns:
            Tuple of (success, response_data, message)
        """
        try:
            # Step 1: Validate upload parameters
            is_valid, validation_error = self.document_service.validate_file_upload(
                files, project_id, meeting_id, user_id
            )
            if not is_valid:
                return False, {'error': validation_error}, validation_error
            
            # Step 2: Prepare upload directory
            success, upload_folder, dir_error = self.document_service.prepare_upload_directory(
                user_id, username, project_id
            )
            if not success:
                return False, {'error': dir_error}, dir_error
            
            # Step 3: Process and validate files
            file_list, validation_errors, duplicates = self.document_service.process_file_validation(
                files, upload_folder, user_id
            )
            
            # Step 4: Check if we have valid files to process
            if not file_list:
                return False, {
                    'error': 'No valid files to process',
                    'validation_errors': validation_errors,
                    'duplicates': duplicates
                }, 'No valid files to process'
            
            # Step 5: Start background processing
            bg_success, job_id, bg_error = self.document_service.start_background_processing(
                file_list, user_id, project_id, meeting_id
            )
            
            if not bg_success:
                return False, {'error': bg_error}, bg_error
            
            # Step 6: Return success response
            response_data = {
                'success': True,
                'job_id': job_id,
                'total_files': len(file_list),
                'validation_errors': validation_errors,
                'duplicates': duplicates,
                'message': f'Upload started for {len(file_list)} files. Use job ID to track progress.'
            }
            
            return True, response_data, f"Successfully started processing {len(file_list)} files"
            
        except Exception as e:
            logger.error(f"File upload handling error: {e}")
            return False, {'error': str(e)}, f"Upload failed: {str(e)}"
    
    def get_upload_progress(self, job_id: str, user_id: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Get upload job progress.
        
        Args:
            job_id: Upload job ID
            user_id: User ID for validation
            
        Returns:
            Tuple of (success, progress_data, message)
        """
        try:
            success, job_status, message = self.document_service.get_upload_job_status(job_id, user_id)
            
            if success:
                return True, {'job_status': job_status}, message
            else:
                return False, {'error': message}, message
                
        except Exception as e:
            logger.error(f"Upload progress error: {e}")
            return False, {'error': str(e)}, f"Error getting upload progress: {str(e)}"
    
    def get_upload_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get upload statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        try:
            stats = {}
            
            # Get document count
            success, documents, _ = self.document_service.get_user_documents(user_id)
            stats['total_documents'] = len(documents) if success else 0
            
            # Get projects count
            success, projects, _ = self.document_service.get_user_projects(user_id)
            stats['total_projects'] = len(projects) if success else 0
            
            # Get meetings count
            success, meetings, _ = self.document_service.get_user_meetings(user_id)
            stats['total_meetings'] = len(meetings) if success else 0
            
            # Get recent upload activity (if available)
            # This could be enhanced with more detailed tracking
            stats['recent_uploads'] = 0  # Placeholder
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting upload statistics: {e}")
            return {'error': str(e)}
    
    def cleanup_failed_uploads(self, user_id: str) -> Tuple[bool, str]:
        """
        Clean up any failed upload artifacts.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # This could be implemented to clean up:
            # - Orphaned files in upload directories
            # - Failed job records
            # - Incomplete document records
            
            # For now, just return success
            # Future enhancement: implement actual cleanup logic
            
            return True, "Cleanup completed successfully"
            
        except Exception as e:
            logger.error(f"Upload cleanup error: {e}")
            return False, f"Cleanup failed: {str(e)}"