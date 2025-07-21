"""
Document API routes for Meetings AI application.
"""
import logging
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from src.services.document_service import DocumentService
from src.services.upload_service import UploadService

logger = logging.getLogger(__name__)


def create_document_blueprint(base_path: str, document_service: DocumentService, upload_service: UploadService) -> Blueprint:
    """
    Create document blueprint with routes.
    
    Args:
        base_path: Base path for routes (e.g., '/meetingsai')
        document_service: Document service instance
        upload_service: Upload service instance
        
    Returns:
        Flask Blueprint
    """
    doc_bp = Blueprint('documents', __name__)
    
    @doc_bp.route(f'{base_path}/api/upload', methods=['POST'])
    @login_required
    def upload_files():
        """Handle file uploads."""
        try:
            if 'files' not in request.files:
                return jsonify({'success': False, 'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            if not files or all(f.filename == '' for f in files):
                return jsonify({'success': False, 'error': 'No files selected'}), 400
            
            # Get form data
            project_id = request.form.get('project_id', '').strip()
            meeting_id = request.form.get('meeting_id', '').strip()
            
            # Handle upload
            success, response_data, message = upload_service.handle_file_upload(
                files=files,
                user_id=current_user.user_id,
                username=current_user.username,
                project_id=project_id if project_id else None,
                meeting_id=meeting_id if meeting_id else None
            )
            
            if success:
                return jsonify(response_data)
            else:
                return jsonify(response_data), 400
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @doc_bp.route(f'{base_path}/api/job_status/<job_id>')
    @login_required
    def get_job_status(job_id):
        """Get upload job status."""
        try:
            success, progress_data, message = upload_service.get_upload_progress(job_id, current_user.user_id)
            
            if success:
                return jsonify({
                    'success': True,
                    **progress_data
                })
            else:
                return jsonify({'success': False, 'error': message}), 404 if 'not found' in message.lower() else 403
                
        except Exception as e:
            logger.error(f"Job status error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @doc_bp.route(f'{base_path}/api/documents')
    @login_required
    def get_documents():
        """Get all documents for the current user."""
        try:
            success, documents, message = document_service.get_user_documents(current_user.user_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'documents': documents,
                    'count': len(documents)
                })
            else:
                return jsonify({'success': False, 'error': message}), 500
                
        except Exception as e:
            logger.error(f"Documents error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @doc_bp.route(f'{base_path}/api/projects')
    @login_required
    def get_projects():
        """Get all projects for the current user."""
        try:
            success, projects, message = document_service.get_user_projects(current_user.user_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'projects': projects,
                    'count': len(projects)
                })
            else:
                return jsonify({'success': False, 'error': message}), 500
                
        except Exception as e:
            logger.error(f"Projects error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @doc_bp.route(f'{base_path}/api/projects', methods=['POST'])
    @login_required
    def create_project():
        """Create a new project."""
        try:
            data = request.get_json()
            project_name = data.get('project_name', '').strip()
            description = data.get('description', '').strip()
            
            success, message, project_id = document_service.create_project(
                current_user.user_id, project_name, description
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': message,
                    'project_id': project_id
                })
            else:
                return jsonify({'success': False, 'error': message}), 400
                
        except Exception as e:
            logger.error(f"Create project error: {e}")
            return jsonify({'success': False, 'error': 'Failed to create project'}), 500
    
    @doc_bp.route(f'{base_path}/api/meetings')
    @login_required
    def get_meetings():
        """Get all meetings for the current user."""
        try:
            project_id = request.args.get('project_id')
            
            success, meetings, message = document_service.get_user_meetings(
                current_user.user_id, project_id
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'meetings': meetings,
                    'count': len(meetings)
                })
            else:
                return jsonify({'success': False, 'error': message}), 500
                
        except Exception as e:
            logger.error(f"Meetings error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @doc_bp.route(f'{base_path}/api/upload/stats')
    @login_required
    def get_upload_stats():
        """Get upload statistics for the current user."""
        try:
            stats = upload_service.get_upload_statistics(current_user.user_id)
            
            if "error" in stats:
                return jsonify({'success': False, 'error': stats['error']}), 500
            
            return jsonify({'success': True, 'stats': stats})
            
        except Exception as e:
            logger.error(f"Upload stats error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return doc_bp