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
    
    # Document Deletion Endpoints - DISABLED per user request
    @doc_bp.route(f'{base_path}/api/documents/<document_id>', methods=['DELETE'])
    @login_required
    def delete_document(document_id):
        """Document deletion is disabled."""
        return jsonify({
            'success': False,
            'error': 'Document deletion is disabled',
            'document_id': document_id
        }), 403
    
    @doc_bp.route(f'{base_path}/api/documents/batch', methods=['DELETE'])
    @login_required
    def delete_multiple_documents():
        """Batch document deletion is disabled."""
        return jsonify({
            'success': False,
            'error': 'Batch document deletion is disabled'
        }), 403
    
    @doc_bp.route(f'{base_path}/api/documents/deletable')
    @login_required
    def get_deletable_documents():
        """Document deletion functionality is disabled."""
        return jsonify({
            'success': False,
            'error': 'Document deletion functionality is disabled'
        }), 403
    
    @doc_bp.route(f'{base_path}/api/documents/storage/stats')
    @login_required
    def get_storage_statistics():
        """Get storage usage statistics."""
        try:
            success, stats, message = document_service.get_storage_statistics(
                current_user.user_id
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'statistics': stats,
                    'message': message
                })
            else:
                return jsonify({
                    'success': False,
                    'error': message
                }), 500
                
        except Exception as e:
            logger.error(f"Storage statistics error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @doc_bp.route(f'{base_path}/api/documents/index/rebuild', methods=['POST'])
    @login_required
    def rebuild_search_index():
        """Force rebuild of the search index (admin operation)."""
        try:
            # Note: You might want to add admin role checking here
            success = document_service.db_manager.rebuild_vector_index_after_deletion()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Search index rebuilt successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to rebuild search index'
                }), 500
                
        except Exception as e:
            logger.error(f"Index rebuild error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @doc_bp.route(f'{base_path}/api/documents/validate-deletion', methods=['POST'])
    @login_required
    def validate_deletion_safety():
        """Validate safety of document deletion before proceeding."""
        try:
            data = request.get_json()
            if not data or 'document_ids' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing document_ids in request body'
                }), 400
            
            document_ids = data['document_ids']
            if not isinstance(document_ids, list) or not document_ids:
                return jsonify({
                    'success': False,
                    'error': 'document_ids must be a non-empty list'
                }), 400
            
            # Validate deletion safety
            safety_report = document_service.db_manager.validate_deletion_safety(
                document_ids, current_user.user_id
            )
            
            return jsonify({
                'success': True,
                'safety_report': safety_report,
                'recommended_action': 'proceed' if safety_report['safe_to_delete'] else 'review_warnings'
            })
            
        except Exception as e:
            logger.error(f"Deletion validation error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @doc_bp.route(f'{base_path}/api/documents/recent-deletions')
    @login_required
    def get_recent_deletions():
        """Get recent deletion history for the user."""
        try:
            days = request.args.get('days', 7, type=int)
            
            recent_deletions = document_service.db_manager.get_recent_deletions(
                current_user.user_id, days
            )
            
            return jsonify({
                'success': True,
                'deletions': recent_deletions,
                'count': len(recent_deletions),
                'days_covered': days
            })
            
        except Exception as e:
            logger.error(f"Recent deletions error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @doc_bp.route(f'{base_path}/api/documents/cleanup-backups', methods=['POST'])
    @login_required
    def cleanup_old_backups():
        """Clean up expired document backups."""
        try:
            # Note: You might want to add admin role checking here
            cleanup_result = document_service.db_manager.cleanup_old_backups()
            
            if cleanup_result['success']:
                return jsonify(cleanup_result)
            else:
                return jsonify(cleanup_result), 500
                
        except Exception as e:
            logger.error(f"Backup cleanup error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return doc_bp