"""
Chat API routes for Meetings AI application.
"""
import logging
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from src.services.chat_service import ChatService

logger = logging.getLogger(__name__)


def create_chat_blueprint(base_path: str, chat_service: ChatService) -> Blueprint:
    """
    Create chat blueprint with routes.
    
    Args:
        base_path: Base path for routes (e.g., '/meetingsai')
        chat_service: Chat service instance
        
    Returns:
        Flask Blueprint
    """
    chat_bp = Blueprint('chat', __name__)
    
    @chat_bp.route(f'{base_path}/api/chat', methods=['POST'])
    @login_required
    def chat():
        """Handle chat messages."""
        try:
            data = request.get_json()
            message = data.get('message', '').strip()
            document_ids = data.get('document_ids', None)
            project_id = data.get('project_id', None)
            project_ids = data.get('project_ids', None)
            meeting_ids = data.get('meeting_ids', None)
            date_filters = data.get('date_filters', None)
            folder_path = data.get('folder_path', None)
            
            if not message:
                return jsonify({'success': False, 'error': 'No message provided'}), 400
            
            # Validate filters
            is_valid, validation_error = chat_service.validate_chat_filters(
                current_user.user_id, document_ids, project_id, meeting_ids
            )
            
            if not is_valid:
                logger.error(f"Filter validation failed: {validation_error}")
                return jsonify({'success': False, 'error': validation_error}), 400
            
            # Process chat query
            response, follow_up_questions, timestamp = chat_service.process_chat_query(
                message=message,
                user_id=current_user.user_id,
                document_ids=document_ids,
                project_id=project_id,
                project_ids=project_ids,
                meeting_ids=meeting_ids,
                date_filters=date_filters,
                folder_path=folder_path
            )
            
            # Check for "no relevant information" responses
            if "no relevant information" in response.lower() or "couldn't find" in response.lower():
                logger.warning("Query returned no relevant information")
            
            return jsonify({
                'success': True,
                'response': response,
                'follow_up_questions': follow_up_questions,
                'timestamp': timestamp
            })
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @chat_bp.route(f'{base_path}/api/stats')
    @login_required
    def get_stats():
        """Get chat statistics."""
        try:
            stats = chat_service.get_chat_statistics(current_user.user_id)
            
            if "error" in stats:
                return jsonify({'success': False, 'error': stats['error']}), 500
            
            return jsonify({'success': True, 'stats': stats})
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @chat_bp.route(f'{base_path}/api/filters')
    @login_required
    def get_filters():
        """Get available filters for chat queries."""
        try:
            filters = chat_service.get_available_filters(current_user.user_id)
            
            if "error" in filters:
                return jsonify({'success': False, 'error': filters['error']}), 500
            
            return jsonify({'success': True, 'filters': filters})
            
        except Exception as e:
            logger.error(f"Filters error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return chat_bp