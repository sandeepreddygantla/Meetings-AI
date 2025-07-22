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
            
            # ===== DEBUG LOGGING: QUERY ENTRY POINT =====
            logger.info("=" * 80)
            logger.info("[QUERY] NEW CHAT QUERY RECEIVED")
            logger.info("=" * 80)
            logger.info(f"[USER] User ID: {current_user.user_id}")
            logger.info(f"[QUESTION] Question: '{message}'")
            logger.info(f"[FILTERS] Filters Applied:")
            logger.info(f"   - Document IDs: {document_ids}")
            logger.info(f"   - Project ID: {project_id}")
            logger.info(f"   - Project IDs: {project_ids}")
            logger.info(f"   - Meeting IDs: {meeting_ids}")
            logger.info(f"   - Date Filters: {date_filters}")
            logger.info(f"   - Folder Path: {folder_path}")
            logger.info("[START] Starting query processing pipeline...")
            
            if not message:
                logger.error("[ERROR] QUERY REJECTED: No message provided")
                return jsonify({'success': False, 'error': 'No message provided'}), 400
            
            # Validate filters
            logger.info("[STEP1] Validating chat filters...")
            is_valid, validation_error = chat_service.validate_chat_filters(
                current_user.user_id, document_ids, project_id, meeting_ids
            )
            
            if not is_valid:
                logger.error(f"[ERROR] FILTER VALIDATION FAILED: {validation_error}")
                return jsonify({'success': False, 'error': validation_error}), 400
            
            logger.info("[OK] Step 1: Filter validation passed")
            
            # Process chat query
            logger.info("[STEP2] Starting chat query processing...")
            logger.info("   -> Routing to ChatService.process_chat_query()")
            
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
            
            # ===== DEBUG LOGGING: FINAL RESPONSE =====
            logger.info("=" * 80)
            logger.info("[COMPLETE] CHAT QUERY PROCESSING COMPLETED")
            logger.info("=" * 80)
            logger.info(f"[RESPONSE] Final Response Length: {len(response)} characters")
            logger.info(f"[FOLLOWUP] Follow-up Questions: {len(follow_up_questions)} generated")
            logger.info(f"[TIME] Processing Timestamp: {timestamp}")
            
            # Check for "no relevant information" responses
            if "no relevant information" in response.lower() or "couldn't find" in response.lower():
                logger.error("[WARNING] DETECTED: 'No relevant information' response - this indicates search issues!")
            else:
                logger.info("[SUCCESS] Response contains relevant information")
            
            logger.info("[SEND] Sending response to client...")
            logger.info("=" * 80)
            
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