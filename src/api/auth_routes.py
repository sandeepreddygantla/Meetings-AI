"""
Authentication API routes for Meetings AI application.
"""
import logging
from flask import Blueprint, request, jsonify, render_template, session
from flask_login import login_required, current_user

from src.services.auth_service import AuthService

logger = logging.getLogger(__name__)


def create_auth_blueprint(base_path: str, auth_service: AuthService) -> Blueprint:
    """
    Create authentication blueprint with routes.
    
    Args:
        base_path: Base path for routes (e.g., '/meetingsai')
        auth_service: Authentication service instance
        
    Returns:
        Flask Blueprint
    """
    auth_bp = Blueprint('auth', __name__)
    
    @auth_bp.route(f'{base_path}/register', methods=['GET', 'POST'])
    def register():
        """User registration endpoint."""
        if request.method == 'GET':
            return render_template('register.html')
        
        try:
            # Get request data
            data = request.get_json()
            if not data:
                logger.error("No JSON data received in registration request")
                return jsonify({'success': False, 'error': 'Invalid request data'}), 400
            
            username = data.get('username', '').strip()
            email = data.get('email', '').strip()
            full_name = data.get('full_name', '').strip()
            password = data.get('password', '')
            confirm_password = data.get('confirm_password', '')
            
            # Basic validation
            if password != confirm_password:
                return jsonify({'success': False, 'error': 'Passwords do not match'}), 400
            
            # Register user
            success, message, user_id = auth_service.register_user(username, email, full_name, password)
            
            if success:
                return jsonify({
                    'success': True, 
                    'message': message,
                    'user_id': user_id
                })
            else:
                return jsonify({'success': False, 'error': message}), 400
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return jsonify({'success': False, 'error': f'Registration failed: {str(e)}'}), 500
    
    @auth_bp.route(f'{base_path}/login', methods=['GET', 'POST'])
    def login():
        """User login endpoint."""
        if request.method == 'GET':
            return render_template('login.html')
        
        try:
            data = request.get_json()
            username = data.get('username', '').strip()
            password = data.get('password', '')
            
            if not username or not password:
                return jsonify({'success': False, 'error': 'Username and password are required'}), 400
            
            # Authenticate user
            success, message, user = auth_service.authenticate_user(username, password)
            
            if not success:
                return jsonify({'success': False, 'error': message}), 401
            
            # Create session
            session_success = auth_service.login_user_session(user, remember=True)
            
            if not session_success:
                return jsonify({'success': False, 'error': 'Failed to create session'}), 500
            
            return jsonify({
                'success': True, 
                'message': message,
                'user': {
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name
                }
            })
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'success': False, 'error': f'Login failed: {str(e)}'}), 500
    
    @auth_bp.route(f'{base_path}/logout', methods=['POST'])
    @login_required
    def logout():
        """User logout endpoint."""
        try:
            success, message = auth_service.logout_user_session()
            
            if success:
                return jsonify({'success': True, 'message': message})
            else:
                return jsonify({'success': False, 'error': message}), 500
                
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return jsonify({'success': False, 'error': f'Logout failed: {str(e)}'}), 500
    
    @auth_bp.route(f'{base_path}/api/auth/status')
    def auth_status():
        """Check authentication status."""
        try:
            user_data = auth_service.get_current_user_data()
            
            if user_data:
                session.permanent = True
                return jsonify({
                    'authenticated': True,
                    'user': user_data
                })
            else:
                return jsonify({'authenticated': False, 'reason': 'not_logged_in'}), 401
                
        except Exception as e:
            logger.error(f"Auth status check error: {e}")
            return jsonify({'authenticated': False, 'reason': 'validation_error'}), 401
    
    return auth_bp