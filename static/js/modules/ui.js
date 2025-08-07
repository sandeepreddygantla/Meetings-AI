/**
 * UI Module - Core user interface functionality
 * Handles basic UI interactions and components
 */

// UI state management
let sidebarOpen = false;
let notificationQueue = [];

function initializeUI() {
    console.log('Initializing UI module');
    
    // Setup sidebar toggle
    setupSidebarToggle();
    
    // Setup notification system
    setupNotificationSystem();
    
    // Setup responsive handlers
    setupResponsiveHandlers();
    
    // Load initial welcome screen
    loadWelcomeScreen();
}

function setupSidebarToggle() {
    const toggleBtn = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    const backdrop = document.getElementById('mobile-backdrop');
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleSidebar);
    }
    
    if (backdrop) {
        backdrop.addEventListener('click', closeMobileSidebar);
    }
    
    // Handle ESC key for sidebar
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && sidebarOpen) {
            closeMobileSidebar();
        }
    });
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const backdrop = document.getElementById('mobile-backdrop');
    const toggleIcon = document.getElementById('sidebar-toggle-icon');
    
    if (sidebar && backdrop && toggleIcon) {
        sidebarOpen = !sidebarOpen;
        
        if (sidebarOpen) {
            sidebar.classList.add('active');
            backdrop.classList.add('active');
            toggleIcon.textContent = '✕';
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
        } else {
            sidebar.classList.remove('active');
            backdrop.classList.remove('active');
            toggleIcon.textContent = '☰';
            document.body.style.overflow = '';
        }
    }
}

function closeMobileSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const backdrop = document.getElementById('mobile-backdrop');
    const toggleIcon = document.getElementById('sidebar-toggle-icon');
    
    if (sidebar && backdrop && toggleIcon) {
        sidebarOpen = false;
        sidebar.classList.remove('active');
        backdrop.classList.remove('active');
        toggleIcon.textContent = '☰';
        document.body.style.overflow = '';
    }
}

function setupNotificationSystem() {
    // Create notification container if it doesn't exist
    if (!document.getElementById('notification-container')) {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
}

function showNotification(message, type = 'info', duration = 5000) {
    const container = document.getElementById('notification-container');
    if (!container) return;
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span>${message}</span>
            <button class="notification-close" onclick="closeNotification(this)">&times;</button>
        </div>
    `;
    
    container.appendChild(notification);
    
    // Animate in
    requestAnimationFrame(() => {
        notification.classList.add('show');
    });
    
    // Auto-remove after duration (if not persistent)
    if (duration > 0) {
        setTimeout(() => {
            closeNotification(notification);
        }, duration);
    }
    
    return notification;
}

function closeNotification(element) {
    const notification = element.closest ? element.closest('.notification') : element;
    if (notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}

function setupResponsiveHandlers() {
    // Handle window resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            handleWindowResize();
        }, 100);
    });
}

function handleWindowResize() {
    // Close sidebar on desktop view
    if (window.innerWidth >= 1024 && sidebarOpen) {
        closeMobileSidebar();
    }
}

function loadWelcomeScreen() {
    const messagesArea = document.getElementById('messages-area');
    if (!messagesArea) return;
    
    messagesArea.innerHTML = `
        <div class="welcome-screen">
            <div class="welcome-icon">💬</div>
            <h1>Welcome to Document Fulfillment</h1>
            <p>Upload your meeting documents and start asking questions to get instant insights.</p>
            <div class="welcome-actions">
                <button class="btn btn-primary" onclick="showUploadModal()">
                    📁 Upload Documents
                </button>
                <button class="btn btn-secondary" onclick="showHelp()">
                    ❓ Learn More
                </button>
            </div>
            <div class="quick-tips">
                <div class="tip">
                    <strong>Tip:</strong> You can mention specific projects with @project:name
                </div>
                <div class="tip">
                    <strong>Tip:</strong> Reference dates with @date:today or @date:yesterday
                </div>
            </div>
        </div>
    `;
}

// Export functions to global scope for compatibility
window.initializeUI = initializeUI;
window.toggleSidebar = toggleSidebar;
window.closeMobileSidebar = closeMobileSidebar;
window.showNotification = showNotification;
window.closeNotification = closeNotification;