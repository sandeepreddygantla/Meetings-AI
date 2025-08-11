/**
 * Configuration module for Meetings AI frontend.
 * Provides dynamic configuration based on server settings.
 */

// Global configuration object
window.APP_CONFIG = {
    basePath: '/meetingsai',  // Default base path
    apiBasePath: '/meetingsai/api',
    staticPath: '/meetingsai/static',
    
    // UI Configuration
    maxFileSize: 100 * 1024 * 1024, // 100MB
    allowedFileTypes: ['.txt', '.docx', '.pdf'],
    maxFilesPerUpload: 10,
    
    // Chat Configuration
    maxQueryLength: 1000,
    followUpQuestionsCount: 4,
    
    // Mention Configuration
    mentionTriggers: ['@', '#'],
    mentionTypes: {
        '@': ['project', 'meeting', 'date', 'file'],
        '#': ['folder', 'folder_files']
    },
    
    // Date formats for mentions
    dateFormats: [
        'today',
        'yesterday', 
        'this week',
        'last week',
        'this month',
        'YYYY-MM-DD',
        'MM/DD/YYYY'
    ],
    
    // API endpoints
    endpoints: {
        auth: {
            login: '/login',
            register: '/register',
            logout: '/logout',
            status: '/api/auth/status'
        },
        chat: {
            send: '/api/chat',
            stats: '/api/stats',
            filters: '/api/filters'
        },
        documents: {
            list: '/api/documents',
            upload: '/api/upload',
            jobStatus: '/api/job_status',
            stats: '/api/upload/stats'
        },
        projects: {
            list: '/api/projects',
            create: '/api/projects',
            meetings: '/api/meetings'
        }
    }
};

/**
 * Get full API URL for an endpoint
 * @param {string} category - Category (auth, chat, documents, projects)
 * @param {string} endpoint - Endpoint name
 * @returns {string} Full API URL
 */
function getApiUrl(category, endpoint) {
    const baseUrl = window.APP_CONFIG.basePath;
    const endpointPath = window.APP_CONFIG.endpoints[category]?.[endpoint];
    
    if (!endpointPath) {
        console.error(`Unknown endpoint: ${category}.${endpoint}`);
        return null;
    }
    
    return baseUrl + endpointPath;
}

/**
 * Get static asset URL
 * @param {string} path - Path to asset
 * @returns {string} Full static URL
 */
function getStaticUrl(path) {
    return window.APP_CONFIG.staticPath + '/' + path.replace(/^\/+/, '');
}

/**
 * Update configuration from server
 * @param {Object} serverConfig - Configuration from server
 */
function updateConfig(serverConfig) {
    if (serverConfig.basePath) {
        window.APP_CONFIG.basePath = serverConfig.basePath;
        window.APP_CONFIG.apiBasePath = serverConfig.basePath + '/api';
        window.APP_CONFIG.staticPath = serverConfig.basePath + '/static';
    }
    
    // Update other configuration as needed
    Object.assign(window.APP_CONFIG, serverConfig);
    
    // Configuration updated
}

/**
 * Initialize configuration
 */
function initializeConfig() {
    // Try to get configuration from server-side rendered data
    const serverConfig = window.SERVER_CONFIG;
    if (serverConfig) {
        updateConfig(serverConfig);
    }
    
    // Configuration initialized
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeConfig);
} else {
    initializeConfig();
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        APP_CONFIG: window.APP_CONFIG,
        getApiUrl,
        getStaticUrl,
        updateConfig
    };
}