/**
 * API Module - Handles all API communications
 * Centralized API request handling with error management
 */

class APIClient {
    constructor() {
        this.baseURL = '/meetingsai/api';
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        };
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            method: options.method || 'GET',
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };

        // Add body for POST/PUT requests
        if (config.method !== 'GET' && options.data) {
            if (options.data instanceof FormData) {
                // Remove Content-Type for FormData to let browser set it
                delete config.headers['Content-Type'];
                config.body = options.data;
            } else {
                config.body = JSON.stringify(options.data);
            }
        }

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Handle different content types
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error(`API request failed: ${url}`, error);
            throw error;
        }
    }

    // Convenience methods
    async get(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: 'GET' });
    }

    async post(endpoint, data, options = {}) {
        return this.request(endpoint, { ...options, method: 'POST', data });
    }

    async put(endpoint, data, options = {}) {
        return this.request(endpoint, { ...options, method: 'PUT', data });
    }

    async delete(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: 'DELETE' });
    }
}

// Create global API client instance
const API = new APIClient();

// Specific API methods
const APIService = {
    // Authentication
    async login(credentials) {
        return API.post('/auth/login', credentials);
    },

    async logout() {
        return API.post('/auth/logout');
    },

    // Documents
    async getDocuments(filters = {}) {
        const params = new URLSearchParams(filters).toString();
        return API.get(`/documents${params ? '?' + params : ''}`);
    },

    async uploadDocuments(formData) {
        return API.post('/upload', formData);
    },

    async deleteDocument(documentId) {
        return API.delete(`/documents/${documentId}`);
    },

    // Chat
    async sendMessage(message, context = {}) {
        return API.post('/chat', { message, ...context });
    },

    // Projects
    async getProjects() {
        return API.get('/projects');
    },

    async createProject(projectData) {
        return API.post('/projects', projectData);
    },

    // System
    async refresh() {
        return API.post('/refresh');
    },

    async getStats() {
        return API.get('/stats');
    }
};

// Error handling wrapper
function withErrorHandling(apiMethod) {
    return async function(...args) {
        try {
            return await apiMethod.apply(this, args);
        } catch (error) {
            console.error('API Error:', error);
            
            // Show user-friendly error message
            if (typeof showNotification === 'function') {
                let errorMessage = 'An error occurred. Please try again.';
                
                if (error.message.includes('HTTP 401')) {
                    errorMessage = 'Session expired. Please log in again.';
                } else if (error.message.includes('HTTP 403')) {
                    errorMessage = 'You do not have permission to perform this action.';
                } else if (error.message.includes('HTTP 404')) {
                    errorMessage = 'The requested resource was not found.';
                } else if (error.message.includes('HTTP 500')) {
                    errorMessage = 'Server error. Please try again later.';
                }
                
                showNotification(errorMessage, 'error');
            }
            
            throw error;
        }
    };
}

// Wrap all API methods with error handling
Object.keys(APIService).forEach(key => {
    APIService[key] = withErrorHandling(APIService[key]);
});

// Export to global scope
window.API = API;
window.APIService = APIService;