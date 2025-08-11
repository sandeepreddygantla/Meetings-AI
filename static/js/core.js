/**
 * Core JavaScript functionality for Meetings AI
 * This file contains essential functions needed for initial page load
 */

// Core global variables
let isProcessing = false;
let conversationHistory = [];
let savedConversations = [];
let currentConversationId = null;
let conversationCounter = 1;

// Storage keys
const STORAGE_KEYS = {
    CONVERSATIONS: 'uhg_conversations',
    CURRENT_ID: 'uhg_current_conversation_id',
    COUNTER: 'uhg_conversation_counter'
};

// Async module loader
const ModuleLoader = {
    loadedModules: new Set(),
    
    async loadModule(moduleName) {
        if (this.loadedModules.has(moduleName)) {
            return true;
        }
        
        try {
            const script = document.createElement('script');
            script.src = `/meetingsai/static/js/modules/${moduleName}.js`;
            script.async = true;
            
            return new Promise((resolve, reject) => {
                script.onload = () => {
                    this.loadedModules.add(moduleName);
                    // Module loaded successfully
                    resolve(true);
                };
                script.onerror = () => {
                    console.error(`Failed to load module ${moduleName}`);
                    reject(false);
                };
                document.head.appendChild(script);
            });
        } catch (error) {
            console.error(`Error loading module ${moduleName}:`, error);
            return false;
        }
    },
    
    async loadModules(moduleNames) {
        const promises = moduleNames.map(name => this.loadModule(name));
        return Promise.all(promises);
    }
};

// Core initialization function
function initializeCore() {
    // Core functionality initialized
    
    // Load essential modules immediately
    ModuleLoader.loadModules(['ui', 'api']).then(() => {
        // Essential modules loaded
        if (typeof initializeUI === 'function') {
            initializeUI();
        }
    });
    
    // Load non-essential modules after a delay
    setTimeout(() => {
        ModuleLoader.loadModules(['upload', 'chat', 'document-management']).then(() => {
            // Additional modules loaded
        });
    }, 1000);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCore);
} else {
    initializeCore();
}