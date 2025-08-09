/**
 * Mentions module for @ and # functionality.
 * Handles project, meeting, date, file, and folder mentions.
 */

class MentionHandler {
    constructor(basePath = '/meetingsai') {
        this.basePath = basePath;
        
        // Data storage
        this.availableProjects = [];
        this.availableMeetings = [];
        this.availableFolders = [];
        this.availableDocuments = [];
        
        // UI state
        this.currentMentionType = null;
        this.isDropdownOpen = false;
        this.selectedIndex = -1;
        
        // Initialize
        this.initializeEventListeners();
        this.loadMentionData();
    }
    
    /**
     * Initialize event listeners
     */
    initializeEventListeners() {
        const messageInput = document.getElementById('message-input');
        if (messageInput) {
            messageInput.addEventListener('input', (e) => this.handleInput(e));
            messageInput.addEventListener('keydown', (e) => this.handleKeydown(e));
        }
        
        // Click outside to close dropdown
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#message-input') && !e.target.closest('.mention-dropdown')) {
                this.hideDropdown();
            }
        });
    }
    
    /**
     * Load mention data from server
     */
    async loadMentionData() {
        try {
            await Promise.all([
                this.loadProjects(),
                this.loadMeetings(),
                this.loadDocuments(),
                this.loadFolders()
            ]);
        } catch (error) {
            console.error('Error loading mention data:', error);
        }
    }

    /**
     * Set preloaded data from app initializer to avoid API calls
     */
    setPreloadedData(data) {
        // Using preloaded data
        
        if (data.projects) {
            this.availableProjects = data.projects;
            // Projects set from preloaded data
        }
        
        if (data.meetings) {
            this.availableMeetings = data.meetings;
            // Meetings set from preloaded data
        }
        
        if (data.documents) {
            this.availableDocuments = data.documents;
            // Documents set from preloaded data
        }
        
        // Process folders from documents
        this.loadFolders();
    }
    
    /**
     * Load projects data
     */
    async loadProjects() {
        try {
            const response = await fetch(`${this.basePath}/api/projects`);
            if (response.ok) {
                const data = await response.json();
                this.availableProjects = data.projects || [];
                // Projects loaded
            }
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    }
    
    /**
     * Load meetings data  
     */
    async loadMeetings() {
        try {
            const response = await fetch(`${this.basePath}/api/meetings`);
            if (response.ok) {
                const data = await response.json();
                this.availableMeetings = data.meetings || [];
                // Meetings loaded
            }
        } catch (error) {
            console.error('Error loading meetings:', error);
        }
    }
    
    /**
     * Load documents data
     */
    async loadDocuments() {
        try {
            const response = await fetch(`${this.basePath}/api/documents`);
            if (response.ok) {
                const data = await response.json();
                this.availableDocuments = data.documents || [];
                // Documents loaded
            }
        } catch (error) {
            console.error('Error loading documents:', error);
        }
    }

    
    /**
     * Load folders data (placeholder - would need folder API)
     */
    async loadFolders() {
        // Placeholder - extract folder structure from documents
        try {
            const folders = new Set();
            this.availableDocuments.forEach(doc => {
                if (doc.folder_path) {
                    folders.add(doc.folder_path);
                }
            });
            this.availableFolders = Array.from(folders).map(path => ({ path }));
            // Folders extracted
        } catch (error) {
            console.error('Error loading folders:', error);
        }
    }
    
    /**
     * Handle input events
     */
    handleInput(event) {
        const input = event.target;
        const mention = this.detectMention(input);
        
        if (mention) {
            this.showDropdown(mention);
        } else {
            this.hideDropdown();
        }
    }
    
    /**
     * Handle keydown events for dropdown navigation
     */
    handleKeydown(event) {
        if (!this.isDropdownOpen) return;
        
        const dropdown = document.querySelector('.mention-dropdown');
        if (!dropdown) return;
        
        const items = dropdown.querySelectorAll('.mention-item');
        
        switch (event.key) {
            case 'ArrowDown':
                event.preventDefault();
                this.selectedIndex = Math.min(this.selectedIndex + 1, items.length - 1);
                this.updateDropdownSelection(items);
                break;
                
            case 'ArrowUp':
                event.preventDefault();
                this.selectedIndex = Math.max(this.selectedIndex - 1, -1);
                this.updateDropdownSelection(items);
                break;
                
            case 'Enter':
            case 'Tab':
                event.preventDefault();
                if (this.selectedIndex >= 0 && items[this.selectedIndex]) {
                    items[this.selectedIndex].click();
                }
                break;
                
            case 'Escape':
                event.preventDefault();
                this.hideDropdown();
                break;
        }
    }
    
    /**
     * Detect mention in input
     */
    detectMention(input) {
        const text = input.value;
        const cursorPos = input.selectionStart;
        
        // Find @ or # symbols and check if cursor is after one
        let mentionPos = -1;
        let symbol = '';
        
        for (let i = cursorPos - 1; i >= 0; i--) {
            if (text[i] === '@' || text[i] === '#') {
                // Check if symbol is at start or after whitespace
                if (i === 0 || /\\s/.test(text[i - 1])) {
                    mentionPos = i;
                    symbol = text[i];
                    break;
                }
            } else if (/\\s/.test(text[i])) {
                break; // Stop at whitespace
            }
        }
        
        if (mentionPos >= 0) {
            const mentionText = text.substring(mentionPos + 1, cursorPos);
            return this.parseMention(symbol, mentionText, mentionPos);
        }
        
        return null;
    }
    
    /**
     * Parse mention text based on symbol
     */
    parseMention(symbol, mentionText, position) {
        if (symbol === '#') {
            return this.parseFolderMention(mentionText, position);
        } else if (symbol === '@') {
            return this.parseAtMention(mentionText, position);
        }
        
        return null;
    }
    
    /**
     * Parse @ mention (project:name, meeting:name, date:today, file:name)
     */
    parseAtMention(mentionText, position) {
        const colonIndex = mentionText.indexOf(':');
        
        if (colonIndex === -1) {
            // Simple @ mention - show all types
            return {
                type: 'document',
                prefix: '',
                value: mentionText,
                searchText: mentionText,
                position: position
            };
        }
        
        const prefix = mentionText.substring(0, colonIndex);
        const value = mentionText.substring(colonIndex + 1);
        
        const validPrefixes = ['project', 'meeting', 'date', 'file'];
        
        if (validPrefixes.includes(prefix)) {
            return {
                type: prefix,
                prefix: prefix,
                value: value,
                searchText: value,
                position: position
            };
        }
        
        return null;
    }
    
    /**
     * Parse # mention (folder, folder>)
     */
    parseFolderMention(mentionText, position) {
        if (mentionText.endsWith('>')) {
            return {
                type: 'folder_files',
                value: mentionText.slice(0, -1),
                searchText: mentionText.slice(0, -1),
                position: position
            };
        } else {
            return {
                type: 'folder',
                value: mentionText,
                searchText: mentionText,
                position: position
            };
        }
    }
    
    /**
     * Show mention dropdown - DISABLED: Main script.js handles enhanced dropdowns
     */
    showDropdown(mention) {
        // Disabled: The main script.js already handles enhanced dropdowns
        // This prevents the duplicate dropdown that shows "No results found"
        // showDropdown() disabled - using main script enhanced dropdown
        return;
    }
    
    /**
     * Hide mention dropdown - DISABLED: Main script.js handles enhanced dropdowns
     */
    hideDropdown() {
        // Disabled: The main script.js already handles enhanced dropdowns
        // Clean up any leftover mention-dropdown elements just in case
        const dropdown = document.querySelector('.mention-dropdown');
        if (dropdown) {
            dropdown.remove();
        }
        this.isDropdownOpen = false;
        this.selectedIndex = -1;
        this.currentMentionType = null;
    }
    
    /**
     * Get suggestions based on mention type
     */
    getSuggestions(mention) {
        const searchText = mention.searchText.toLowerCase();
        
        switch (mention.type) {
            case 'project':
                return this.filterProjects(searchText);
                
            case 'meeting':
                return this.filterMeetings(searchText);
                
            case 'date':
                return this.getDateSuggestions(searchText);
                
            case 'file':
                return this.filterDocuments(searchText);
                
            case 'folder':
                return this.filterFolders(searchText);
                
            case 'folder_files':
                return this.getFilesInFolder(mention.value);
                
            default:
                // Generic document search
                return this.filterDocuments(searchText);
        }
    }
    
    /**
     * Filter projects by search text
     */
    filterProjects(searchText) {
        if (!searchText.trim()) {
            return this.availableProjects.slice(0, 10);
        }
        
        return this.availableProjects
            .filter(project => 
                project.project_name.toLowerCase().includes(searchText) ||
                project.description.toLowerCase().includes(searchText)
            )
            .slice(0, 10);
    }
    
    /**
     * Filter meetings by search text
     */
    filterMeetings(searchText) {
        if (!searchText.trim()) {
            return this.availableMeetings.slice(0, 10);
        }
        
        return this.availableMeetings
            .filter(meeting => 
                meeting.title.toLowerCase().includes(searchText)
            )
            .slice(0, 10);
    }
    
    /**
     * Get date suggestions
     */
    getDateSuggestions(searchText) {
        const dateSuggestions = [
            { value: 'today', display: 'Today' },
            { value: 'yesterday', display: 'Yesterday' },
            { value: 'this week', display: 'This Week' },
            { value: 'last week', display: 'Last Week' },
            { value: 'this month', display: 'This Month' }
        ];
        
        if (!searchText.trim()) {
            return dateSuggestions;
        }
        
        return dateSuggestions.filter(item => 
            item.value.toLowerCase().includes(searchText) ||
            item.display.toLowerCase().includes(searchText)
        );
    }
    
    /**
     * Filter documents by search text
     */
    filterDocuments(searchText) {
        if (!searchText.trim()) {
            return this.availableDocuments.slice(0, 10);
        }
        
        return this.availableDocuments
            .filter(doc => 
                doc.filename.toLowerCase().includes(searchText) ||
                doc.original_filename.toLowerCase().includes(searchText)
            )
            .slice(0, 10);
    }
    
    /**
     * Filter folders by search text
     */
    filterFolders(searchText) {
        if (!searchText.trim()) {
            return this.availableFolders.slice(0, 10);
        }
        
        return this.availableFolders
            .filter(folder => 
                folder.path.toLowerCase().includes(searchText)
            )
            .slice(0, 10);
    }
    
    /**
     * Get files in a specific folder
     */
    getFilesInFolder(folderPath) {
        return this.availableDocuments
            .filter(doc => doc.folder_path === folderPath)
            .slice(0, 10);
    }
    
    /**
     * Create dropdown element
     */
    createDropdown() {
        const dropdown = document.createElement('div');
        dropdown.className = 'mention-dropdown';
        dropdown.style.cssText = `
            position: absolute;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            min-width: 200px;
        `;
        
        document.body.appendChild(dropdown);
        return dropdown;
    }
    
    /**
     * Populate dropdown with suggestions
     */
    populateDropdown(dropdown, suggestions, mention) {
        dropdown.innerHTML = '';
        
        if (suggestions.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'mention-item';
            noResults.textContent = 'No results found';
            noResults.style.cssText = `
                padding: 8px 12px;
                color: #666;
                font-style: italic;
            `;
            dropdown.appendChild(noResults);
            return;
        }
        
        suggestions.forEach((item, index) => {
            const element = this.createSuggestionElement(item, mention.type, index);
            dropdown.appendChild(element);
        });
    }
    
    /**
     * Create suggestion element
     */
    createSuggestionElement(item, type, index) {
        const element = document.createElement('div');
        element.className = 'mention-item';
        element.style.cssText = `
            padding: 8px 12px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
            transition: background-color 0.2s;
        `;
        
        // Set content based on type
        let displayText = '';
        let secondaryText = '';
        
        switch (type) {
            case 'project':
                displayText = item.project_name;
                secondaryText = item.description;
                break;
            case 'meeting':
                displayText = item.title;
                secondaryText = item.date ? new Date(item.date).toLocaleDateString() : '';
                break;
            case 'date':
                displayText = item.display || item.value;
                break;
            case 'file':
            case 'document':
                displayText = item.filename;
                secondaryText = item.original_filename !== item.filename ? item.original_filename : '';
                break;
            case 'folder':
                displayText = item.path;
                break;
            default:
                displayText = item.name || item.title || item.value;
        }
        
        element.innerHTML = `
            <div class="mention-primary">${displayText}</div>
            ${secondaryText ? `<div class="mention-secondary" style="font-size: 0.8em; color: #666;">${secondaryText}</div>` : ''}
        `;
        
        // Add event listeners
        element.addEventListener('mouseenter', () => {
            this.selectedIndex = index;
            this.updateDropdownSelection();
        });
        
        element.addEventListener('click', () => {
            this.selectMention(item, type);
        });
        
        return element;
    }
    
    /**
     * Update dropdown selection highlighting
     */
    updateDropdownSelection(items = null) {
        const dropdown = document.querySelector('.mention-dropdown');
        if (!dropdown) return;
        
        if (!items) {
            items = dropdown.querySelectorAll('.mention-item');
        }
        
        items.forEach((item, index) => {
            if (index === this.selectedIndex) {
                item.style.backgroundColor = '#f0f8ff';
            } else {
                item.style.backgroundColor = '';
            }
        });
    }
    
    /**
     * Position dropdown near input
     */
    positionDropdown(dropdown) {
        const input = document.getElementById('message-input');
        if (!input) return;
        
        const rect = input.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        dropdown.style.left = rect.left + 'px';
        dropdown.style.top = (rect.bottom + scrollTop + 5) + 'px';
    }
    
    /**
     * Select a mention and replace in input
     */
    selectMention(item, type) {
        const input = document.getElementById('message-input');
        if (!input) return;
        
        const mention = this.detectMention(input);
        if (!mention) return;
        
        // Build replacement text
        let replacement = '';
        let displayName = '';
        
        switch (type) {
            case 'project':
                displayName = item.project_name;
                replacement = `@project:${displayName}`;
                break;
            case 'meeting':
                displayName = item.title;
                replacement = `@meeting:${displayName}`;
                break;
            case 'date':
                displayName = item.value || item.display;
                replacement = `@date:${displayName}`;
                break;
            case 'file':
            case 'document':
                displayName = item.filename;
                replacement = `@file:${displayName}`;
                break;
            case 'folder':
                replacement = `#${item.path}`;
                break;
            case 'folder_files':
                replacement = `#${item.path}>`;
                break;
        }
        
        // Replace in input
        const text = input.value;
        const beforeMention = text.substring(0, mention.position);
        const afterMention = text.substring(input.selectionStart);
        
        input.value = beforeMention + replacement + ' ' + afterMention;
        
        // Set cursor position
        const newPosition = beforeMention.length + replacement.length + 1;
        input.setSelectionRange(newPosition, newPosition);
        
        // Hide dropdown
        this.hideDropdown();
        
        // Trigger input event
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    
    /**
     * Parse message for mentions and return structured data
     */
    parseMessageForMentions(message) {
        const mentions = [];
        
        // Parse @ mentions
        const atMentionRegex = /@(project|meeting|date|file):([^@?]+?)(?=\s+(?:what|how|when|where|why|who|tell|explain|show|give|list|can|could|would|should|do|did|does|is|are|were|was)|$|@)/gi;
        let match;
        
        while ((match = atMentionRegex.exec(message)) !== null) {
            mentions.push({
                type: match[1],
                value: match[2].trim(),
                fullMatch: match[0]
            });
        }
        
        // Parse # mentions
        const folderMentionRegex = /#([^#\\s>]+)(>?)/g;
        while ((match = folderMentionRegex.exec(message)) !== null) {
            mentions.push({
                type: match[2] === '>' ? 'folder_files' : 'folder',
                value: match[1],
                fullMatch: match[0]
            });
        }
        
        return mentions;
    }
    
    /**
     * Clean message by removing mention syntax
     */
    cleanMessageFromMentions(message) {
        let cleanMessage = message.trim();
        
        // Remove @ mentions
        cleanMessage = cleanMessage.replace(/@(project|meeting|date|file):([^@\\s]+)/g, '');
        
        // Remove # mentions
        cleanMessage = cleanMessage.replace(/#([^#\\s>]+)(>?)/g, '');
        
        // Clean up extra spaces
        cleanMessage = cleanMessage.replace(/\\s+/g, ' ').trim();
        
        return cleanMessage;
    }
}

// Export for global usage
window.MentionHandler = MentionHandler;

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MentionHandler;
}