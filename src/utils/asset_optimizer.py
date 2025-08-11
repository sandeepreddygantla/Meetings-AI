"""
Asset optimization utilities for better performance.
Handles CSS and JS minification and compression.
"""

import os
import re
import gzip
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AssetOptimizer:
    """Handles asset optimization for better web performance"""
    
    def __init__(self, static_dir: str):
        self.static_dir = static_dir
        self._css_cache = {}
        self._js_cache = {}
    
    def minify_css(self, css_content: str) -> str:
        """Simple CSS minification"""
        try:
            # Remove comments
            css_content = re.sub(r'/\*.*?\*/', '', css_content, flags=re.DOTALL)
            
            # Remove extra whitespace
            css_content = re.sub(r'\s+', ' ', css_content)
            
            # Remove whitespace around specific characters
            css_content = re.sub(r'\s*([{}:;,>+~])\s*', r'\1', css_content)
            
            # Remove trailing semicolons
            css_content = re.sub(r';}', '}', css_content)
            
            # Remove leading/trailing whitespace
            css_content = css_content.strip()
            
            return css_content
            
        except Exception as e:
            logger.warning(f"CSS minification failed: {e}")
            return css_content
    
    def minify_js(self, js_content: str) -> str:
        """Basic JS minification (simple whitespace removal)"""
        try:
            # Remove single-line comments (but preserve URLs)
            js_content = re.sub(r'(?<!:)//.*$', '', js_content, flags=re.MULTILINE)
            
            # Remove multi-line comments
            js_content = re.sub(r'/\*.*?\*/', '', js_content, flags=re.DOTALL)
            
            # Remove extra whitespace but preserve line breaks in strings
            lines = []
            in_string = False
            string_char = None
            
            for line in js_content.split('\n'):
                if not in_string:
                    # Remove leading/trailing whitespace
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Collapse multiple spaces to single space
                    line = re.sub(r'\s+', ' ', line)
                
                lines.append(line)
            
            return '\n'.join(lines)
            
        except Exception as e:
            logger.warning(f"JS minification failed: {e}")
            return js_content
    
    def get_optimized_css(self, filename: str) -> Optional[str]:
        """Get minified CSS content with caching"""
        file_path = os.path.join(self.static_dir, filename)
        
        if not os.path.exists(file_path):
            return None
        
        # Check cache
        file_mtime = os.path.getmtime(file_path)
        cache_key = f"{filename}_{file_mtime}"
        
        if cache_key in self._css_cache:
            return self._css_cache[cache_key]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            minified = self.minify_css(content)
            self._css_cache[cache_key] = minified
            
            # Clean old cache entries
            if len(self._css_cache) > 50:
                self._css_cache.clear()
            
            return minified
            
        except Exception as e:
            logger.error(f"Error reading CSS file {filename}: {e}")
            return None
    
    def get_optimized_js(self, filename: str) -> Optional[str]:
        """Get minified JS content with caching"""
        file_path = os.path.join(self.static_dir, filename)
        
        if not os.path.exists(file_path):
            return None
        
        # Check cache
        file_mtime = os.path.getmtime(file_path)
        cache_key = f"{filename}_{file_mtime}"
        
        if cache_key in self._js_cache:
            return self._js_cache[cache_key]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            minified = self.minify_js(content)
            self._js_cache[cache_key] = minified
            
            # Clean old cache entries
            if len(self._js_cache) > 50:
                self._js_cache.clear()
            
            return minified
            
        except Exception as e:
            logger.error(f"Error reading JS file {filename}: {e}")
            return None
    
    def create_gzipped_version(self, content: str) -> bytes:
        """Create gzipped version of content"""
        try:
            return gzip.compress(content.encode('utf-8'), compresslevel=6)
        except Exception as e:
            logger.error(f"Gzip compression failed: {e}")
            return content.encode('utf-8')
    
    def get_cache_stats(self) -> dict:
        """Get optimization cache statistics"""
        return {
            'css_cache_size': len(self._css_cache),
            'js_cache_size': len(self._js_cache),
            'total_cached_assets': len(self._css_cache) + len(self._js_cache)
        }