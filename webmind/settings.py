# settings.py (c) Gregory L. Magnusson MIT license 2024
# Settings persistence system with config file that regenerates if deleted

import os
import json
import logging
from pathlib import Path

class SettingsManager:
    """
    Manages application settings with persistent storage to config file.
    Config file regenerates with defaults if deleted.
    """
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config_path = Path(config_file)
        
        # Default settings
        self.defaults = {
            'theme': 'everforest',
            'dark_mode': True,
            'autonomous_reasoning': False,
            'sidebar_width': 260,
            'footer_height': 160,
            'ollama_base_url': 'http://localhost:11434',
            'ollama_cloud_base_url': 'https://ollama.com',
            # Timeout settings (in seconds)
            'api_timeout': 60,  # Default timeout for API providers (OpenAI, Groq, Together, AI71)
            'ollama_cloud_timeout': 300,  # Timeout for Ollama Cloud streaming requests
            'ollama_cloud_timeout_non_streaming': 60,  # Timeout for Ollama Cloud non-streaming requests
            'ollama_timeout': 10,  # Timeout for local Ollama requests
            'retry_timeout': 30.0,  # Timeout for retry decorator
            'retry_max_attempts': 3  # Maximum retry attempts
        }
        
        # Load settings
        self.settings = self.load_settings()
        
    def load_settings(self):
        """Load settings from config file, or create defaults if file doesn't exist"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    merged = {**self.defaults, **settings}
                    logging.info(f"Settings loaded from {self.config_file}")
                    return merged
            else:
                # File doesn't exist, create with defaults
                self.settings = self.defaults.copy()
                self.save_settings()
                logging.info(f"Created new config file {self.config_file} with defaults")
                return self.settings
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            # On error, use defaults and regenerate file
            self.settings = self.defaults.copy()
            self.save_settings()
            return self.settings
    
    def save_settings(self):
        """Save settings to config file"""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            logging.info(f"Settings saved to {self.config_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            return False
    
    def get(self, key, default=None):
        """Get a setting value"""
        return self.settings.get(key, default if default is not None else self.defaults.get(key))
    
    def set(self, key, value):
        """Set a setting value and save"""
        self.settings[key] = value
        self.save_settings()
    
    def update(self, **kwargs):
        """Update multiple settings at once"""
        self.settings.update(kwargs)
        self.save_settings()
    
    def reset(self):
        """Reset all settings to defaults"""
        self.settings = self.defaults.copy()
        self.save_settings()
        logging.info("Settings reset to defaults")
    
    def sync_from_localStorage(self, js_settings):
        """Sync settings from browser localStorage (called from JavaScript)"""
        try:
            updated = False
            if 'ui-theme' in js_settings:
                if self.settings.get('theme') != js_settings['ui-theme']:
                    self.set('theme', js_settings['ui-theme'])
                    updated = True
            if 'theme' in js_settings:
                dark_mode = js_settings['theme'] == 'dark'
                if self.settings.get('dark_mode') != dark_mode:
                    self.set('dark_mode', dark_mode)
                    updated = True
            if 'autonomous-reasoning' in js_settings:
                autonomous = js_settings['autonomous-reasoning'] == 'true'
                if self.settings.get('autonomous_reasoning') != autonomous:
                    self.set('autonomous_reasoning', autonomous)
                    updated = True
            if 'sidebar-width' in js_settings:
                width = int(js_settings['sidebar-width']) if js_settings['sidebar-width'] else 260
                if self.settings.get('sidebar_width') != width:
                    self.set('sidebar_width', width)
                    updated = True
            if 'footer-height' in js_settings:
                height = int(js_settings['footer-height']) if js_settings['footer-height'] else 160
                if self.settings.get('footer_height') != height:
                    self.set('footer_height', height)
                    updated = True
            return updated
        except Exception as e:
            logging.error(f"Error syncing from localStorage: {e}")
            return False
    
    def sync_to_localStorage(self):
        """Return settings formatted for JavaScript localStorage"""
        return {
            'ui-theme': self.settings.get('theme', 'everforest'),
            'theme': 'dark' if self.settings.get('dark_mode', True) else 'light',
            'autonomous-reasoning': 'true' if self.settings.get('autonomous_reasoning', False) else 'false',
            'sidebar-width': str(self.settings.get('sidebar_width', 260)),
            'footer-height': str(self.settings.get('footer_height', 160))
        }

