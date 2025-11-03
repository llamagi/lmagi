# settings.py (c) Gregory L. Magnusson MIT license 2024
# Settings persistence system with config file that regenerates if deleted

import os
import json
import logging
import shutil
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
            'api_timeout': 120,  # Default timeout for API providers (OpenAI, Groq, Together, AI71)
            'ollama_cloud_timeout': 300,  # Timeout for Ollama Cloud streaming requests
            'ollama_cloud_timeout_non_streaming': 120,  # Timeout for Ollama Cloud non-streaming requests
            'ollama_timeout': 10,  # Timeout for local Ollama requests
            'retry_timeout': 30.0,  # Timeout for retry decorator
            'retry_max_attempts': 3  # Maximum retry attempts
        }
        
        # Load settings
        self.settings = self.load_settings()
        
    def load_settings(self):
        """Load settings from config file, or create defaults if file doesn't exist or is corrupted"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    merged = {**self.defaults, **settings}
                    # Check if any keys were missing (added from defaults)
                    missing_keys = set(self.defaults.keys()) - set(settings.keys())
                    if missing_keys:
                        logging.info(f"Added missing keys to config: {missing_keys}")
                        # Save merged settings to ensure all keys are persisted
                        self.settings = merged
                        self.save_settings()
                    else:
                        self.settings = merged
                    logging.info(f"Settings loaded from {self.config_file}")
                    return self.settings
            else:
                # File doesn't exist, create with defaults
                self.settings = self.defaults.copy()
                self.save_settings()
                logging.info(f"Created new config file {self.config_file} with defaults")
                return self.settings
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file {self.config_file}: {e}")
            # Backup corrupted file if possible
            try:
                backup_path = self.config_path.with_suffix('.json.bak')
                if self.config_path.exists():
                    shutil.copy2(self.config_path, backup_path)
                    logging.info(f"Backed up corrupted config to {backup_path}")
            except Exception as backup_error:
                logging.warning(f"Could not backup corrupted config: {backup_error}")
            # Use defaults and regenerate file
            self.settings = self.defaults.copy()
            self.save_settings()
            logging.info(f"Regenerated config file {self.config_file} with defaults after corruption")
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
    
    def clear_logs(self):
        """Clear all log files"""
        cleared = []
        errors = []
        
        # Clear runtime_monitor.log
        try:
            log_path = Path('runtime_monitor.log')
            if log_path.exists():
                log_path.unlink()
                cleared.append('runtime_monitor.log')
        except Exception as e:
            errors.append(f"Error clearing runtime_monitor.log: {e}")
        
        # Clear mindx/errors/log.txt
        try:
            mindx_log = Path('mindx/errors/log.txt')
            if mindx_log.exists():
                mindx_log.unlink()
                cleared.append('mindx/errors/log.txt')
        except Exception as e:
            errors.append(f"Error clearing mindx/errors/log.txt: {e}")
        
        # Clear memory/truth/logs.txt
        try:
            truth_log = Path('memory/truth/logs.txt')
            if truth_log.exists():
                truth_log.unlink()
                cleared.append('memory/truth/logs.txt')
        except Exception as e:
            errors.append(f"Error clearing memory/truth/logs.txt: {e}")
        
        return {
            'success': len(errors) == 0,
            'cleared': cleared,
            'errors': errors,
            'count': len(cleared)
        }
    
    def clear_mindx_data(self):
        """Clear all JSON files in mindx directory"""
        cleared = 0
        errors = []
        
        try:
            mindx_path = Path('mindx')
            if mindx_path.exists():
                # Clear all internalmemory.json files
                for file in mindx_path.glob('*internalmemory.json'):
                    try:
                        file.unlink()
                        cleared += 1
                    except Exception as e:
                        errors.append(f"Error deleting {file.name}: {e}")
                
                # Clear all nopremise files
                for file in mindx_path.glob('nopremise*'):
                    try:
                        file.unlink()
                        cleared += 1
                    except Exception as e:
                        errors.append(f"Error deleting {file.name}: {e}")
        except Exception as e:
            errors.append(f"Error accessing mindx directory: {e}")
        
        return {
            'success': len(errors) == 0,
            'cleared': cleared,
            'errors': errors
        }
    
    def clear_memory_data(self):
        """Clear all memory files (STM, LTM, truth, logs)"""
        cleared = []
        errors = []
        
        # Clear STM (short-term memory)
        try:
            stm_path = Path('memory/stm')
            if stm_path.exists():
                count = 0
                for file in stm_path.glob('*.json'):
                    try:
                        file.unlink()
                        count += 1
                    except Exception as e:
                        errors.append(f"Error deleting {file.name}: {e}")
                if count > 0:
                    cleared.append(f'memory/stm ({count} files)')
        except Exception as e:
            errors.append(f"Error clearing memory/stm: {e}")
        
        # Clear LTM (long-term memory)
        try:
            ltm_path = Path('memory/ltm')
            if ltm_path.exists():
                count = 0
                for file in ltm_path.glob('*.json'):
                    try:
                        file.unlink()
                        count += 1
                    except Exception as e:
                        errors.append(f"Error deleting {file.name}: {e}")
                if count > 0:
                    cleared.append(f'memory/ltm ({count} files)')
        except Exception as e:
            errors.append(f"Error clearing memory/ltm: {e}")
        
        # Clear truth files (but keep logs.txt)
        try:
            truth_path = Path('memory/truth')
            if truth_path.exists():
                count = 0
                for file in truth_path.glob('*.json'):
                    try:
                        file.unlink()
                        count += 1
                    except Exception as e:
                        errors.append(f"Error deleting {file.name}: {e}")
                if count > 0:
                    cleared.append(f'memory/truth ({count} files)')
        except Exception as e:
            errors.append(f"Error clearing memory/truth: {e}")
        
        # Clear episodic memory
        try:
            episodic_path = Path('memory/episodic')
            if episodic_path.exists():
                count = 0
                for file in episodic_path.glob('*.json'):
                    try:
                        file.unlink()
                        count += 1
                    except Exception as e:
                        errors.append(f"Error deleting {file.name}: {e}")
                if count > 0:
                    cleared.append(f'memory/episodic ({count} files)')
        except Exception as e:
            errors.append(f"Error clearing memory/episodic: {e}")
        
        # Clear memory/logs JSON files
        try:
            logs_path = Path('memory/logs')
            if logs_path.exists():
                count = 0
                for file in logs_path.glob('*.json'):
                    try:
                        file.unlink()
                        count += 1
                    except Exception as e:
                        errors.append(f"Error deleting {file.name}: {e}")
                if count > 0:
                    cleared.append(f'memory/logs ({count} files)')
        except Exception as e:
            errors.append(f"Error clearing memory/logs: {e}")
        
        total_files = sum(
            int(c.split('(')[1].split()[0]) 
            for c in cleared 
            if '(' in c and c.split('(')[1].split()[0].isdigit()
        )
        
        return {
            'success': len(errors) == 0,
            'cleared': cleared,
            'errors': errors,
            'total_files': total_files
        }
    
    def clear_all_data(self):
        """Clear all logs and data files"""
        results = {
            'logs': self.clear_logs(),
            'mindx': self.clear_mindx_data(),
            'memory': self.clear_memory_data()
        }
        
        total_success = all(r['success'] for r in results.values())
        total_cleared = results['logs']['count'] + results['mindx']['cleared'] + results['memory']['total_files']
        all_errors = []
        for r in results.values():
            all_errors.extend(r.get('errors', []))
        
        return {
            'success': total_success,
            'logs': results['logs'],
            'mindx': results['mindx'],
            'memory': results['memory'],
            'total_cleared': total_cleared,
            'errors': all_errors
        }

