# ollama_handler.py
# code extrapolated from ollama-python for interaction with an ollama installation
# ollama_handler (c) 2024 codephreak MIT licence

import logging
import subprocess
import asyncio
import aiohttp
import ujson as json
from nicegui import ui

class OllamaHandler:
    """
    Class to interact with Llama3 model via the Ollama service.
    """
    def __init__(self, base_url=None):
        if base_url is None:
            # Try to load from settings if available
            try:
                from webmind.settings import SettingsManager
                settings = SettingsManager()
                base_url = settings.get('ollama_base_url', 'http://localhost:11434')
            except:
                base_url = 'http://localhost:11434'
        
        # Ensure base_url doesn't have trailing /api - we'll add it
        base_url = base_url.rstrip('/')
        if base_url.endswith('/api'):
            base_url = base_url[:-4]
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.models = []
        self.selected_model = None
    
    def update_base_url(self, base_url):
        """Update the base URL and rebuild API URL"""
        base_url = base_url.rstrip('/')
        if base_url.endswith('/api'):
            base_url = base_url[:-4]
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        logging.info(f"Ollama base URL updated to: {base_url}")

    def check_installation(self):
        """
        Check if Ollama is installed and accessible.
        """
        command = "ollama list"
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("Ollama is installed and accessible.")
                return True
            else:
                logging.error("Ollama is not accessible.")
                return False
        except Exception as e:
            logging.error(f"Failed to check Ollama installation: {e}")
            return False

    def list_models(self):
        """
        List all available models in the Ollama service.
        Uses HTTP API for remote servers, falls back to CLI for localhost.
        """
        # Use HTTP API for listing models (works for both local and remote)
        try:
            import httpx
            response = httpx.get(f'{self.api_url}/tags', timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                if models:
                    # Format similar to CLI output: header + model lines
                    result = ['NAME\t\tID\t\tSIZE\t\tMODIFIED']
                    for model in models:
                        name = model.get('name', 'unknown')
                        model_id = model.get('model', '')[:12] if model.get('model') else ''
                        size = self._format_size(model.get('size', 0))
                        modified = self._format_modified(model.get('modified_at', ''))
                        result.append(f'{name}\t\t{model_id}\t\t{size}\t\t{modified}')
                    self.models = result
                    return self.models
                else:
                    self.models = ['NAME\t\tID\t\tSIZE\t\tMODIFIED']
                    return self.models
            else:
                logging.error(f"Ollama API error: HTTP {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Ollama API error (HTTP): {e}")
            # Fallback to CLI for localhost if HTTP fails
            try:
                command = "ollama list"
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    self.models = result.stdout.strip().splitlines()
                    return self.models
                else:
                    logging.error(f"Ollama CLI error: {result.stderr}")
                    return []
            except Exception as cli_error:
                logging.error(f"Ollama CLI error: {cli_error}")
                return []
    
    def _format_size(self, size_bytes):
        """Format bytes to human-readable size"""
        if not size_bytes:
            return '0 B'
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f'{size_bytes:.1f} {unit}'
            size_bytes /= 1024.0
        return f'{size_bytes:.1f} PB'
    
    def _format_modified(self, modified_str):
        """Format modified timestamp"""
        if not modified_str:
            return ''
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(modified_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M')
        except:
            return modified_str[:16] if modified_str else ''

    async def generate_response_async(self, knowledge, model="llama3"):
        """
        Generate a response from the Llama3 model based on the given knowledge prompt using streaming.
        """
        try:
            response_content = ""
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": knowledge,
                    "stream": True
                }
                async with session.post(f"{self.api_url}/generate", json=payload) as response:
                    async for line in response.content:
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if "response" in data:
                                response_content += data["response"]
                            elif "error" in data:
                                logging.error(f"Error in response: {data['error']}")
                                return f"Error: {data['error']}"
            return response_content
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            return "Error: Unable to generate a response due to an issue with the Ollama API."

    async def show_ollama_info_async(self, container):
        """
        Show information about the Ollama service.
        """
        command = "ollama show"
        try:
            result = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await result.communicate()
            if result.returncode == 0:
                with container:  # Ensure the correct UI context
                    ui.notify('Ollama information displayed successfully.', type='positive')
                return stdout.decode().strip()
            else:
                logging.error(f"Ollama API error: {stderr.decode().strip()}")
                with container:  # Ensure the correct UI context
                    ui.notify(f'Error displaying Ollama information: {stderr.decode().strip()}', type='negative')
                return ""
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            with container:  # Ensure the correct UI context
                ui.notify(f'Exception occurred while showing Ollama information: {e}', type='negative')
            return ""

    def install_ollama(self):
        """
        deb variant Linux install Ollama on using the provided installation script
        terminal command as subprocess
        """
        command = "sudo curl -fsSL https://ollama.com/install.sh | sh"
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return "Ollama installation successful."
            else:
                logging.error(f"Ollama install error: {result.stderr}")
                return "Error: Unable to install Ollama."
        except Exception as e:
            logging.error(f"Ollama install error: {e}")
            return "Error: Unable to install Ollama."

    async def test_ollama(self):
        """
        Test Ollama by generating a response to a default prompt.
        """
        return await self.generate_response_async("explain easy Augmented Generative Intelligence LLM reasoning enhancement framework", self.selected_model)

    def select_model(self, model_name):
        """
        Select the model to use for generating responses.
        """
        self.selected_model = model_name
        logging.info(f"Selected model: {model_name}")

