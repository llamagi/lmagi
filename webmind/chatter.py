# chatter.py (c) Gregory L. Magnusson MIT license 2024
# modular file to include input response mechanisms for multi-model environment
# API name must be openai, groq, together, ai71, or ollama_cloud from API
# ollama integration is from URL

from openai import OpenAI
from groq import Groq
from together import AsyncTogether
from ai71 import AI71  # Import AI71 library
from webmind.utils import retry_with_timeout
import subprocess
import asyncio
import logging
import os


class GPT4o:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.current_model = "gpt-4o"  # Default model

    def set_model(self, model_name):
        self.current_model = model_name

    def get_current_model(self):
        return self.current_model

    def generate_response(self, knowledge):
        prompt = f"{knowledge}"
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ]
            )
            decision = response.choices[0].message.content
            return decision.lower()
        except Exception as e:
            error_msg = f"OpenAI API error: {e}"
            logging.error(error_msg, exc_info=True)
            # Return error string for backwards compatibility
            return "error: unable to generate a response due to an issue with the openai api."

class AI71Model:
    def __init__(self, ai71_api_key):
        self.client = AI71(ai71_api_key)
        self.current_model = "tiiuae/falcon-180B-chat"  # Default model

    def set_model(self, model_name):
        self.current_model = model_name

    def get_current_model(self):
        return self.current_model

    def generate_response(self, knowledge):
        prompt = f"{knowledge}"
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            return response.choices[0].message.content.lower()
        except Exception as e:
            error_msg = f"AI71 API error: {e}"
            logging.error(error_msg, exc_info=True)
            return "error: unable to generate a response due to an issue with the ai71 api."

class GroqModel:
    def __init__(self, groq_api_key):
        self.client = Groq(api_key=groq_api_key)
        self.current_model = "mixtral-8x7b-32768"  # Default model

    def set_model(self, model_name):
        self.current_model = model_name

    def get_current_model(self):
        return self.current_model

    def generate_response(self, knowledge):
        prompt = f"{knowledge}"
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ],
                model=self.current_model,
            )
            decision = chat_completion.choices[0].message.content
            return decision.lower()
        except Exception as e:
            error_msg = f"Groq API error: {e}"
            logging.error(error_msg, exc_info=True)
            return "error: unable to generate a response due to an issue with the groq api."

class OllamaCloudModel:
    """
    Ollama Cloud API integration using Bearer token authentication.
    Supports Ollama Cloud (ollama.com) and custom Ollama Cloud instances.
    """
    def __init__(self, api_key, base_url="https://ollama.com"):
        self.api_key = api_key
        # Ensure base_url doesn't have trailing /api - we'll add it
        if not base_url:
            base_url = "https://ollama.com"
        base_url = base_url.strip().rstrip('/')
        
        # Handle case where base_url already includes /api (e.g., https://api.ollama.com)
        # For Ollama Cloud, the API should be at https://ollama.com/api
        # If user provides https://api.ollama.com, we should use it as-is
        if base_url.endswith('/api'):
            # Remove /api if it's at the end - we'll add it back
            base_url = base_url[:-4]
        elif '://api.' in base_url:
            # If URL contains '://api.', it might be a different API endpoint structure
            # For Ollama Cloud, use https://ollama.com/api, not https://api.ollama.com
            if 'ollama.com' in base_url:
                base_url = 'https://ollama.com'
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.current_model = None  # No default - must be set explicitly
        logging.debug(f"OllamaCloudModel initialized with base_url: {self.base_url}, api_url: {self.api_url}")
    
    def set_model(self, model_name):
        self.current_model = model_name
    
    def get_current_model(self):
        return self.current_model
    
    def list_models(self):
        """
        List all available models from Ollama Cloud API.
        Returns list of model names.
        """
        try:
            import httpx
            if not self.api_key:
                logging.error("Ollama Cloud API key is missing")
                return []
            
            if not self.base_url or not self.api_url:
                logging.error(f"Ollama Cloud base_url or api_url is invalid: base_url={self.base_url}, api_url={self.api_url}")
                return []
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            # Use /api/tags endpoint
            url = f'{self.api_url}/tags'
            logging.info(f"Fetching Ollama Cloud models from: {url}")
            logging.debug(f"Using headers: Authorization=Bearer {self.api_key[:10]}...")
            response = httpx.get(url, headers=headers, timeout=10.0)
            logging.debug(f"Ollama Cloud API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logging.debug(f"Ollama Cloud API response data: {data}")
                models = data.get('models', [])
                # Parse model names according to Ollama API format
                # Models array contains objects with 'name' field (e.g., "gpt-oss:120b-cloud")
                model_names = []
                for model in models:
                    if isinstance(model, dict):
                        # Standard format: model object with 'name' field
                        model_name = model.get('name') or model.get('model')
                        if model_name:
                            model_names.append(model_name)
                    elif isinstance(model, str):
                        # If it's a string, use it directly
                        model_names.append(model)
                
                if model_names:
                    logging.info(f"Found {len(model_names)} Ollama Cloud models: {model_names}")
                else:
                    logging.warning(f"No models found in response. Response data: {data}")
                return model_names
            else:
                error_text = response.text if hasattr(response, 'text') else 'No error details'
                logging.error(f"Ollama Cloud API error: HTTP {response.status_code} - {error_text}")
                return []
        except httpx.ConnectError as e:
            logging.error(f"Ollama Cloud connection error: {e}. URL: {self.api_url}/tags")
            return []
        except Exception as e:
            logging.error(f"Ollama Cloud API error (list_models): {e}", exc_info=True)
            logging.error(f"URL was: {self.api_url}/tags, base_url: {self.base_url}")
            return []
    
    def generate_response(self, knowledge):
        """
        Synchronous wrapper for generate_response_async.
        Uses thread pool executor to avoid event loop conflicts when called from async contexts.
        """
        import concurrent.futures
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Event loop is running - use thread pool executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.generate_response_async(knowledge, self.current_model))
                    return future.result()
            else:
                # No running loop - safe to use asyncio.run
                return asyncio.run(self.generate_response_async(knowledge, self.current_model))
        except RuntimeError:
            # No event loop exists - safe to use asyncio.run
            return asyncio.run(self.generate_response_async(knowledge, self.current_model))
    
    @retry_with_timeout(max_retries=3, timeout=30.0)
    async def generate_response_async(self, knowledge, model=None):
        if model is None:
            model = self.current_model
        
        if not model:
            error_msg = "No model selected. Please select an Ollama Cloud model first."
            logging.error(error_msg)
            raise Exception(error_msg)
        
        import aiohttp
        import ujson as json
        
        # Get timeout from settings
        try:
            from webmind.settings import SettingsManager
            timeout_settings = SettingsManager()
            ollama_cloud_timeout = timeout_settings.get('ollama_cloud_timeout_non_streaming', 60)
        except:
            ollama_cloud_timeout = 60
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Try chat endpoint first (preferred for cloud)
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": knowledge}
            ],
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            # Try chat endpoint first
            try:
                async with session.post(
                    f"{self.api_url}/chat",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=ollama_cloud_timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "message" in data and "content" in data["message"]:
                            return data["message"]["content"].lower()
                        elif "response" in data:
                            return data["response"].lower()
                    else:
                        error_text = await response.text()
                        logging.debug(f"Ollama Cloud /api/chat returned status {response.status}: {error_text}")
            except aiohttp.ClientError as e:
                logging.debug(f"Ollama Cloud /api/chat ClientError: {e}")
            
            # Fallback to generate endpoint
            payload_generate = {
                "model": model,
                "prompt": knowledge,
                "stream": False
            }
            try:
                async with session.post(
                    f"{self.api_url}/generate",
                    json=payload_generate,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=ollama_cloud_timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "response" in data:
                            return data["response"].lower()
                        elif "error" in data:
                            logging.error(f"Ollama Cloud error: {data['error']}")
                            raise Exception(f"Ollama Cloud API error: {data['error']}")
                    else:
                        error_text = await response.text()
                        logging.error(f"Ollama Cloud API error: HTTP {response.status} - {error_text}")
                        raise Exception(f"Ollama Cloud API returned HTTP {response.status}: {error_text}")
            except aiohttp.ClientError as e:
                logging.error(f"Ollama Cloud /api/generate ClientError: {e}")
                raise Exception(f"Ollama Cloud API connection error: {str(e)}")
        
        raise Exception("Unable to generate response from Ollama Cloud API - both /api/chat and /api/generate failed")

class OllamaModel:
    """
    Local Ollama API integration for local or network-based Ollama instances.
    Uses the same API structure as Ollama Cloud but without authentication.
    """
    def __init__(self, model=None, base_url=None):
        # Get base URL from settings if not provided
        if base_url is None:
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
        self.current_model = model  # No default - model must be selected explicitly (like OllamaCloudModel)
    
    def set_model(self, model_name):
        self.current_model = model_name
    
    def get_current_model(self):
        return self.current_model
    
    def generate_response(self, knowledge):
        """
        Synchronous wrapper for generate_response_async.
        Uses thread pool executor to avoid event loop conflicts when called from async contexts.
        """
        import concurrent.futures
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Event loop is running - use thread pool executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.generate_response_async(knowledge, self.current_model))
                    return future.result()
            else:
                # No running loop - safe to use asyncio.run
                return asyncio.run(self.generate_response_async(knowledge, self.current_model))
        except RuntimeError:
            # No event loop exists - safe to use asyncio.run
            return asyncio.run(self.generate_response_async(knowledge, self.current_model))

    @retry_with_timeout(max_retries=3, timeout=30.0)
    async def generate_response_async(self, knowledge, model=None):
        if model is None:
            model = self.current_model
        
        if not model:
            error_msg = "No model selected. Please select an Ollama model first."
            logging.error(error_msg)
            raise Exception(error_msg)
        
        # Get timeout from settings
        try:
            from webmind.settings import SettingsManager
            timeout_settings = SettingsManager()
            ollama_timeout = timeout_settings.get('ollama_timeout', 10)
        except:
            ollama_timeout = 10
        
        try:
            import aiohttp
            import ujson as json
            response_content = ""
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": knowledge,
                    "stream": False  # Non-streaming for sync compatibility
                }
                async with session.post(
                    f"{self.api_url}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=ollama_timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "response" in data:
                            response_content = data["response"]
                        elif "error" in data:
                            logging.error(f"Ollama error: {data['error']}")
                            raise Exception(f"Ollama API error: {data['error']}")
                    else:
                        error_text = await response.text()
                        logging.error(f"Ollama API error: HTTP {response.status} - {error_text}")
                        raise Exception(f"Ollama API returned HTTP {response.status}: {error_text}")
            return response_content.lower()
        except Exception as e:
            error_msg = f"Ollama API error: {e}"
            logging.error(error_msg, exc_info=True)
            raise  # Re-raise to let retry decorator handle it

def check_ollama_installation():
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

class TogetherModel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.async_client = AsyncTogether(api_key=api_key)  # Use the provided api_key directly
        self.current_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Default model

    def set_model(self, model_name):
        self.current_model = model_name

    def get_current_model(self):
        return self.current_model

    @retry_with_timeout(max_retries=3, timeout=30.0)
    async def generate_response_async(self, knowledge):
        messages = [{"role": "user", "content": knowledge}]
        try:
            response = await self.async_client.chat.completions.create(
                model=self.current_model,
                messages=messages
            )
            return response.choices[0].message.content.lower()
        except Exception as e:
            error_msg = f"Together.ai API error: {e}"
            logging.error(error_msg, exc_info=True)
            return "error: unable to generate a response due to an issue with the together.ai api."

    def generate_response(self, knowledge):
        """
        Synchronous wrapper for generate_response_async.
        Uses thread pool executor to avoid event loop conflicts when called from async contexts.
        """
        import concurrent.futures
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Event loop is running - use thread pool executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.generate_response_async(knowledge))
                    return future.result()
            else:
                # No running loop - safe to use asyncio.run
                return asyncio.run(self.generate_response_async(knowledge))
        except RuntimeError:
            # No event loop exists - safe to use asyncio.run
            return asyncio.run(self.generate_response_async(knowledge))
