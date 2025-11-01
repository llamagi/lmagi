# chatter.py (c) Gregory L. Magnusson MIT license 2024
# modular file to include input response mechanisms for multi-model environment
# API name must be openai, groq, or together from API
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

class OllamaModel:
    def __init__(self, model="llama3"):
        self.api_url = "http://localhost:11434/api"
        self.current_model = model

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
                async with session.post(f"{self.api_url}/generate", json=payload) as response:
                    data = await response.json()
                    if "response" in data:
                        response_content = data["response"]
                    elif "error" in data:
                        logging.error(f"Ollama error: {data['error']}")
                        return f"error: {data['error']}"
            return response_content.lower()
        except Exception as e:
            error_msg = f"Ollama API error: {e}"
            logging.error(error_msg, exc_info=True)
            return "error: unable to generate a response due to an issue with the ollama api."

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
