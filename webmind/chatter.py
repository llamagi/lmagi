# chatter.py (c) Gregory L. Magnusson MIT license 2024
# modular file to include input response mechanisms for multi-model environment
# API name must be openai, groq, or together from API
# ollama integration is from URL

import openai
from groq import Groq
from together import AsyncTogether
from ai71 import AI71  # Import AI71 library
import subprocess
import asyncio
import logging
import os


class GPT4o:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key
        self.current_model = "gpt-4o"  # Default model

    def set_model(self, model_name):
        self.current_model = model_name

    def get_current_model(self):
        return self.current_model

    def generate_response(self, knowledge):
        prompt = f"{knowledge}"
        try:
            response = openai.chatcompletion.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ]
            )
            decision = response.choices[0].message.content
            return decision.lower()
        except openai.error.OpenAIError as e:
            logging.error(f"openai api error: {e}")
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
            logging.error(f"ai71 api error: {e}")
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
            logging.error(f"groq api error: {e}")
            return "error: unable to generate a response due to an issue with the groq api."

class OllamaModel:
    def __init__(self):
        self.api_url = "http://localhost:11434/api"

    async def generate_response_async(self, knowledge, model="llama3"):
        try:
            response_content = ""
            stream = ollama.chat(model=model, messages=[{'role': 'user', 'content': knowledge}], stream=True)
            async for chunk in stream:
                response_content += chunk['message']['content']
            return response_content
        except Exception as e:
            logging.error(f"ollama api error: {e}")
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

    async def generate_response_async(self, knowledge):
        messages = [{"role": "user", "content": knowledge}]
        try:
            response = await self.async_client.chat.completions.create(
                model=self.current_model,
                messages=messages
            )
            return response.choices[0].message.content.lower()
        except Exception as e:
            logging.error(f"together.ai api error: {e}")
            return "error: unable to generate a response due to an issue with the together.ai api."

    def generate_response(self, knowledge):
        return asyncio.run(self.generate_response_async(knowledge))
