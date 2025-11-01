# openmind.py
# openmind (c) 2024 Gregory L. Magnusson MIT licence
# internal reasoning loop for continuous AGI reasoning without user interaction
# openmind internal reasoning asynchronous task ensuring non-blocking execution and efficient concurrency
# modular integration of automind reasoning with memory
# ollama model  handling from ollama_handler.py for input response
# API handlling from api.py and chatter.py for openai, together.ai, groq.com, ai71.ai
# log internal reasoning conclusion     ./memory/logs/thoughts.json
# log not premise                       ./memory/logs/notpremise.json
# log short term memory input response  ./memory/stm/{timestamp}memory.json 


import os
import time
from datetime import datetime
from nicegui import ui  # importing ui for easyAGI
from memory.memory import create_memory_folders, store_in_stm, save_conversation_memory, save_internal_reasoning, DialogEntry, save_valid_truth
from webmind.ollama_handler import OllamaHandler  # Import OllamaHandler for modular Ollama interactions
from automind.automind import FundamentalAGI
from webmind.chatter import GPT4o, GroqModel, TogetherModel, AI71Model, OllamaModel, OllamaCloudModel
from webmind.api import APIManager
import ujson as json
import asyncio
import logging
import httpx

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class OpenMind:
    def __init__(self):
        self.api_manager = APIManager()
        self.agi_instance = None
        self.initialize_memory()
        self.message_container = ui.column()
        # Initialize OllamaHandler with settings
        try:
            from webmind.settings import SettingsManager
            settings = SettingsManager()
            ollama_url = settings.get('ollama_base_url', 'http://localhost:11434')
            self.ollama_handler = OllamaHandler(base_url=ollama_url)
        except:
            self.ollama_handler = OllamaHandler()  # Fallback to default
        self.internal_queue = asyncio.Queue()
        self.prompt = ""  # Initialize an empty prompt field
        self.keys_container = ui.column()  # initialize keys_container
        self.log = None  # placeholder for log
        self.initialization_warning_shown = False
        self.autonomous_reasoning = False  # Track autonomous reasoning state
        self.reasoning_task = None  # Track reasoning task for cancellation
        self.tasks = set()  # Track all asyncio tasks for cleanup

    def initialize_memory(self):
        create_memory_folders()

    def _create_task(self, coro):
        """Create and track an asyncio task"""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def cleanup(self):
        """Cancel all tracked tasks and clean up resources"""
        # Cancel reasoning task if it exists
        if self.reasoning_task and not self.reasoning_task.done():
            self.reasoning_task.cancel()
        
        # Cancel all tracked tasks
        for task in self.tasks.copy():
            if not task.done():
                task.cancel()
        
        # Wait for all cancellations to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        
        # Clear reasoning task reference
        self.reasoning_task = None

    def use_api_key(self, service, key):
        self.api_manager.api_keys[service] = key
        self._create_task(self.initialize_agi())
        if self.message_container.client.connected:
            with self.message_container:
                ui.notify(f'Using API key for {service}', type='positive')
        logging.info(f'Using API key for {service}')

    def add_api_key(self):
        service = str(self.service_input.value).strip() if self.service_input.value else ''
        api_key = self.key_input.value.strip()
        logging.debug(f"Adding API key for {service}: {api_key[:4]}...{api_key[-4:]}")
        if service and api_key:
            self.api_manager.api_keys[service] = api_key
            self.api_manager.save_api_key(service, api_key)
            self._create_task(self.initialize_agi())
            if self.message_container.client.connected:
                ui.notify(f'API key for {service} added and loaded successfully')
            # Reset inputs
            self.service_input.value = None  # Clear select dropdown
            self.key_input.value = ''
            # Refresh API menu if it exists
            if hasattr(self, 'refresh_api_menu'):
                try:
                    self.refresh_api_menu()
                except Exception:
                    pass
            ui.run_javascript('setTimeout(() => { window.location.href = "/"; }, 1000);')
        else:
            ui.notify('Provide both service name and API key')

    def delete_api_key(self, service):
        logging.debug(f"Deleting API key for {service}")
        if service in self.api_manager.api_keys:
            del self.api_manager.api_keys[service]
            self.api_manager.remove_api_key(service)
            self._create_task(self.initialize_agi())
            if self.message_container.client.connected:
                ui.notify(f'API key for {service} removed successfully')
            self.list_api_keys()  # Refresh the list after deletion
        else:
            ui.notify(f'No API key found for {service}')

    def list_api_keys(self):
        if self.api_manager.api_keys:
            keys_list = [(service, key) for service, key in self.api_manager.api_keys.items()]
            logging.debug(f"Stored API keys: {keys_list}")
            if self.keys_container.client.connected:
                self.keys_container.clear()
                for service, key in keys_list:
                    with self.keys_container:
                        ui.label(f"{service}: {key[:4]}...{key[-4:]}").classes('flex-1')
                        ui.button('Delete', on_click=lambda s=service: self.delete_api_key(s)).classes('ml-4')
                ui.notify('Stored API keys:\n' + "\n".join([f"{service}: {key[:4]}...{key[-4:]}" for service, key in keys_list]))
        else:
            if self.keys_container.client.connected:
                ui.notify('No API keys in storage')
                self.keys_container.clear()
                with self.keys_container:
                    ui.label('No API keys in storage')

    async def select_model(self, model_name):
        def notify_user(message, message_type='info'):
            if self.message_container.client.connected:
                with self.message_container:
                    ui.notify(message, type=message_type)

        def log_and_notify(message, level='info', message_type='info'):
            if level == 'debug':
                logging.debug(message)
            elif level == 'warning':
                logging.warning(message)
            notify_user(message, message_type)

        # Clean up any existing AGI instance or chatter model
        if hasattr(self, 'agi_instance') and self.agi_instance is not None:
            self.agi_instance = None

        model_initialized = False

        if model_name == 'openai':
            openai_key = self.api_manager.get_api_key('openai')
            if openai_key:
                chatter = GPT4o(openai_key)  # openai
                self.agi_instance = FundamentalAGI(chatter)
                log_and_notify('Using OpenAI for AGI')
                model_initialized = True
            else:
                log_and_notify('OpenAI API key not found. Please add the key first.', 'warning', 'negative')

        if model_name == 'groq' and not model_initialized:
            groq_key = self.api_manager.get_api_key('groq')
            if groq_key:
                chatter = GroqModel(groq_key)  # groq
                self.agi_instance = FundamentalAGI(chatter)
                log_and_notify('Using Groq for AGI')
                model_initialized = True
            else:
                log_and_notify('Groq API key not found. Please add the key first.', 'warning', 'negative')

        if model_name == 'together' and not model_initialized:
            together_key = self.api_manager.get_api_key('together')
            if together_key:
                chatter = TogetherModel(together_key)  # together
                self.agi_instance = FundamentalAGI(chatter)
                log_and_notify('Using Together AI for AGI')
                model_initialized = True
            else:
                log_and_notify('Together AI API key not found. Please add the key first.', 'warning', 'negative')

        if model_name == 'ai71' and not model_initialized:
            ai71_key = self.api_manager.get_api_key('ai71')
            if ai71_key:
                chatter = AI71Model(ai71_key)  # ai71
                self.agi_instance = FundamentalAGI(chatter)
                log_and_notify('Using AI71 for AGI')
                model_initialized = True
            else:
                log_and_notify('AI71 API key not found. Please add the key first.', 'warning', 'negative')

        if model_name == 'ollama_cloud' and not model_initialized:
            ollama_cloud_key = self.api_manager.get_api_key('ollama_cloud')
            if ollama_cloud_key:
                # Get base URL from settings
                try:
                    from webmind.settings import SettingsManager
                    settings = SettingsManager()
                    base_url = settings.get('ollama_cloud_base_url', 'https://ollama.com')
                except:
                    base_url = 'https://ollama.com'
                
                chatter = OllamaCloudModel(ollama_cloud_key, base_url=base_url)
                self.agi_instance = FundamentalAGI(chatter)
                log_and_notify('Using Ollama Cloud for AGI')
                model_initialized = True
            else:
                log_and_notify('Ollama Cloud API key not found. Please add the key first.', 'warning', 'negative')

        if not model_initialized:
            log_and_notify(f'Failed to initialize AGI with {model_name}', 'warning', 'negative')

    async def initialize_agi(self):
        openai_key = self.api_manager.get_api_key('openai')
        groq_key = self.api_manager.get_api_key('groq')
        together_key = self.api_manager.get_api_key('together')
        ai71_key = self.api_manager.get_api_key('ai71')
        ollama_cloud_key = self.api_manager.get_api_key('ollama_cloud')
        llama_running = self.check_llama_running()

        if openai_key:
            chatter = GPT4o(openai_key)
            self.agi_instance = FundamentalAGI(chatter)
            if self.message_container.client.connected:
                with self.message_container:
                    ui.notify('Using OpenAI for ezAGI')
            logging.debug("AGI initialized with OpenAI")
        elif groq_key:
            chatter = GroqModel(groq_key)
            self.agi_instance = FundamentalAGI(chatter)
            if self.message_container.client.connected:
                with self.message_container:
                    ui.notify('Using Groq for ezAGI')
            logging.debug("AGI initialized with Groq")
        elif together_key:
            chatter = TogetherModel(together_key)
            self.agi_instance = FundamentalAGI(chatter)
            if self.message_container.client.connected:
                with self.message_container:
                    ui.notify('Using Together AI for ezAGI')
            logging.debug("AGI initialized with Together AI")
        elif ai71_key:
            chatter = AI71Model(ai71_key)
            self.agi_instance = FundamentalAGI(chatter)
            if self.message_container.client.connected:
                with self.message_container:
                    ui.notify('Using AI71 for ezAGI')
            logging.debug("AGI initialized with AI71")
        elif ollama_cloud_key:
            # Get base URL from settings
            try:
                from webmind.settings import SettingsManager
                settings = SettingsManager()
                base_url = settings.get('ollama_cloud_base_url', 'https://api.ollama.com')
            except:
                base_url = 'https://api.ollama.com'
            
            chatter = OllamaCloudModel(ollama_cloud_key, base_url=base_url)
            self.agi_instance = FundamentalAGI(chatter)
            if self.message_container.client.connected:
                with self.message_container:
                    ui.notify('Using Ollama Cloud for ezAGI')
            logging.debug("AGI initialized with Ollama Cloud")
        elif llama_running:
            # Call ollama_handler to list models when LLaMA is found running
            models = self.ollama_handler.list_models()
            if models and len(models) > 1:  # First line is header
                # Extract first available model name
                first_model = models[1].split()[0] if len(models) > 1 else "llama3"
                chatter = OllamaModel(model=first_model)
                self.agi_instance = FundamentalAGI(chatter)
                model_list = ", ".join([m.split()[0] for m in models[1:]])
                try:
                    if self.message_container and hasattr(self.message_container, 'client') and self.message_container.client.connected:
                        with self.message_container:
                            ui.notify(f'Using Ollama for ezAGI with model: {first_model}')
                except Exception:
                    pass  # Client may have been deleted, ignore
                logging.debug(f"AGI initialized with Ollama model: {first_model}. Available: {model_list}")
            else:
                try:
                    if self.message_container and hasattr(self.message_container, 'client') and self.message_container.client.connected:
                        with self.message_container:
                            ui.notify('LLaMA found running, but no models are available.')
                except Exception:
                    pass  # Client may have been deleted, ignore
                logging.debug("LLaMA running on localhost:11434, but no models are available")
        else:
            self.agi_instance = None
            if not self.initialization_warning_shown:
                try:
                    if self.message_container and hasattr(self.message_container, 'client') and self.message_container.client.connected:
                        with self.message_container:
                            ui.notify('No valid API key or LLaMA instance found. Please add an API key or start LLaMA')
                except Exception:
                    pass  # Client may have been deleted, ignore
                logging.debug("No valid API key or LLaMA instance found. AGI not initialized")
                self.initialization_warning_shown = True

    def check_llama_running(self):
        try:
            # Use the configured Ollama base URL
            base_url = self.ollama_handler.base_url
            response = httpx.get(f'{base_url}/api/tags', timeout=2.0)
            if response.status_code == 200:
                return True
        except httpx.RequestError as e:
            logging.debug(f"Ollama connection failed: {e}")
        return False

    async def get_conclusion_from_agi(self, prompt):
        """
        Get a conclusion from the AGI based on the provided prompt
        This method is asynchronous to allow non-blocking operations
        """
        if self.agi_instance is None:
            return "AGI not initialized. Please add an API key or start LLaMA"
        conclusion = await asyncio.get_event_loop().run_in_executor(None, self.agi_instance.get_conclusion_from_agi, prompt)
        return conclusion

    def communicate_response(self, conclusion):
        """
        Log and print the conclusion from the AGI.
        """
        self.display_internal_conclusion(conclusion)
        return conclusion

    async def reasoning_loop(self):
        """
        Internal reasoning loop for continuous AGI reasoning without user interaction
        Uses the last user prompt to continue reasoning about the same topic
        The conclusions are displayed in the response window and saved to ./memory/logs/thoughts.json including ./memory/logs/notpremise.json
        """
        last_conclusion = None  # Track last conclusion for autonomous continuation
        last_prompt = None  # Track last prompt used
        
        while True:
            if self.agi_instance is None:
                openai_key = self.api_manager.get_api_key('openai')
                groq_key = self.api_manager.get_api_key('groq')
                together_key = self.api_manager.get_api_key('together')
                ai71_key = self.api_manager.get_api_key('ai71')
                ollama_cloud_key = self.api_manager.get_api_key('ollama_cloud')
                llama_running = self.check_llama_running()
                if openai_key or groq_key or together_key or ai71_key or ollama_cloud_key or llama_running:
                    await self.initialize_agi()
                else:
                    if not self.initialization_warning_shown:
                        logging.debug("Waiting for API key or LLaMA instance...")
                        try:
                            if self.message_container and hasattr(self.message_container, 'client') and self.message_container.client.connected:
                                with self.message_container:
                                    ui.notify('AGI not initialized. Add an API key or start LLaMA.')
                        except Exception:
                            pass  # Client may have been deleted, ignore
                        self.initialization_warning_shown = True
                    await asyncio.sleep(30)  # Wait before checking again
                    continue

            # Autonomous reasoning: prioritize user's last prompt, then continue reasoning from last conclusion
            if self.prompt and self.prompt.strip():
                # User has provided a new prompt - use it and update tracking
                prompt = self.prompt
                last_prompt = prompt
                last_conclusion = None  # Reset conclusion context since we have new prompt
            elif last_prompt and last_conclusion:
                # Continue reasoning about the same topic using last conclusion as context
                prompt = f"Continuing from: {last_prompt}\n\nPrevious conclusion: {last_conclusion[:300]}...\n\nWhat deeper insights or related aspects can we explore?"
            elif last_prompt:
                # We have a prompt but no conclusion yet - re-examine the prompt
                prompt = f"Re-examining: {last_prompt}\n\nWhat additional perspectives or implications should we consider?"
            else:
                # No user input yet - wait for user prompt before autonomous reasoning
                logging.debug("No prompt available for autonomous reasoning. Waiting for user input...")
                await asyncio.sleep(10)
                continue
            
            conclusion = await self.get_conclusion_from_agi(prompt)
            last_conclusion = conclusion  # Store for next iteration
            self.display_internal_conclusion(conclusion)
            save_internal_reasoning({"timestamp": int(time.time()), "prompt": prompt, "conclusion": conclusion})
            try:
                if self.message_container and hasattr(self.message_container, 'client') and self.message_container.client.connected:
                    with self.message_container:
                        ui.notify('Reasoning loop conclusion saved')
            except Exception:
                pass  # Client may have been deleted, ignore

            await asyncio.sleep(10)  # Adjust the delay as necessary

    def display_internal_conclusion(self, conclusion):
        """
        Display internal reasoning conclusion in the response window and log it to a JSON file
        """
        if conclusion != "No premises available for logic as conclusion.":
            if self.message_container.client.connected:
                with self.message_container:
                    response_message = ui.chat_message(name='intr', sent=False)
                    response_message.clear()
                    with response_message:
                        ui.markdown(conclusion)
            logging.info(f"Internal reasoning conclusion: {conclusion}")

        # Determine which log file to write to
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "conclusion": conclusion
        }
        
        if conclusion == "No premises available for logic as conclusion.":
            log_file_path = "./memory/logs/notpremise.json"
        else:
            log_file_path = "./memory/logs/thoughts.json"

        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as file:
                json.dump([log_entry], file, indent=4)
        else:
            with open(log_file_path, 'r+') as file:
                data = json.load(file)
                data.append(log_entry)
                file.seek(0)
                json.dump(data, file, indent=4)

        # Also log the conclusions to conclusions.txt
        with open('./memory/logs/conclusions.txt', 'a') as file:
            file.write(f"{datetime.now().isoformat()}: {conclusion}\n")

    async def main_loop(self):
        """
        Main loop to handle both internal reasoning and user input.
        """
        # Start reasoning loop only if autonomous reasoning is enabled
        if self.autonomous_reasoning:
            reasoning_task = self._create_task(self.reasoning_loop())
            reasoning_task.add_done_callback(self._handle_task_result)

        while True:
            prompt = await self.internal_queue.get()
            if prompt == 'exit':
                break
            self.prompt = prompt  # Update the prompt with the new input
            conclusion = await self.get_conclusion_from_agi(prompt)
            self.communicate_response(conclusion)
            # Save the input-response pair using save_conversation_memory
            save_conversation_memory({"dialog": {"instruction": prompt, "response": conclusion}})

    async def send_message(self, question):
        if self.message_container.client.connected:
            with self.message_container:
                ui.chat_message(text=question, name='query', sent=True)
                response_message = ui.chat_message(name='ezAGI', sent=False)
                spinner = ui.spinner(type='dots')

        try:
            conclusion = await self.get_conclusion_from_agi(question)
            if response_message and self.message_container.client.connected:
                response_message.clear()
                with response_message:
                    ui.markdown(conclusion)

            await self.run_javascript_with_retry('window.scrollTo(0, document.body.scrollHeight)', retries=3, timeout=30.1)

            # Store the dialog entry
            entry = DialogEntry(question, conclusion)
            store_in_stm(entry)
            # saves conversation following each input response to ./memory/stm/timestampmemory.json from memory.py
            save_conversation_memory({"dialog": {"instruction": question, "response": conclusion}})
        except Exception as e:
            logging.error(f"Error getting conclusion from easyAGI: {e}")
            if self.log:
                self.log.push(f"Error getting conclusion from easyAGI: {e}")
        finally:
            try:
                if self.message_container.client.connected:
                    self.message_container.remove(spinner)  # Correctly remove the spinner
            except KeyError:
                logging.warning("Spinner element not found in message_container")

    async def run_javascript_with_retry(self, script, retries=5, timeout=12.0):
        for attempt in range(retries):
            task = asyncio.create_task(ui.run_javascript(script, timeout=timeout))
            task.add_done_callback(self._handle_task_result)
            try:
                await task
                return
            except TimeoutError:
                logging.warning(f"JavaScript did not respond within {timeout} s on attempt {attempt + 1}")
        raise TimeoutError(f"JavaScript did not respond after {retries} attempts")

    def _handle_task_result(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Task cancellation should not be logged as an error.
        except Exception as e:
            logging.exception('Exception raised by task = %r', task)

    def read_log_file(self, file_path):
        """
        Read the content of a log file and return it.
        """
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            logging.error(f"Log file not found: {file_path}")
            return f"Log file not found: {file_path}"
        except Exception as e:
            logging.error(f"Error reading log file {file_path}: {e}")
            return f"Error reading log file {file_path}: {e}"
