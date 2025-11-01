# lmagi.py - Backend web server for easyAGI
# lmagi (c) Gregory L. Magnusson MIT license 2024
# easyAGI (c) Gregory L. Magnusson MIT license 2024
# easy augmented generative intelligence UIUX
# multi-model LLM with automind reasoning from premise to draw_conclusion
# conversation from main_loop(self) is saved to ./memory/stm/timestampmemory.json from memory.py creating short term memory store of input response
# reasoning_loop(self)conversation from internal_conclusions are saved in ./memory/logs/thoughts.json
# 
# ENTRYPOINT: Use lmagi_gui.py to launch the application (preferred method)
# This file can also be run directly for development/testing: python lmagi.py


from nicegui import ui, app  # handle UIUX
from fastapi.staticfiles import StaticFiles  # integrate fastapi static folder and gfx folder
from webmind.ollama_handler import OllamaHandler  # Import OllamaHandler for modular Ollama interactions
from webmind.html_head import add_head_html  # handler for the html head imports and meta tags
from webmind.navigation import Navigation, SideNav  # Unified navigation system and drawer
from automind.openmind import OpenMind  # Importing OpenMind class from openmind.py
import concurrent.futures
import ujson as json
import asyncio
import aiohttp
import logging
import signal
import sys
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Serve static graphic files and easystyle.css from the 'gfx' directory
app.mount('/gfx', StaticFiles(directory='gfx'), name='gfx')

openmind = OpenMind()  # initialize OpenMind instance
ollama_model = OllamaHandler()  # initialize OllamaHandler instance

# Toggle for autonomous reasoning
async def toggle_autonomous_reasoning(value):
    openmind.autonomous_reasoning = value
    if value:
        # Start reasoning loop task if not already running
        if not openmind.reasoning_task or openmind.reasoning_task.done():
            openmind.reasoning_task = openmind._create_task(openmind.reasoning_loop())
            logging.info("Autonomous reasoning enabled")
    else:
        # Stop reasoning loop task
        if openmind.reasoning_task and not openmind.reasoning_task.done():
            openmind.reasoning_task.cancel()
            try:
                await openmind.reasoning_task
            except asyncio.CancelledError:
                pass
            openmind.reasoning_task = None
            logging.info("Autonomous reasoning disabled")

@ui.page('/')
def main():
    global executor, message_container, log, keys_container, text, selected_api
    executor = concurrent.futures.ThreadPoolExecutor()  # initialize thread pool executor to manage and execute multiple tasks concurrently
    selected_api = None  # Variable to store selected API

    async def send() -> None:
        question = text.value  # get value from input field
        text.value = ''  # clear input field for openmind
        if not question:
            ui.notify('Please enter a prompt.', type='warning')
            logging.warning("No prompt entered. Please enter a prompt.")
            return

        if selected_api:
            await generate_api_response(question)

    async def generate_api_response(prompt) -> None:
        await openmind.send_message(prompt)  # send the question to OpenMind
        await openmind.internal_queue.put(prompt)  # add question to the internal queue for processing

    def select_api(service):
        global selected_api
        selected_api = service
        openmind.select_model(service)
        ui.notify(f'Selected API: {service}', type='info')
        logging.info(f'Selected API: {service}')

    # configure HTML head content from html_head.py external module in the webmind folder
    add_head_html(ui)
    dark_mode = ui.dark_mode()
    drawer = SideNav(current_page='chat').create_drawer()
    # Create reactive reference for autonomous reasoning state
    autonomous_state_ref = {'value': openmind.autonomous_reasoning}

    async def toggle_dark_mode():
        dark_mode.value = not dark_mode.value  # toggle dark mode value
        # persist preference
        await ui.run_javascript(
            f'localStorage.setItem("theme", "{"dark" if dark_mode.value else "light"}")'
        )
        # nothing else needed; CSS responds to body--dark

    async def init_theme_from_storage():
        stored = await ui.run_javascript('localStorage.getItem("theme")')
        if stored == 'dark':
            dark_mode.value = True
        elif stored == 'light':
            dark_mode.value = False

    # Wrapper to sync reactive state with openmind.autonomous_reasoning
    async def autonomous_change_handler(value):
        autonomous_state_ref['value'] = value
        openmind.autonomous_reasoning = value
        await toggle_autonomous_reasoning(value)

    # Create unified navigation header
    nav = Navigation(current_page='chat', dark_mode=dark_mode, drawer=drawer)
    nav.create_header(
        autonomous_callback=autonomous_change_handler,
        dark_mode_callback=toggle_dark_mode,
        autonomous_state=autonomous_state_ref['value']
    )

    # initialize theme once UI is ready
    ui.timer(0.1, init_theme_from_storage, once=True)

    # Model selector FAB (floating action button)
    with ui.page_sticky(position='top-left', x_offset=20, y_offset=80):
        with ui.button(icon='psychology').props('fab color=primary'):
            with ui.menu().props('anchor="bottom left"'):
                ui.menu_item('Model Selection').props('disable')
                ui.separator()
                keys_list = openmind.api_manager.api_keys.items()
                for service, key in keys_list:
                    def create_model_menu(service):
                        with ui.menu_item(clickable=True, on_click=lambda s=service: select_api(s)):
                            ui.label(service.capitalize()).classes('font-bold')
                    create_model_menu(service)

    # (tabs removed; logs moved to /logs, API keys moved to /settings)

    # Chat display area
    with ui.column().classes('page-content'):
        message_container = ui.column().classes('chat-container')
        openmind.message_container = message_container

    # Terminal-style prompt footer
    with ui.footer().classes('footer terminal-footer'):
        with ui.row().classes('w-full items-center gap-2'):
            ui.label('>').classes('terminal-prefix')
            text = ui.textarea(placeholder='Type your prompt, press Enter to send...').props('rows=1 autogrow').classes('prompt-input')
            ui.button(icon='send', on_click=send).classes('send-btn').props('flat round')
        ui.markdown('[easyAGI](https://rage.pythai.net)').classes('footer-link')

    # Start main loop to process user input (reasoning loop started separately if autonomous mode enabled)
    # Note: main_loop processes user input queue; reasoning_loop handles autonomous reasoning
    openmind._create_task(openmind.main_loop())

logging.debug("starting easyAGI")

# Entry point - only run when launched directly (not when imported)
# Note: lmagi_gui.py is the preferred entrypoint for normal use
if __name__ in {"__main__", "__mp_main__"}:
    # Check if running in headless mode (launched by GUI)
    headless = os.environ.get('LMAGI_HEADLESS', '0') == '1'
    try:
        ui.run(title='easyAGI', port=8080, show=not headless)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        sys.exit(0)

@ui.page('/ollama')
def ollama_page():
    global ollama_models, selected_model, response_output_ollama, ollama_menu_container
    ollama_models = []  # List to store Ollama model references
    selected_model = None  # Variable to store selected Ollama model
    ollama_menu_container = None  # Container for model menu items

    async def send() -> None:
        question = text.value  # get value from input field
        text.value = ''  # clear input field for openmind
        if not question:
            ui.notify('Please enter a prompt.', type='warning')
            logging.warning("No prompt entered. Please enter a prompt.")
            return

        if selected_model:
            await generate_ollama_response(question)

    async def generate_ollama_response(prompt) -> None:
        if not selected_model:
            ui.notify('Please select an Ollama model first.', type='warning')
            logging.warning("No model selected. Please select an Ollama model first.")
            return

        logging.debug(f"Generating response using model: {selected_model} with prompt: {prompt}")
        try:
            response_content = ""
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": selected_model,
                    "prompt": prompt,
                    "stream": True
                }
                logging.debug(f"Sending payload: {payload}")

                async with session.post(ollama_model.api_url + "/generate", json=payload) as response:
                    async for line in response.content:
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if "response" in data:
                                response_content += data["response"]
                                # Update with markdown rendering for streaming
                                response_output_ollama.set_content(response_content)
                                logging.debug(f"Received response chunk: {data['response']}")
                            elif "error" in data:
                                logging.error(f"Error in response: {data['error']}")
                                ui.notify(f"Error: {data['error']}", type='negative')

                logging.info("Generated response successfully.")
                logging.debug(f"Complete response content: {response_content}")

                await openmind.send_message(prompt)  # Send the prompt to OpenMind
                await openmind.internal_queue.put(prompt)  # Add prompt to the internal queue for processing
                logging.debug(f"Sent prompt to OpenMind: {prompt}")

                await openmind.send_message(response_content)  # Send the response to OpenMind
                await openmind.internal_queue.put(response_content)  # Add response to the internal queue for processing
                logging.debug(f"Sent response to OpenMind: {response_content}")

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            ui.notify(f"Error generating response: {e}", type='negative')

    def list_ollama_models():
        try:
            logging.debug("Running 'ollama list' command.")
            result = ollama_model.list_models()
            if result:
                logging.debug(f"'ollama list' output:\n{result}")
                global ollama_models
                ollama_models = result  # Keep the full result list
                if ollama_models and len(ollama_models) > 1:
                    ui.notify('Models listed successfully.', type='positive')
                    logging.info("Models listed successfully.")
                    update_ollama_menu()
                else:
                    ui.notify('No models found.', type='negative')
                    logging.warning("No models found.")
            else:
                logging.error("Error listing models.")
                ui.notify('Error listing models.', type='negative')
        except Exception as e:
            logging.error(f"Exception during model listing: {e}")
            ui.notify('Exception occurred while listing models.', type='negative')

    # configure HTML head content from html_head.py external module in the webmind folder
    add_head_html(ui)
    dark_mode = ui.dark_mode()
    drawer = SideNav(current_page='ollama').create_drawer()
    # Create reactive reference for autonomous reasoning state
    autonomous_state_ref = {'value': openmind.autonomous_reasoning}

    async def toggle_dark_mode():
        dark_mode.value = not dark_mode.value  # toggle dark mode value
        await ui.run_javascript(
            f'localStorage.setItem("theme", "{"dark" if dark_mode.value else "light"}")'
        )

    async def init_theme_from_storage():
        stored = await ui.run_javascript('localStorage.getItem("theme")')
        if stored == 'dark':
            dark_mode.value = True
        elif stored == 'light':
            dark_mode.value = False

    # Wrapper to sync reactive state with openmind.autonomous_reasoning
    async def autonomous_change_handler(value):
        autonomous_state_ref['value'] = value
        openmind.autonomous_reasoning = value
        await toggle_autonomous_reasoning(value)

    # Create unified navigation header
    nav = Navigation(current_page='ollama', dark_mode=dark_mode, drawer=drawer)
    nav.create_header(
        autonomous_callback=autonomous_change_handler,
        dark_mode_callback=toggle_dark_mode,
        autonomous_state=autonomous_state_ref['value']
    )

    ui.timer(0.1, init_theme_from_storage, once=True)

    def select_ollama_model(model_name):
        """Handle model selection from FAB"""
        global selected_model
        selected_model = model_name
        ollama_model.select_model(model_name)
        ui.notify(f'Selected model: {model_name}', type='positive')
        logging.info(f"User selected Ollama model: {model_name}")

    # Ollama model selector FAB - positioned same as main page
    with ui.page_sticky(position='top-left', x_offset=20, y_offset=80):
        with ui.button(icon='smart_toy').props('fab color=secondary'):
            with ui.menu().props('anchor="bottom left"') as ollama_menu:
                ui.menu_item('Ollama Models').props('disable')
                ui.separator()
                ollama_menu_items_container = ui.column()

    def update_ollama_menu():
        """Populate the menu with available Ollama models"""
        ollama_menu_items_container.clear()
        with ollama_menu_items_container:
            if ollama_models and len(ollama_models) > 1:
                for model_line in ollama_models[1:]:  # Skip header line
                    model_name = model_line.split()[0]
                    ui.menu_item(model_name, on_click=lambda m=model_name: select_ollama_model(m))
            else:
                ui.menu_item('No models found').props('disable')

    # Populate models after menu is created
    list_ollama_models()
    update_ollama_menu()

    # terminal-style footer for ollama too
    with ui.footer().classes('footer terminal-footer'):
        with ui.row().classes('w-full items-center gap-2'):
            ui.label('>').classes('terminal-prefix')
            text = ui.textarea(placeholder='Type your prompt, press Enter to send...').props('rows=1 autogrow').classes('prompt-input')
            ui.button(icon='send', on_click=send).classes('send-btn').props('flat round')
        ui.markdown('[easyAGI](https://rage.pythai.net)').classes('footer-link')

    response_output_ollama = ui.markdown().classes('text-lg mt-4')


@ui.page('/settings')
def settings_page():
    """Application settings: appearance and API keys"""
    add_head_html(ui)
    dark_mode = ui.dark_mode()
    drawer = SideNav(current_page='settings').create_drawer()

    async def init_theme_from_storage():
        stored = await ui.run_javascript('localStorage.getItem("theme")')
        if stored == 'dark':
            dark_mode.value = True
        elif stored == 'light':
            dark_mode.value = False

    async def on_theme_switch(e):
        # set according to switch and persist
        dark_mode.value = bool(e.value)
        await ui.run_javascript(
            f'localStorage.setItem("theme", "{"dark" if dark_mode.value else "light"}")'
        )

    # Sync autonomous toggle with backend
    autonomous_state_ref = {'value': openmind.autonomous_reasoning}

    async def autonomous_change_handler(value):
        autonomous_state_ref['value'] = value
        openmind.autonomous_reasoning = value
        await toggle_autonomous_reasoning(value)

    nav = Navigation(current_page='settings', dark_mode=dark_mode, drawer=drawer)
    nav.create_header(
        autonomous_callback=autonomous_change_handler,
        dark_mode_callback=lambda: None,  # theme is managed by the switch on this page
        autonomous_state=autonomous_state_ref['value']
    )

    ui.timer(0.1, init_theme_from_storage, once=True)

    with ui.column().classes('w-full max-w-screen-md mx-auto gap-4 p-4'):
        ui.label('Settings').classes('text-2xl font-bold')

        # Appearance Card
        with ui.card().classes('w-full'):
            ui.label('Appearance').classes('text-lg font-semibold')
            ui.separator()
            ui.switch('Dark Mode', value=dark_mode.value, on_change=on_theme_switch)

        # API Keys Card
        with ui.card().classes('w-full'):
            ui.label('API Keys').classes('text-lg font-semibold')
            ui.separator()
            with ui.row().classes('items-center w-full gap-2'):
                openmind.service_input = ui.input('Service (e.g., together, openai, groq)').classes('flex-1 input')
                openmind.key_input = ui.input('API Key').classes('flex-1 input')
            with ui.row().classes('gap-2'):
                ui.button('Add API Key', on_click=openmind.add_api_key, icon='add').classes('api-action')
                ui.button('List API Keys', on_click=openmind.list_api_keys, icon='list').classes('api-action')
            keys_container = ui.column().classes('w-full')
            openmind.keys_container = keys_container

@ui.page('/logs')
def logs_page():
    add_head_html(ui)
    dark_mode = ui.dark_mode()
    drawer = SideNav(current_page='logs').create_drawer()

    async def init_theme_from_storage():
        stored = await ui.run_javascript('localStorage.getItem("theme")')
        if stored == 'dark':
            dark_mode.value = True
        elif stored == 'light':
            dark_mode.value = False

    # simple header
    nav = Navigation(current_page='logs', dark_mode=dark_mode, drawer=drawer)
    nav.create_header()

    ui.timer(0.1, init_theme_from_storage, once=True)

    log_files = {
        "Premises Log": "./memory/logs/premises.json",
        "Not Premise Log": "./memory/logs/notpremise.json",
        "Truth Tables Log": "./memory/truth/logs.txt",
        "Thoughts Log": "./memory/logs/thoughts.json",
        "Conclusions Log": "./memory/logs/conclusions.txt",
        "Decisions Log": "./memory/logs/truth.json",
    }

    def view_log(file_path):
        log_content = openmind.read_log_file(file_path)
        log_container.clear()
        with log_container:
            ui.markdown(log_content).classes('w-full')

    with ui.row().classes('w-full gap-2 q-pa-md'):
        with ui.column().classes('w-1/4'):
            ui.label('Logs').classes('text-lg font-bold')
            for log_name, log_path in log_files.items():
                ui.button(log_name, on_click=lambda p=log_path: view_log(p)).classes('logbuttons')

        with ui.column().classes('w-3/4'):
            ui.label('Log Viewer').classes('text-lg font-bold')
            log_container = ui.column().classes('w-full')

def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT and SIGTERM"""
    logging.info(f"Received signal {sig}. Shutting down gracefully...")
    # Clean up any running tasks
    if hasattr(openmind, 'cleanup'):
        # Note: cleanup is async, but signal handler can't be async
        # In production, consider using asyncio.run() or proper async shutdown
        logging.info("Cleaning up tasks...")
    sys.exit(0)
