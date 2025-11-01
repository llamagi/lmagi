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
from webmind.settings import SettingsManager  # Settings persistence system
from automind.openmind import OpenMind  # Importing OpenMind class from openmind.py
import concurrent.futures
import ujson as json
import asyncio
import aiohttp
import httpx
import logging
import signal
import sys
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Serve static graphic files and easystyle.css from the 'gfx' directory
app.mount('/gfx', StaticFiles(directory='gfx'), name='gfx')

# Initialize settings manager
settings_manager = SettingsManager()

openmind = OpenMind()  # initialize OpenMind instance
# Initialize OllamaHandler with settings
try:
    ollama_url = settings_manager.get('ollama_base_url', 'http://localhost:11434')
    ollama_model = OllamaHandler(base_url=ollama_url)
except:
    ollama_model = OllamaHandler()  # Fallback to default

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
    global executor, message_container, log, keys_container, text, selected_api, conversation_history
    executor = concurrent.futures.ThreadPoolExecutor()  # initialize thread pool executor to manage and execute multiple tasks concurrently
    selected_api = None  # Variable to store selected API
    conversation_history = []  # Store conversation history for context

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
        # If Ollama Cloud is selected, handle streaming response like Ollama page
        if selected_api == 'ollama_cloud':
            # Display user query in conversation-style format
            if message_container.client.connected:
                with message_container:
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    with ui.column().classes('message-block'):
                        ui.label('query').classes('text-lg font-semibold')
                        ui.html('<hr style="margin: 0.5rem 0; border: none; border-top: 1px solid var(--accent-blue);">')
                        ui.label(timestamp).classes('text-xs text-gray-500')
                        ui.markdown(prompt).classes('mt-2')
                
                # Auto-scroll after adding query (simple like Ollama page)
                try:
                    await ui.run_javascript('const c=document.querySelector(".chat-container"); if(c){c.scrollTop=c.scrollHeight;}')
                except Exception:
                    pass
            
            # Generate streaming response
            await generate_ollama_cloud_streaming_response(prompt)
        else:
            # For other APIs, use AGI
            await openmind.send_message(prompt)  # send the question to OpenMind
            await openmind.internal_queue.put(prompt)  # add question to the internal queue for processing
    
    async def generate_ollama_cloud_streaming_response(prompt) -> None:
        """Generate streaming response for Ollama Cloud with conversation history"""
        global conversation_history
        
        try:
            from webmind.settings import SettingsManager
            settings = SettingsManager()
            base_url = settings.get('ollama_cloud_base_url', 'https://ollama.com')
            from webmind.chatter import OllamaCloudModel
            ollama_cloud_key = openmind.api_manager.get_api_key('ollama_cloud')
            
            if not ollama_cloud_key:
                ui.notify('Ollama Cloud API key not found', type='warning')
                return
            
            # Ensure AGI is initialized with ollama_cloud
            if not hasattr(openmind, 'agi_instance') or not openmind.agi_instance:
                await openmind.select_model('ollama_cloud')
                # Wait a moment for initialization
                import asyncio
                await asyncio.sleep(1.0)  # Increased wait time
            
            # Get the selected model - wait and retry if not set yet
            model_name = None
            max_retries = 5
            for attempt in range(max_retries):
                if hasattr(openmind, 'agi_instance') and openmind.agi_instance:
                    if hasattr(openmind.agi_instance, 'agi') and hasattr(openmind.agi_instance.agi, 'chatter'):
                        model_name = openmind.agi_instance.agi.chatter.get_current_model()
                        if model_name:
                            logging.info(f"Found selected model: {model_name}")
                            break
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                    logging.debug(f"Waiting for model selection (attempt {attempt + 1}/{max_retries})")
            
            if not model_name:
                ui.notify('Please select an Ollama Cloud model first', type='warning')
                logging.warning("No Ollama Cloud model selected after retries")
                return
            
            cloud_model = OllamaCloudModel(ollama_cloud_key, base_url=base_url)
            cloud_model.set_model(model_name)
            
            logging.info(f"Generating streaming response using Ollama Cloud model: {model_name}")
            
            import aiohttp
            import json
            import asyncio
            
            # Display response header before streaming (conversation-style)
            response_markdown = None
            if message_container.client.connected:
                with message_container:
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    response_container = ui.column().classes('message-block')
                    with response_container:
                        ui.label('ezAGI').classes('text-lg font-semibold')
                        ui.html('<hr style="margin: 0.5rem 0; border: none; border-top: 1px solid var(--accent-blue);">')
                        ui.label(timestamp).classes('text-xs text-gray-500')
                        # Create markdown element for streaming content
                        response_markdown = ui.markdown().classes('mt-2')
            
            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": prompt})
            
            # Smart context window management:
            # - Keep last 20 messages (10 exchanges)
            # - Or limit by approximate token count (~4000 tokens = ~20 messages)
            # - This prevents context window overflow while maintaining conversation flow
            max_messages = 20
            if len(conversation_history) > max_messages:
                # Keep system message if present, then last N user/assistant pairs
                conversation_history = conversation_history[-max_messages:]
            
            logging.debug(f"Conversation context: {len(conversation_history)} messages")
            
            response_content = ""
            headers = {
                "Authorization": f"Bearer {ollama_cloud_key}",
                "Content-Type": "application/json"
            }
            
            # Use /api/chat endpoint with messages array for conversation context
            # According to docs: https://docs.ollama.com/cloud
            payload = {
                "model": model_name,
                "messages": conversation_history,  # Include full conversation history
                "stream": True
            }
            
            logging.debug(f"Making request to {cloud_model.api_url}/chat with model: {model_name}, {len(conversation_history)} messages in context")
            
            # Get timeout from settings
            from webmind.settings import SettingsManager
            timeout_settings = SettingsManager()
            ollama_cloud_timeout = timeout_settings.get('ollama_cloud_timeout', 300)
            
            # Batch updates for smoother streaming (update every 50ms instead of every chunk)
            last_update_time = asyncio.get_event_loop().time()
            update_interval = 0.05  # 50ms between UI updates
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{cloud_model.api_url}/chat",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=ollama_cloud_timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"Ollama Cloud API error: HTTP {response.status} - {error_text}")
                        logging.error(f"Request URL: {cloud_model.api_url}/chat")
                        logging.error(f"Request payload: {payload}")
                        if response_markdown and message_container.client.connected:
                            response_markdown.set_content(f"**Error:** Ollama Cloud API returned HTTP {response.status}\n\n{error_text}")
                        ui.notify(f'Ollama Cloud API error: HTTP {response.status}', type='negative')
                        return
                    
                    async for line in response.content:
                        if line:
                            try:
                                line_str = line.decode('utf-8').strip()
                                if line_str:
                                    data = json.loads(line_str)
                                    
                                    # Check for errors first
                                    if "error" in data:
                                        logging.error(f"Error in response: {data['error']}")
                                        error_msg = f"**Error:** {data['error']}"
                                        if response_markdown and message_container.client.connected:
                                            response_markdown.set_content(error_msg)
                                        ui.notify(f"Error: {data['error']}", type='negative')
                                        continue
                                    
                                    # Handle response content
                                    if "message" in data and "content" in data["message"]:
                                        # Chat endpoint format
                                        response_content += data["message"]["content"]
                                    elif "response" in data:
                                        # Generate endpoint format (fallback)
                                        response_content += data["response"]
                                    else:
                                        continue
                                    
                                    # Batch UI updates for smoother streaming (update every 50ms)
                                    current_time = asyncio.get_event_loop().time()
                                    should_update_ui = current_time - last_update_time >= update_interval
                                    
                                    if should_update_ui:
                                        if response_markdown and message_container.client.connected:
                                            response_markdown.set_content(response_content)
                                        last_update_time = current_time
                                    
                                    # Auto-scroll on every chunk when content changes (like Ollama page)
                                    try:
                                        await ui.run_javascript('const c=document.querySelector(".chat-container"); if(c){c.scrollTop=c.scrollHeight;}')
                                    except Exception:
                                        pass
                                    
                                    logging.debug(f"Received response chunk: {len(response_content)} chars")
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logging.error(f"Error processing stream chunk: {e}")
                    
                    # Final update with complete response
                    if response_markdown and message_container.client.connected:
                        response_markdown.set_content(response_content)
                    
                    # Add assistant response to conversation history
                    conversation_history.append({"role": "assistant", "content": response_content})
                    
                    logging.info("Generated streaming response successfully.")
                    logging.debug(f"Complete response content: {response_content[:100]}... ({len(response_content)} chars)")
                    logging.info(f"Conversation history now has {len(conversation_history)} messages")
                    
                    # Auto-scroll to bottom after streaming completes (simple like Ollama page)
                    try:
                        await ui.run_javascript('const c=document.querySelector(".chat-container"); if(c){c.scrollTop=c.scrollHeight;}')
                    except Exception:
                        pass
                    
        except Exception as e:
            logging.error(f"Error generating Ollama Cloud streaming response: {e}", exc_info=True)
            error_msg = f"**Error:** {str(e)}"
            if message_container.client.connected:
                with message_container:
                    ui.markdown(error_msg).classes('text-lg mt-4')
            ui.notify(f'Error generating response: {str(e)}', type='negative')

    def select_api(service):
        global selected_api
        selected_api = service
        # Schedule async model selection
        try:
            openmind._create_task(openmind.select_model(service))
        except Exception:
            pass
        ui.notify(f'Selected API: {service}', type='info')
        logging.info(f'Selected API: {service}')

    # Auto-select first available API if none chosen
    def auto_select_api():
        services_priority = ['openai', 'groq', 'together', 'ai71', 'ollama_cloud']
        if not selected_api:
            for svc in services_priority:
                key = openmind.api_manager.get_api_key(svc)
                if key:
                    select_api(svc)
                    break

    # configure HTML head content from html_head.py; seed localStorage from server settings to avoid flash
    add_head_html(ui, settings_manager.sync_to_localStorage())
    dark_mode = ui.dark_mode()
    # Ensure dark mode reflects persisted settings before building header/UI
    try:
        dark_mode.value = settings_manager.get('dark_mode', True)
    except Exception:
        dark_mode.value = True
    
    # CRITICAL FIX #5: Restore autonomous state BEFORE creating header
    # Use SettingsManager to restore state synchronously
    autonomous_state_ref = {'value': settings_manager.get('autonomous_reasoning', False)}
    openmind.autonomous_reasoning = autonomous_state_ref['value']
    
    # Also sync from localStorage if available (for browser-only persistence)
    async def restore_autonomous_state_before_header():
        try:
            stored = await ui.run_javascript('localStorage.getItem("autonomous-reasoning") === "true" || window.restoredAutonomousState === true')
            if stored:
                autonomous_state_ref['value'] = True
                openmind.autonomous_reasoning = True
                # Sync to SettingsManager
                settings_manager.set('autonomous_reasoning', True)
        except:
            pass
    
    # Use immediate timer to restore before header creation
    ui.timer(0.01, restore_autonomous_state_before_header, once=True)
    
    drawer = SideNav(current_page='chat').create_drawer()

    async def toggle_dark_mode():
        dark_mode.value = not dark_mode.value  # toggle dark mode value
        # Persist to both localStorage and SettingsManager
        await ui.run_javascript(f'''
            localStorage.setItem('theme', '{'dark' if dark_mode.value else 'light'}');
            const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
            if (window.applyTheme) {{
                window.applyTheme(savedTheme, {str(dark_mode.value).lower()});
            }} else {{
                // Fallback: manually sync body--dark class
                if (document.body) {{
                    if ({str(dark_mode.value).lower()}) {{
                        document.body.classList.add('body--dark');
                    }} else {{
                        document.body.classList.remove('body--dark');
                    }}
                }}
            }}
        ''')
        # Sync to SettingsManager
        settings_manager.set('dark_mode', dark_mode.value)

    async def init_theme_from_storage():
        # Load from SettingsManager first (server-side persistence)
        theme_name = settings_manager.get('theme', 'everforest')
        dark_mode_setting = settings_manager.get('dark_mode', True)
        
        # Sync to localStorage (browser persistence)
        await ui.run_javascript(f'''
            localStorage.setItem('ui-theme', '{theme_name}');
            localStorage.setItem('theme', '{'dark' if dark_mode_setting else 'light'}');
        ''')
        
        # Set dark_mode value
        dark_mode.value = dark_mode_setting
        
        # CRITICAL FIX #2: Sync with unified theme system after restoring
        await ui.run_javascript(f'''
            const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
            if (window.applyTheme) {{
                window.applyTheme(savedTheme, {str(dark_mode.value).lower()});
            }}
        ''')

    # Wrapper to sync reactive state with openmind.autonomous_reasoning
    async def autonomous_change_handler(e):
        # Extract value from event object (NiceGUI's ValueChangeEventArguments)
        value = e.value if hasattr(e, 'value') else e
        autonomous_state_ref['value'] = value
        openmind.autonomous_reasoning = value
        # Persist to both localStorage and SettingsManager
        await ui.run_javascript(f'localStorage.setItem("autonomous-reasoning", "{str(value).lower()}")')
        settings_manager.set('autonomous_reasoning', value)
        await toggle_autonomous_reasoning(value)

    # Create unified navigation header
    nav = Navigation(current_page='chat', dark_mode=dark_mode, drawer=drawer)
    nav.create_header(
        autonomous_callback=autonomous_change_handler,
        dark_mode_callback=toggle_dark_mode,
        autonomous_state=autonomous_state_ref['value']
    )

    # Initialize theme and autonomous state from storage
    async def init_settings_from_storage():
        await init_theme_from_storage()
        # Restore autonomous reasoning state (already set from SettingsManager, but sync from localStorage too)
        stored_autonomous = await ui.run_javascript('localStorage.getItem("autonomous-reasoning")')
        if stored_autonomous == 'true':
            autonomous_state_ref['value'] = True
            openmind.autonomous_reasoning = True
            settings_manager.set('autonomous_reasoning', True)
            await toggle_autonomous_reasoning(True)
            # Update switch if it exists
            await ui.run_javascript('''
                const switches = document.querySelectorAll("input[type=\'checkbox\']");
                switches.forEach(sw => {
                    if (sw.closest(".q-switch") && sw.parentElement.textContent.includes("Autonomous")) {
                        sw.checked = true;
                        sw.dispatchEvent(new Event("change", { bubbles: true }));
                    }
                });
            ''')
    
    ui.timer(0.1, init_settings_from_storage, once=True)
    # Attempt auto-select after initial settings load
    ui.timer(0.2, auto_select_api, once=True)
    # Ensure AGI is initialized from stored keys on page load
    try:
        openmind._create_task(openmind.initialize_agi())
    except Exception:
        pass

    # Model selector moved to footer menu (removed FAB from content area)

    # (tabs removed; logs moved to /logs, API keys moved to /settings)

    # Chat display area - conversation-style with scrollable container (like Ollama page)
    with ui.column().classes('page-content'):
        with ui.column().classes('chat-container'):
            message_container = ui.column()
            openmind.message_container = message_container

    # Terminal-style prompt footer
    with ui.footer().classes('footer terminal-footer'):
        with ui.row().classes('w-full items-center gap-2'):
            ui.label('>').classes('terminal-prefix')
            text = ui.textarea(placeholder='Type your prompt (Enter=send, Shift+Enter=newline)').props('rows=1 autogrow').classes('prompt-input')
            ui.button(icon='send', on_click=send).classes('send-btn').props('flat round')
            # API model selector button with dropdown menu (same pattern as Ollama page)
            with ui.button(icon='psychology').props('round flat').classes('q-ml-sm'):
                with ui.menu().props('anchor="top right"') as api_menu:
                    ui.menu_item('Model Selection').props('disable')
                    ui.separator()
                    api_menu_items_container = ui.column()
            
            def select_ollama_cloud_model(model_name):
                """Select a specific Ollama Cloud model"""
                global selected_api
                selected_api = 'ollama_cloud'
                logging.info(f"Selecting Ollama Cloud model: {model_name}")
                try:
                    from webmind.settings import SettingsManager
                    settings = SettingsManager()
                    base_url = settings.get('ollama_cloud_base_url', 'https://ollama.com')
                    from webmind.chatter import OllamaCloudModel
                    ollama_cloud_key = openmind.api_manager.get_api_key('ollama_cloud')
                    if ollama_cloud_key:
                        # First, ensure AGI is initialized with ollama_cloud
                        async def ensure_model_selected():
                            # Initialize AGI if needed
                            if not hasattr(openmind, 'agi_instance') or not openmind.agi_instance:
                                logging.info("Initializing AGI with ollama_cloud")
                                await openmind.select_model('ollama_cloud')
                                await asyncio.sleep(0.5)  # Wait for initialization
                            
                            # Now set the model
                            if hasattr(openmind, 'agi_instance') and openmind.agi_instance:
                                if hasattr(openmind.agi_instance, 'agi') and hasattr(openmind.agi_instance.agi, 'chatter'):
                                    if hasattr(openmind.agi_instance.agi.chatter, 'set_model'):
                                        openmind.agi_instance.agi.chatter.set_model(model_name)
                                        logging.info(f'Successfully set Ollama Cloud model to: {model_name}')
                                        # Verify it was set
                                        current = openmind.agi_instance.agi.chatter.get_current_model()
                                        logging.info(f'Verified current model: {current}')
                                    else:
                                        logging.error("chatter.set_model method not found")
                                        # Reinitialize
                                        await openmind.select_model('ollama_cloud')
                                        await asyncio.sleep(0.5)
                                        if hasattr(openmind.agi_instance, 'agi') and hasattr(openmind.agi_instance.agi, 'chatter'):
                                            if hasattr(openmind.agi_instance.agi.chatter, 'set_model'):
                                                openmind.agi_instance.agi.chatter.set_model(model_name)
                                                logging.info(f'Set model after reinit: {model_name}')
                                else:
                                    logging.error("chatter not found, reinitializing")
                                    await openmind.select_model('ollama_cloud')
                                    await asyncio.sleep(0.5)
                                    if hasattr(openmind.agi_instance, 'agi') and hasattr(openmind.agi_instance.agi, 'chatter'):
                                        if hasattr(openmind.agi_instance.agi.chatter, 'set_model'):
                                            openmind.agi_instance.agi.chatter.set_model(model_name)
                                            logging.info(f'Set model after reinit: {model_name}')
                            else:
                                logging.error("AGI instance not available")
                        
                        import asyncio
                        openmind._create_task(ensure_model_selected())
                        
                        ui.notify(f'Selected Ollama Cloud model: {model_name}', type='info')
                        logging.info(f'Selected Ollama Cloud model: {model_name}')
                except Exception as e:
                    logging.error(f"Error selecting Ollama Cloud model: {e}", exc_info=True)
                    ui.notify(f'Error selecting model: {str(e)}', type='negative')
            
            def update_api_menu():
                """Populate the menu with available providers and their models (same pattern as Ollama page)"""
                api_menu_items_container.clear()
                with api_menu_items_container:
                    keys_list = list(openmind.api_manager.api_keys.items())
                    
                    # Handle Ollama Cloud separately to show models
                    ollama_cloud_key = openmind.api_manager.get_api_key('ollama_cloud')
                    if ollama_cloud_key:
                        try:
                            from webmind.settings import SettingsManager
                            settings = SettingsManager()
                            base_url = settings.get('ollama_cloud_base_url', 'https://ollama.com')
                            logging.info(f"Loading Ollama Cloud models with base_url: {base_url}")
                            if not base_url or not base_url.strip():
                                logging.warning("Ollama Cloud base_url is empty, using default")
                                base_url = 'https://ollama.com'
                            from webmind.chatter import OllamaCloudModel
                            cloud_model = OllamaCloudModel(ollama_cloud_key, base_url=base_url)
                            models = cloud_model.list_models()
                            
                            ui.menu_item('Ollama Cloud').props('disable')
                            if models:
                                logging.info(f"Displaying {len(models)} Ollama Cloud models in menu")
                                for model_name in models:
                                    # Use same pattern as Ollama page - lambda with default parameter
                                    ui.menu_item(model_name, on_click=lambda m=model_name: select_ollama_cloud_model(m))
                            else:
                                logging.warning("No Ollama Cloud models found")
                                ui.menu_item('No models available').props('disable')
                            ui.separator()
                        except Exception as e:
                            logging.error(f"Error loading Ollama Cloud models: {e}", exc_info=True)
                            ui.menu_item('Ollama Cloud (Error)').props('disable')
                            ui.separator()
                    
                    # Handle other providers
                    for service, key in keys_list:
                        if service == 'ollama_cloud':
                            continue  # Already handled above
                        # Use same pattern - lambda with default parameter
                        ui.menu_item(service.capitalize(), on_click=lambda s=service: select_api(s))
            
            # Consolidated refresh that runs after UI is built (same pattern as Ollama page)
            def refresh_api_menu():
                try:
                    update_api_menu()
                    # Auto-select first Ollama Cloud model if ollama_cloud is selected and no model is set
                    if selected_api == 'ollama_cloud':
                        ollama_cloud_key = openmind.api_manager.get_api_key('ollama_cloud')
                        if ollama_cloud_key:
                            try:
                                from webmind.settings import SettingsManager
                                settings = SettingsManager()
                                base_url = settings.get('ollama_cloud_base_url', 'https://ollama.com')
                                from webmind.chatter import OllamaCloudModel
                                cloud_model = OllamaCloudModel(ollama_cloud_key, base_url=base_url)
                                models = cloud_model.list_models()
                                if models:
                                    # Check if a model is already selected
                                    current_model = None
                                    if hasattr(openmind, 'agi_instance') and openmind.agi_instance:
                                        if hasattr(openmind.agi_instance, 'agi') and hasattr(openmind.agi_instance.agi, 'chatter'):
                                            current_model = openmind.agi_instance.agi.chatter.get_current_model()
                                    # Auto-select first model if none selected
                                    if not current_model:
                                        select_ollama_cloud_model(models[0])
                                        logging.info(f"Auto-selected first Ollama Cloud model: {models[0]}")
                            except Exception as e:
                                logging.debug(f"Could not auto-select Ollama Cloud model: {e}")
                except Exception as e:
                    logging.error(f"Error refreshing API menu: {e}", exc_info=True)
            
            # Defer refresh to ensure menu container exists
            ui.timer(0.1, refresh_api_menu, once=True)
            
            # Store refresh function for later use
            openmind.refresh_api_menu = refresh_api_menu
        ui.markdown('[easyAGI](https://rage.pythai.net)').classes('footer-link')
    # Install Enter/Shift+Enter handler on prompt
    ui.timer(0.05, lambda: ui.run_javascript('''
        (function(){
          const area = document.querySelector('.prompt-input textarea');
          if (!area || area.__enterHandlerInstalled) return;
          area.__enterHandlerInstalled = true;
          area.addEventListener('keydown', function(ev){
            if (ev.key === 'Enter' && !ev.shiftKey) {
              ev.preventDefault();
              const btn = document.querySelector('.send-btn');
              if (btn) btn.click();
            }
          });
        })();
    '''), once=True)

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
                                # Auto scroll chat container as content streams
                                try:
                                    await ui.run_javascript('const c=document.querySelector(".chat-container"); if(c){c.scrollTop=c.scrollHeight;}')
                                except Exception:
                                    pass
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

    # configure HTML head content; seed localStorage from server settings to avoid flash
    add_head_html(ui, settings_manager.sync_to_localStorage())
    dark_mode = ui.dark_mode()
    # Ensure dark mode reflects persisted settings before building header/UI
    try:
        dark_mode.value = settings_manager.get('dark_mode', True)
    except Exception:
        dark_mode.value = True
    
    # CRITICAL FIX #5: Restore autonomous state BEFORE creating header
    # Use SettingsManager to restore state synchronously
    autonomous_state_ref = {'value': settings_manager.get('autonomous_reasoning', False)}
    openmind.autonomous_reasoning = autonomous_state_ref['value']
    
    # Also sync from localStorage if available (for browser-only persistence)
    async def restore_autonomous_state_before_header():
        try:
            stored = await ui.run_javascript('localStorage.getItem("autonomous-reasoning") === "true" || window.restoredAutonomousState === true')
            if stored:
                autonomous_state_ref['value'] = True
                openmind.autonomous_reasoning = True
                # Sync to SettingsManager
                settings_manager.set('autonomous_reasoning', True)
        except:
            pass
    
    ui.timer(0.01, restore_autonomous_state_before_header, once=True)
    
    drawer = SideNav(current_page='ollama').create_drawer()

    async def toggle_dark_mode():
        dark_mode.value = not dark_mode.value  # toggle dark mode value
        # Persist to both localStorage and SettingsManager
        await ui.run_javascript(f'''
            localStorage.setItem('theme', '{'dark' if dark_mode.value else 'light'}');
            const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
            if (window.applyTheme) {{
                window.applyTheme(savedTheme, {str(dark_mode.value).lower()});
            }}
        ''')
        # Sync to SettingsManager
        settings_manager.set('dark_mode', dark_mode.value)

    async def init_theme_from_storage():
        # Load from SettingsManager first (server-side persistence)
        theme_name = settings_manager.get('theme', 'everforest')
        dark_mode_setting = settings_manager.get('dark_mode', True)
        
        # Sync to localStorage (browser persistence)
        await ui.run_javascript(f'''
            localStorage.setItem('ui-theme', '{theme_name}');
            localStorage.setItem('theme', '{'dark' if dark_mode_setting else 'light'}');
        ''')
        
        # Set dark_mode value
        dark_mode.value = dark_mode_setting
        
        # CRITICAL FIX #2: Sync with unified theme system after restoring
        await ui.run_javascript(f'''
            const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
            if (window.applyTheme) {{
                window.applyTheme(savedTheme, {str(dark_mode.value).lower()});
            }}
        ''')

    # Wrapper to sync reactive state with openmind.autonomous_reasoning
    async def autonomous_change_handler(e):
        # Extract value from event object (NiceGUI's ValueChangeEventArguments)
        value = e.value if hasattr(e, 'value') else e
        autonomous_state_ref['value'] = value
        openmind.autonomous_reasoning = value
        # Persist to both localStorage and SettingsManager
        await ui.run_javascript(f'localStorage.setItem("autonomous-reasoning", "{str(value).lower()}")')
        settings_manager.set('autonomous_reasoning', value)
        await toggle_autonomous_reasoning(value)

    # Create unified navigation header
    nav = Navigation(current_page='ollama', dark_mode=dark_mode, drawer=drawer)
    nav.create_header(
        autonomous_callback=autonomous_change_handler,
        dark_mode_callback=toggle_dark_mode,
        autonomous_state=autonomous_state_ref['value']
    )

    # Initialize theme and autonomous state from storage
    async def init_settings_from_storage():
        await init_theme_from_storage()
        # Restore autonomous reasoning state (already set from SettingsManager, but sync from localStorage too)
        stored_autonomous = await ui.run_javascript('localStorage.getItem("autonomous-reasoning")')
        if stored_autonomous == 'true':
            autonomous_state_ref['value'] = True
            openmind.autonomous_reasoning = True
            settings_manager.set('autonomous_reasoning', True)
            await toggle_autonomous_reasoning(True)
            # Update switch if it exists
            await ui.run_javascript('''
                const switches = document.querySelectorAll("input[type=\'checkbox\']");
                switches.forEach(sw => {
                    if (sw.closest(".q-switch") && sw.parentElement.textContent.includes("Autonomous")) {
                        sw.checked = true;
                        sw.dispatchEvent(new Event("change", { bubbles: true }));
                    }
                });
            ''')
    
    ui.timer(0.1, init_settings_from_storage, once=True)

    def select_ollama_model(model_name):
        """Handle model selection from FAB"""
        global selected_model
        selected_model = model_name
        ollama_model.select_model(model_name)
        ui.notify(f'Selected model: {model_name}', type='positive')
        logging.info(f"User selected Ollama model: {model_name}")

    # Ollama model selector moved to footer menu (removed FAB)

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

    # Consolidated refresh that runs after UI is built
    def refresh_ollama_models():
        try:
            list_ollama_models()
            update_ollama_menu()
            if (not selected_model) and ollama_models and len(ollama_models) > 1:
                first_model = ollama_models[1].split()[0]
                select_ollama_model(first_model)
        except Exception:
            pass
    # Defer refresh to ensure menu container exists
    ui.timer(0.05, refresh_ollama_models, once=True)

    # Chat display area for Ollama responses
    with ui.column().classes('page-content'):
        with ui.column().classes('chat-container'):
            response_output_ollama = ui.markdown().classes('text-lg mt-4')

    # terminal-style footer for ollama too
    with ui.footer().classes('footer terminal-footer'):
        with ui.row().classes('w-full items-center gap-2'):
            ui.label('>').classes('terminal-prefix')
            text = ui.textarea(placeholder='Type your prompt (Enter=send, Shift+Enter=newline)').props('rows=1 autogrow').classes('prompt-input')
            ui.button(icon='send', on_click=send).classes('send-btn').props('flat round')
            # Ollama model selector button with dropdown
            with ui.button(icon='smart_toy').props('round flat').classes('q-ml-sm'):
                with ui.menu().props('anchor="top right"') as ollama_menu:
                    ui.menu_item('Ollama Models').props('disable')
                    ui.separator()
                    ollama_menu_items_container = ui.column()
        ui.markdown('[easyAGI](https://rage.pythai.net)').classes('footer-link')
    # Install Enter/Shift+Enter handler on prompt (ollama)
    ui.timer(0.05, lambda: ui.run_javascript('''
        (function(){
          const area = document.querySelector('.prompt-input textarea');
          if (!area || area.__enterHandlerInstalled) return;
          area.__enterHandlerInstalled = true;
          area.addEventListener('keydown', function(ev){
            if (ev.key === 'Enter' && !ev.shiftKey) {
              ev.preventDefault();
              const btn = document.querySelector('.send-btn');
              if (btn) btn.click();
            }
          });
        })();
    '''), once=True)


@ui.page('/settings')
def settings_page():
    """Application settings: appearance and API keys"""
    add_head_html(ui, settings_manager.sync_to_localStorage())
    dark_mode = ui.dark_mode()
    # Ensure dark mode reflects persisted settings before building header/UI
    try:
        dark_mode.value = settings_manager.get('dark_mode', True)
    except Exception:
        dark_mode.value = True
    
    # CRITICAL FIX #5: Restore autonomous state BEFORE creating header
    # Use SettingsManager to restore state synchronously
    autonomous_state_ref = {'value': settings_manager.get('autonomous_reasoning', False)}
    openmind.autonomous_reasoning = autonomous_state_ref['value']
    
    # Also sync from localStorage if available (for browser-only persistence)
    async def restore_autonomous_state_before_header():
        try:
            stored = await ui.run_javascript('localStorage.getItem("autonomous-reasoning") === "true" || window.restoredAutonomousState === true')
            if stored:
                autonomous_state_ref['value'] = True
                openmind.autonomous_reasoning = True
                # Sync to SettingsManager
                settings_manager.set('autonomous_reasoning', True)
        except:
            pass
    
    ui.timer(0.01, restore_autonomous_state_before_header, once=True)
    
    drawer = SideNav(current_page='settings').create_drawer()

    async def init_theme_from_storage():
        # Load from SettingsManager first (server-side persistence)
        theme_name = settings_manager.get('theme', 'everforest')
        dark_mode_setting = settings_manager.get('dark_mode', True)
        
        # Sync to localStorage (browser persistence)
        await ui.run_javascript(f'''
            localStorage.setItem('ui-theme', '{theme_name}');
            localStorage.setItem('theme', '{'dark' if dark_mode_setting else 'light'}');
        ''')
        
        # Set dark_mode value
        dark_mode.value = dark_mode_setting
        
        # CRITICAL FIX #2: Sync with unified theme system after restoring
        await ui.run_javascript(f'''
            const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
            if (window.applyTheme) {{
                window.applyTheme(savedTheme, {str(dark_mode.value).lower()});
            }}
        ''')

    async def on_theme_switch(e):
        # set according to switch and persist
        dark_mode.value = bool(e.value)
        # CRITICAL FIX #2: Sync dark mode with unified theme system
        await ui.run_javascript(f'''
            localStorage.setItem('theme', '{'dark' if dark_mode.value else 'light'}');
            const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
            if (window.applyTheme) {{
                window.applyTheme(savedTheme, {str(dark_mode.value).lower()});
            }} else {{
                // Fallback: manually sync body--dark class
                if (document.body) {{
                    if ({str(dark_mode.value).lower()}) {{
                        document.body.classList.add('body--dark');
                    }} else {{
                        document.body.classList.remove('body--dark');
                    }}
                }}
            }}
        ''')
        # Persist to SettingsManager
        settings_manager.set('dark_mode', dark_mode.value)

    async def autonomous_change_handler(value):
        autonomous_state_ref['value'] = value
        openmind.autonomous_reasoning = value
        # Persist to both localStorage and SettingsManager
        await ui.run_javascript(f'localStorage.setItem("autonomous-reasoning", "{str(value).lower()}")')
        settings_manager.set('autonomous_reasoning', value)
        await toggle_autonomous_reasoning(value)

    nav = Navigation(current_page='settings', dark_mode=dark_mode, drawer=drawer)
    nav.create_header(
        autonomous_callback=autonomous_change_handler,
        dark_mode_callback=lambda: None,  # theme is managed by the switch on this page
        autonomous_state=autonomous_state_ref['value']
    )

    # Initialize theme and autonomous state from storage
    async def init_settings_from_storage():
        await init_theme_from_storage()
        
        # Restore autonomous reasoning state (already set from SettingsManager, but sync from localStorage too)
        stored_autonomous = await ui.run_javascript('localStorage.getItem("autonomous-reasoning")')
        if stored_autonomous == 'true':
            autonomous_state_ref['value'] = True
            openmind.autonomous_reasoning = True
            settings_manager.set('autonomous_reasoning', True)
            await toggle_autonomous_reasoning(True)
            # Update switch if it exists
            await ui.run_javascript('''
                const switches = document.querySelectorAll("input[type=\'checkbox\']");
                switches.forEach(sw => {
                    if (sw.closest(".q-switch") && sw.parentElement.textContent.includes("Autonomous")) {
                        sw.checked = true;
                        sw.dispatchEvent(new Event("change", { bubbles: true }));
                    }
                });
            ''')
    
    ui.timer(0.1, init_settings_from_storage, once=True)

    # Settings container with scrollbar - using page-content class for full width
    with ui.column().classes('page-content settings-page-content'):
        with ui.column().classes('w-full gap-4 p-4').style('flex: 1 1 0; min-height: 0; overflow-y: auto; width: 100%; box-sizing: border-box;'):
            ui.label('Settings').classes('text-2xl font-bold')

            # Appearance Card
            with ui.card().classes('w-full'):
                ui.label('Appearance').classes('text-lg font-semibold')
                ui.separator()
                
                # Dark Mode Toggle
                with ui.row().classes('items-center justify-between w-full q-mb-md'):
                    ui.label('Dark Mode').classes('font-mono')
                    ui.switch(value=dark_mode.value, on_change=on_theme_switch)
                
                # Theme Selector
                with ui.column().classes('w-full'):
                    ui.label('Color Theme').classes('font-mono q-mb-2')
                    async def change_theme(theme_name):
                        # CRITICAL FIX #3: Use unified applyTheme function
                        await ui.run_javascript(f'''
                            const savedDarkMode = localStorage.getItem('theme') === 'dark';
                            if (window.applyTheme) {{
                                window.applyTheme('{theme_name}', savedDarkMode);
                            }} else {{
                                // Fallback if applyTheme not available
                                if (document.body) {{
                                    document.body.setAttribute('data-theme', '{theme_name}');
                                }}
                                if (document.documentElement) {{
                                    document.documentElement.setAttribute('data-theme', '{theme_name}');
                                }}
                                if (window.updateThemeStyle) {{
                                    window.updateThemeStyle('{theme_name}');
                                }}
                            }}
                            localStorage.setItem('ui-theme', '{theme_name}');
                        ''')
                        # Persist to SettingsManager
                        settings_manager.set('theme', theme_name)
                        ui.notify(f'Theme changed to {theme_name}', type='positive')
                    
                    async def init_theme_selector():
                        saved = await ui.run_javascript('localStorage.getItem("ui-theme") || "everforest"')
                        theme_select.value = saved
                    
                    theme_select = ui.select(
                        options={
                            'gruvbox': '🎨 Gruvbox',
                            'everforest': '🌲 Everforest',
                            'nord': '❄️ Nord',
                            'catppuccin': '☕ Catppuccin Mocha'
                        },
                        value='everforest',
                        on_change=lambda e: change_theme(e.value)
                    ).classes('theme-selector w-full').props('outlined')
                    
                    ui.timer(0.1, init_theme_selector, once=True)

            # API Keys Card
            with ui.card().classes('w-full'):
                ui.label('API Keys').classes('text-lg font-semibold')
                ui.separator()
                
                # API Provider dropdown
                api_providers = {
                    'openai': 'OpenAI',
                    'groq': 'Groq',
                    'together': 'Together AI',
                    'ai71': 'AI71',
                    'ollama_cloud': 'Ollama Cloud'
                }
                
                with ui.row().classes('items-center w-full gap-2'):
                    openmind.service_input = ui.select(
                        options=api_providers,
                        label='Service',
                        value='openai'
                    ).classes('flex-1 input').props('outlined')
                    openmind.key_input = ui.input('API Key').classes('flex-1 input').props('password')
                with ui.row().classes('gap-2'):
                    ui.button('Add API Key', on_click=openmind.add_api_key, icon='add').classes('api-action')
                    ui.button('List API Keys', on_click=openmind.list_api_keys, icon='list').classes('api-action')
                    keys_container = ui.column().classes('w-full')
                    openmind.keys_container = keys_container
            
            # Ollama Configuration Card
            with ui.card().classes('w-full'):
                ui.label('Ollama Configuration').classes('text-lg font-semibold')
                ui.separator()
                
                # Base URL input
                ollama_url_input = ui.input(
                    'Ollama Base URL',
                    value=settings_manager.get('ollama_base_url', 'http://localhost:11434'),
                    placeholder='http://localhost:11434'
                ).classes('w-full input').props('outlined')
                
                ui.label('Configure the base URL for your Ollama instance. Can be localhost or another computer on the network.').classes('text-sm text-gray-500 q-mt-2')
                
                async def save_ollama_url():
                    base_url = ollama_url_input.value.strip()
                    if not base_url:
                        ui.notify('Please enter a valid Ollama base URL', type='warning')
                        return
                    
                    # Validate URL format
                    if not (base_url.startswith('http://') or base_url.startswith('https://')):
                        ui.notify('URL must start with http:// or https://', type='warning')
                        return
                    
                    # Save to settings
                    settings_manager.set('ollama_base_url', base_url)
                    
                    # Update OllamaHandler instances
                    ollama_model.update_base_url(base_url)
                    openmind.ollama_handler.update_base_url(base_url)
                    
                    # Test connection and reload models
                    ui.notify('Testing Ollama connection and reloading models...', type='info')
                    try:
                        import httpx
                        test_url = base_url.rstrip('/')
                        if test_url.endswith('/api'):
                            test_url = test_url[:-4]
                        response = httpx.get(f'{test_url}/api/tags', timeout=5.0)
                        if response.status_code == 200:
                            models_data = response.json()
                            model_count = len(models_data.get('models', []))
                            ui.notify(f'Ollama connection successful! Found {model_count} model(s). URL saved.', type='positive')
                            
                            # Reload models in ollama_handler
                            try:
                                models = ollama_model.list_models()
                                if models:
                                    logging.info(f"Ollama models reloaded: {len(models)} models found")
                            except Exception as e:
                                logging.warning(f"Could not reload models: {e}")
                            
                            # Update URL display
                            url_status.text = ollama_model.base_url
                        else:
                            ui.notify(f'Ollama connection test returned status {response.status_code}', type='warning')
                    except Exception as e:
                        ui.notify(f'Could not connect to Ollama: {str(e)}', type='negative')
                        logging.warning(f'Ollama connection test failed: {e}')
                
                async def test_ollama_connection():
                    base_url = ollama_url_input.value.strip()
                    if not base_url:
                        ui.notify('Please enter a URL first', type='warning')
                        return
                    
                    ui.notify('Testing connection...', type='info')
                    try:
                        import httpx
                        test_url = base_url.rstrip('/')
                        if test_url.endswith('/api'):
                            test_url = test_url[:-4]
                        
                        # Temporarily update to test
                        original_url = ollama_model.base_url
                        ollama_model.update_base_url(base_url)
                        
                        response = httpx.get(f'{test_url}/api/tags', timeout=5.0)
                        if response.status_code == 200:
                            models_data = response.json()
                            model_count = len(models_data.get('models', []))
                            ui.notify(f'✓ Connection successful! Found {model_count} model(s)', type='positive')
                            
                            # Reload models
                            try:
                                models = ollama_model.list_models()
                                if models:
                                    logging.info(f"Ollama models reloaded: {len(models)} models found")
                            except Exception as e:
                                logging.warning(f"Could not reload models: {e}")
                        else:
                            ui.notify(f'Connection test returned status {response.status_code}', type='warning')
                            # Restore original URL if test failed
                            ollama_model.update_base_url(original_url)
                    except httpx.ConnectError:
                        ui.notify('Could not connect to Ollama. Check if Ollama is running and the URL is correct.', type='negative')
                        # Restore original URL if test failed
                        try:
                            ollama_model.update_base_url(original_url)
                        except:
                            pass
                    except Exception as e:
                        ui.notify(f'Connection test failed: {str(e)}', type='negative')
                        # Restore original URL if test failed
                        try:
                            ollama_model.update_base_url(original_url)
                        except:
                            pass
                
                with ui.row().classes('gap-2 q-mt-4'):
                    ui.button('Save & Test Connection', on_click=save_ollama_url, icon='save').classes('api-action')
                    ui.button('Test Connection', on_click=test_ollama_connection, icon='network_check').classes('api-action')
                
                # Show current status
                with ui.row().classes('items-center gap-2 q-mt-2'):
                    ui.label('Current URL:').classes('font-mono text-sm')
                    url_status = ui.label(ollama_model.base_url).classes('font-mono text-sm text-gray-600')
                
                async def update_url_display():
                    url_status.text = ollama_model.base_url
                
                ui.timer(0.1, update_url_display, once=True)
            
            # Ollama Cloud Configuration Card
            with ui.card().classes('w-full'):
                ui.label('Ollama Cloud Configuration').classes('text-lg font-semibold')
                ui.separator()
                
                ui.label('Configure Ollama Cloud API access. Add your API key in the API Keys section above (service: ollama_cloud), then configure the base URL here.').classes('text-sm text-gray-500 q-mb-4')
                
                # Base URL input for Ollama Cloud
                ollama_cloud_url_input = ui.input(
                    'Ollama Cloud Base URL',
                    value=settings_manager.get('ollama_cloud_base_url', 'https://ollama.com'),
                    placeholder='https://ollama.com'
                ).classes('w-full input').props('outlined')
                
                async def save_ollama_cloud_url():
                    base_url = ollama_cloud_url_input.value.strip()
                    if not base_url:
                        ui.notify('Please enter a valid Ollama Cloud base URL', type='warning')
                        return
                    
                    # Validate URL format
                    if not (base_url.startswith('http://') or base_url.startswith('https://')):
                        ui.notify('URL must start with http:// or https://', type='warning')
                        return
                    
                    # Save to settings
                    settings_manager.set('ollama_cloud_base_url', base_url)
                    
                    # Test connection
                    ui.notify('Testing Ollama Cloud connection...', type='info')
                    ollama_cloud_key = openmind.api_manager.get_api_key('ollama_cloud')
                    if not ollama_cloud_key:
                        ui.notify('Please add Ollama Cloud API key first in the API Keys section above', type='warning')
                        return
                    
                    try:
                        import httpx
                        test_url = base_url.rstrip('/')
                        if test_url.endswith('/api'):
                            test_url = test_url[:-4]
                        
                        headers = {
                            "Authorization": f"Bearer {ollama_cloud_key}",
                            "Content-Type": "application/json"
                        }
                        
                        response = httpx.get(f'{test_url}/api/tags', headers=headers, timeout=5.0)
                        if response.status_code == 200:
                            models_data = response.json()
                            model_count = len(models_data.get('models', []))
                            ui.notify(f'Ollama Cloud connection successful! Found {model_count} model(s). URL saved.', type='positive')
                            
                            # Update URL display
                            cloud_url_status.text = base_url
                        else:
                            ui.notify(f'Ollama Cloud connection test returned status {response.status_code}', type='warning')
                    except Exception as e:
                        ui.notify(f'Could not connect to Ollama Cloud: {str(e)}', type='negative')
                        logging.warning(f'Ollama Cloud connection test failed: {e}')
                
                async def test_ollama_cloud_connection():
                    base_url = ollama_cloud_url_input.value.strip()
                    if not base_url:
                        ui.notify('Please enter a URL first', type='warning')
                        return
                    
                    ollama_cloud_key = openmind.api_manager.get_api_key('ollama_cloud')
                    if not ollama_cloud_key:
                        ui.notify('Please add Ollama Cloud API key first in the API Keys section above', type='warning')
                        return
                    
                    ui.notify('Testing connection...', type='info')
                    try:
                        import httpx
                        test_url = base_url.rstrip('/')
                        if test_url.endswith('/api'):
                            test_url = test_url[:-4]
                        
                        headers = {
                            "Authorization": f"Bearer {ollama_cloud_key}",
                            "Content-Type": "application/json"
                        }
                        
                        response = httpx.get(f'{test_url}/api/tags', headers=headers, timeout=5.0)
                        if response.status_code == 200:
                            models_data = response.json()
                            model_count = len(models_data.get('models', []))
                            ui.notify(f'✓ Connection successful! Found {model_count} model(s)', type='positive')
                        else:
                            ui.notify(f'Connection test returned status {response.status_code}', type='warning')
                    except httpx.ConnectError:
                        ui.notify('Could not connect to Ollama Cloud. Check the URL and API key.', type='negative')
                    except Exception as e:
                        ui.notify(f'Connection test failed: {str(e)}', type='negative')
                
                with ui.row().classes('gap-2 q-mt-4'):
                    ui.button('Save & Test Connection', on_click=save_ollama_cloud_url, icon='save').classes('api-action')
                    ui.button('Test Connection', on_click=test_ollama_cloud_connection, icon='network_check').classes('api-action')
                
                # Show current status
                with ui.row().classes('items-center gap-2 q-mt-2'):
                    ui.label('Current URL:').classes('font-mono text-sm')
                    cloud_url_status = ui.label(settings_manager.get('ollama_cloud_base_url', 'https://ollama.com')).classes('font-mono text-sm text-gray-600')
                
                async def update_cloud_url_display():
                    cloud_url_status.text = settings_manager.get('ollama_cloud_base_url', 'https://ollama.com')
                
                ui.timer(0.1, update_cloud_url_display, once=True)
            
            # Timeout Settings Card
            with ui.card().classes('w-full'):
                ui.label('Advanced Timeout Settings').classes('text-lg font-semibold')
                ui.separator()
                
                ui.label('Configure timeout values for API providers and Ollama connections (in seconds).').classes('text-sm text-gray-500 q-mb-4')
                
                # API Timeout (for OpenAI, Groq, Together, AI71)
                api_timeout_input = ui.number(
                    'API Provider Timeout',
                    value=settings_manager.get('api_timeout', 60),
                    min=1,
                    max=600,
                    step=1,
                    format='%.0f'
                ).classes('w-full input').props('outlined')
                ui.label('Timeout for OpenAI, Groq, Together AI, and AI71 API requests (default: 60 seconds)').classes('text-xs text-gray-500 q-mt-1 q-mb-2')
                
                # Ollama Cloud Streaming Timeout
                ollama_cloud_timeout_input = ui.number(
                    'Ollama Cloud Streaming Timeout',
                    value=settings_manager.get('ollama_cloud_timeout', 300),
                    min=10,
                    max=1800,
                    step=10,
                    format='%.0f'
                ).classes('w-full input').props('outlined')
                ui.label('Timeout for Ollama Cloud streaming requests (default: 300 seconds = 5 minutes)').classes('text-xs text-gray-500 q-mt-1 q-mb-2')
            
                # Ollama Cloud Non-Streaming Timeout
                ollama_cloud_non_streaming_timeout_input = ui.number(
                    'Ollama Cloud Non-Streaming Timeout',
                    value=settings_manager.get('ollama_cloud_timeout_non_streaming', 60),
                    min=1,
                    max=600,
                    step=1,
                    format='%.0f'
                ).classes('w-full input').props('outlined')
                ui.label('Timeout for Ollama Cloud non-streaming requests (default: 60 seconds)').classes('text-xs text-gray-500 q-mt-1 q-mb-2')
                
                # Ollama Local Timeout
                ollama_timeout_input = ui.number(
                    'Ollama Local Timeout',
                    value=settings_manager.get('ollama_timeout', 10),
                    min=1,
                    max=300,
                    step=1,
                    format='%.0f'
                ).classes('w-full input').props('outlined')
                ui.label('Timeout for local Ollama requests (default: 10 seconds)').classes('text-xs text-gray-500 q-mt-1 q-mb-2')
                
                # Retry Timeout
                retry_timeout_input = ui.number(
                    'Retry Timeout',
                    value=settings_manager.get('retry_timeout', 30.0),
                    min=1,
                    max=300,
                    step=1,
                    format='%.1f'
                ).classes('w-full input').props('outlined')
                ui.label('Timeout for each retry attempt (default: 30.0 seconds)').classes('text-xs text-gray-500 q-mt-1 q-mb-2')
                
                # Retry Max Attempts
                retry_max_attempts_input = ui.number(
                    'Max Retry Attempts',
                    value=settings_manager.get('retry_max_attempts', 3),
                    min=1,
                    max=10,
                    step=1,
                    format='%.0f'
                ).classes('w-full input').props('outlined')
                ui.label('Maximum number of retry attempts for failed requests (default: 3)').classes('text-xs text-gray-500 q-mt-1 q-mb-2')
                
                async def save_timeout_settings():
                    """Save all timeout settings"""
                    try:
                        settings_manager.set('api_timeout', int(api_timeout_input.value))
                        settings_manager.set('ollama_cloud_timeout', int(ollama_cloud_timeout_input.value))
                        settings_manager.set('ollama_cloud_timeout_non_streaming', int(ollama_cloud_non_streaming_timeout_input.value))
                        settings_manager.set('ollama_timeout', int(ollama_timeout_input.value))
                        settings_manager.set('retry_timeout', float(retry_timeout_input.value))
                        settings_manager.set('retry_max_attempts', int(retry_max_attempts_input.value))
                        ui.notify('Timeout settings saved successfully!', type='positive')
                        logging.info("Timeout settings saved")
                    except Exception as e:
                        ui.notify(f'Error saving timeout settings: {str(e)}', type='negative')
                        logging.error(f"Error saving timeout settings: {e}")
                
                ui.button('Save Timeout Settings', on_click=save_timeout_settings, icon='save').classes('api-action q-mt-4')
            
            # Data Management Card
            with ui.card().classes('w-full'):
                ui.label('Data Management').classes('text-lg font-semibold')
                ui.separator()
                
                ui.label('Clear logs and data files to free up disk space. This will delete all runtime-generated files but preserve folder structure.').classes('text-sm text-gray-500 q-mb-4')
                
                async def clear_logs_handler():
                    """Clear all log files"""
                    result = settings_manager.clear_logs()
                    if result['success']:
                        if result['count'] > 0:
                            ui.notify(f'Cleared {result["count"]} log file(s): {", ".join(result["cleared"])}', type='positive')
                        else:
                            ui.notify('No log files found to clear', type='info')
                    else:
                        error_msg = '; '.join(result['errors'])
                        ui.notify(f'Errors clearing logs: {error_msg}', type='warning')
                        logging.error(f"Error clearing logs: {result['errors']}")
                
                async def clear_mindx_handler():
                    """Clear all mindx data files"""
                    result = settings_manager.clear_mindx_data()
                    if result['success']:
                        if result['cleared'] > 0:
                            ui.notify(f'Cleared {result["cleared"]} mindx data file(s)', type='positive')
                        else:
                            ui.notify('No mindx data files found to clear', type='info')
                    else:
                        error_msg = '; '.join(result['errors'])
                        ui.notify(f'Errors clearing mindx data: {error_msg}', type='warning')
                        logging.error(f"Error clearing mindx data: {result['errors']}")
                
                async def clear_memory_handler():
                    """Clear all memory data files"""
                    result = settings_manager.clear_memory_data()
                    if result['success']:
                        if result['total_files'] > 0:
                            cleared_locations = ', '.join(result['cleared'])
                            ui.notify(f'Cleared {result["total_files"]} memory file(s) from: {cleared_locations}', type='positive')
                        else:
                            ui.notify('No memory data files found to clear', type='info')
                    else:
                        error_msg = '; '.join(result['errors'])
                        ui.notify(f'Errors clearing memory data: {error_msg}', type='warning')
                        logging.error(f"Error clearing memory data: {result['errors']}")
                
                async def clear_all_handler():
                    """Clear all logs and data files"""
                    result = settings_manager.clear_all_data()
                    if result['success']:
                        total = result['total_cleared']
                        if total > 0:
                            message = f'Cleared {total} file(s) total:\n'
                            message += f"  • Logs: {result['logs']['count']} file(s)\n"
                            message += f"  • Mindx: {result['mindx']['cleared']} file(s)\n"
                            message += f"  • Memory: {result['memory']['total_files']} file(s)"
                            ui.notify(message, type='positive', timeout=5)
                        else:
                            ui.notify('No files found to clear', type='info')
                    else:
                        error_msg = '; '.join(result['errors'])
                        ui.notify(f'Some errors occurred: {error_msg}', type='warning')
                        logging.error(f"Errors clearing all data: {result['errors']}")
                
                with ui.column().classes('w-full gap-3'):
                    with ui.row().classes('w-full gap-2'):
                        ui.button('Clear Logs', on_click=clear_logs_handler, icon='description').classes('flex-1').props('color=orange')
                        ui.button('Clear Mindx Data', on_click=clear_mindx_handler, icon='psychology').classes('flex-1').props('color=orange')
                    
                    with ui.row().classes('w-full gap-2'):
                        ui.button('Clear Memory Data', on_click=clear_memory_handler, icon='memory').classes('flex-1').props('color=orange')
                        ui.button('Clear All Data', on_click=clear_all_handler, icon='delete_sweep').classes('flex-1').props('color=red')
                    
                    ui.label('⚠️ Warning: These actions cannot be undone. Make sure to backup important data before clearing.').classes('text-xs text-orange-600 q-mt-2')

@ui.page('/logs')
def logs_page():
    add_head_html(ui, settings_manager.sync_to_localStorage())
    dark_mode = ui.dark_mode()
    # Ensure dark mode reflects persisted settings before building header/UI
    try:
        dark_mode.value = settings_manager.get('dark_mode', True)
    except Exception:
        dark_mode.value = True
    drawer = SideNav(current_page='logs').create_drawer()

    async def init_theme_from_storage():
        # Load from SettingsManager first (server-side persistence)
        theme_name = settings_manager.get('theme', 'everforest')
        dark_mode_setting = settings_manager.get('dark_mode', True)
        
        # Sync to localStorage (browser persistence)
        await ui.run_javascript(f'''
            localStorage.setItem('ui-theme', '{theme_name}');
            localStorage.setItem('theme', '{'dark' if dark_mode_setting else 'light'}');
        ''')
        
        # Set dark_mode value
        dark_mode.value = dark_mode_setting
        
        # CRITICAL FIX #2: Sync with unified theme system after restoring
        await ui.run_javascript(f'''
            const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
            if (window.applyTheme) {{
                window.applyTheme(savedTheme, {str(dark_mode.value).lower()});
            }}
        ''')

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
