# navigation.py (c) Gregory L. Magnusson MIT license 2024
# Unified navigation component for lmagi application
# Provides consistent navigation across all pages

from nicegui import ui

class Navigation:
    """
    Unified navigation system for lmagi application
    Provides consistent header navigation across all pages
    """

    def __init__(self, current_page='main', dark_mode=None, drawer=None):
        """
        Initialize navigation component

        Args:
            current_page: Current page identifier ('main', 'ollama', 'chat', 'logs', 'api')
            dark_mode: Optional dark mode object for theme switching
        """
        self.current_page = current_page
        self.dark_mode = dark_mode
        self.drawer = drawer

    def create_header(self, autonomous_callback=None, dark_mode_callback=None, autonomous_state=None):
        """
        Create navigation header with page links and controls

        Args:
            autonomous_callback: Callback for autonomous reasoning toggle
            dark_mode_callback: Callback for dark mode toggle
            autonomous_state: Current state of autonomous reasoning (True/False)
        """
        with ui.header().classes('items-center justify-between bg-blue-600 text-white p-4 shadow-md'):
            with ui.row().classes('items-center gap-3'):
                # Optional drawer toggle for mobile
                if self.drawer is not None:
                    ui.button(icon='menu', on_click=self.drawer.toggle).props('flat round color=white')
                # App logo/title
                with ui.link(target='/'):
                    ui.label('🧠 lmagi').classes('text-2xl font-bold cursor-pointer')

                # Navigation menu
                with ui.row().classes('gap-2'):
                    self._create_nav_button('Chat', '/', 'chat', 'chat')
                    self._create_nav_button('Ollama', '/ollama', 'ollama', 'psychology')
                    self._create_nav_button('Logs', '/#logs', 'logs', 'description')
                    # Settings page (includes API Keys)
                    self._create_nav_button('Settings', '/settings', 'settings', 'settings')

            # Right side controls
            with ui.row().classes('items-center gap-2'):
                # Autonomous reasoning toggle
                if autonomous_callback:
                    # Use autonomous_state if provided, otherwise default to False
                    # Pass as plain boolean to avoid serialization issues
                    initial_state = bool(autonomous_state) if autonomous_state is not None else False
                    ui.switch('Autonomous', value=initial_state, on_change=autonomous_callback).props(
                        'color=white'
                    ).classes('text-sm')

                # Dark mode toggle (icon-only)
                if dark_mode_callback and self.dark_mode:
                    dark_mode_btn = ui.button(
                        on_click=lambda: self._toggle_dark_mode(dark_mode_btn, dark_mode_callback),
                        icon='dark_mode' if not self.dark_mode.value else 'light_mode'
                    ).props('flat round color=white')

    def _create_nav_button(self, label, target, page_id, icon):
        """Create navigation button with active state"""
        is_active = self.current_page == page_id

        with ui.link(target=target):
            ui.button(label, icon=icon).props(
                f'flat color=white {"outline" if is_active else ""}'
            ).classes('nav-button' + (' active' if is_active else ''))

    async def _toggle_dark_mode(self, button, callback):
        """Toggle dark mode and update button"""
        if callback:
            await callback()
        button.set_text('Light Mode' if self.dark_mode.value else 'Dark Mode')
        button.props(f'icon={"light_mode" if self.dark_mode.value else "dark_mode"}')


class SideNav:
    """
    Side navigation drawer for mobile-responsive design
    """

    def __init__(self, current_page='main'):
        self.current_page = current_page

    def create_drawer(self):
        """Create side navigation drawer"""
        with ui.left_drawer(fixed=False, bordered=True).classes('bg-blue-100') as drawer:
            with ui.column().classes('w-full'):
                ui.label('Navigation').classes('text-xl font-bold p-4')
                ui.separator()

                self._create_drawer_item('💬 Chat', '/', 'chat')
                self._create_drawer_item('🤖 Ollama', '/ollama', 'ollama')
                self._create_drawer_item('📊 Logs', '/#logs', 'logs')
                self._create_drawer_item('🔑 API Keys', '/#api', 'api')

                ui.separator()
                ui.label('About').classes('text-sm p-4 text-gray-600')
                ui.markdown('[easyAGI Project](https://rage.pythai.net)').classes('p-4 text-sm')

        return drawer

    def _create_drawer_item(self, label, target, page_id):
        """Create drawer navigation item"""
        is_active = self.current_page == page_id

        with ui.item(on_click=lambda t=target: ui.navigate.to(t)).props('clickable').classes('cursor-pointer' + (' bg-blue-200' if is_active else '')):
            with ui.item_section():
                ui.label(label).classes(
                    'text-blue-800 font-bold' if is_active else 'text-blue-600'
                )


class BreadcrumbNav:
    """
    Breadcrumb navigation for showing current location
    """

    def __init__(self, path_items):
        """
        Initialize breadcrumb navigation

        Args:
            path_items: List of tuples [(label, url), ...]
        """
        self.path_items = path_items

    def create_breadcrumb(self):
        """Create breadcrumb navigation"""
        with ui.row().classes('items-center gap-2 p-2 bg-gray-100'):
            ui.icon('home').classes('text-gray-600')

            for i, (label, url) in enumerate(self.path_items):
                if i > 0:
                    ui.icon('chevron_right').classes('text-gray-400')

                if i == len(self.path_items) - 1:
                    # Current page - no link
                    ui.label(label).classes('font-bold text-blue-600')
                else:
                    # Link to previous pages
                    with ui.link(target=url):
                        ui.label(label).classes('text-blue-500 hover:underline cursor-pointer')


class QuickActions:
    """
    Quick action floating buttons
    """

    def __init__(self):
        self.actions = []

    def add_action(self, label, icon, callback):
        """Add a quick action"""
        self.actions.append((label, icon, callback))

    def create_fab(self, color='primary', icon='menu'):
        """Create floating action button menu"""
        with ui.page_sticky(position='bottom-right', x_offset=20, y_offset=20):
            with ui.button(icon=icon).props(f'fab color={color}'):
                with ui.menu().props('anchor="top left"'):
                    for label, action_icon, callback in self.actions:
                        with ui.menu_item(clickable=True, on_click=callback):
                            with ui.item_section():
                                ui.icon(action_icon)
                            with ui.item_section():
                                ui.label(label)
