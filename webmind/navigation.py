# navigation.py (c) Gregory L. Magnusson MIT license 2024
# Unified navigation component for lmagi application
# Provides consistent navigation across all pages with gruvbox hacker theme

from nicegui import ui

class Navigation:
    """
    Minimal header navigation for lmagi application
    Only shows essential controls (autonomous toggle, dark mode)
    Main navigation is in the sidebar
    """

    def __init__(self, current_page='main', dark_mode=None, drawer=None):
        """
        Initialize navigation component

        Args:
            current_page: Current page identifier ('chat', 'ollama', 'logs', 'settings')
            dark_mode: Optional dark mode object for theme switching
            drawer: Optional drawer reference for mobile menu
        """
        self.current_page = current_page
        self.dark_mode = dark_mode
        self.drawer = drawer

    def create_header(self, autonomous_callback=None, dark_mode_callback=None, autonomous_state=None):
        """
        Create minimal header with only essential controls

        Args:
            autonomous_callback: Callback for autonomous reasoning toggle
            dark_mode_callback: Callback for dark mode toggle
            autonomous_state: Current state of autonomous reasoning (True/False)
        """
        with ui.header().classes('app-header items-center justify-end q-px-md q-py-sm'):
            with ui.row().classes('items-center gap-3'):
                # App title/logo (minimal, left-aligned)
                with ui.link(target='/').classes('absolute left-16'):
                    ui.label('🧠 lmagi').classes('text-xl font-bold cursor-pointer')
                
                # Sidebar collapse/expand hamburger (always visible, left-aligned)
                ui.button(
                    icon='menu',
                    on_click=lambda: ui.run_javascript('window.toggleSidebarCollapse && window.toggleSidebarCollapse()')
                ).props('flat round color=white').classes('absolute left-2 q-mr-md')

                # Autonomous reasoning toggle
                if autonomous_callback:
                    initial_state = bool(autonomous_state) if autonomous_state is not None else False
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('settings').classes('text-sm')
                        ui.switch('Autonomous', value=initial_state, on_change=autonomous_callback).props(
                            'color=white'
                        ).classes('text-sm font-mono')

                # Dark mode toggle (icon-only)
                if dark_mode_callback and self.dark_mode:
                    dark_mode_btn = ui.button(
                        on_click=lambda: self._toggle_dark_mode(dark_mode_btn, dark_mode_callback),
                        icon='dark_mode' if not self.dark_mode.value else 'light_mode'
                    ).props('flat round color=white').classes('nav-control-btn')

                # Fullscreen toggle
                fullscreen_btn = ui.button(
                    on_click=self._toggle_fullscreen,
                    icon='fullscreen'
                ).props('flat round color=white').classes('nav-control-btn fullscreen-btn')

    async def _toggle_dark_mode(self, button, callback):
        """Toggle dark mode and update button"""
        if callback:
            await callback()
        button.props(f'icon={"light_mode" if self.dark_mode.value else "dark_mode"}')

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        ui.run_javascript('''
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(err => {
                    console.log('Error attempting to enable fullscreen:', err);
                });
            } else {
                document.exitFullscreen();
            }
            
            // Update icon based on fullscreen state
            setTimeout(() => {
                const isFullscreen = !!document.fullscreenElement;
                const btn = document.querySelector('.fullscreen-btn');
                if (btn) {
                    const icon = btn.querySelector('i');
                    if (icon) {
                        icon.textContent = isFullscreen ? 'fullscreen_exit' : 'fullscreen';
                    }
                }
            }, 100);
        ''')


class SideNav:
    """
    Enhanced side navigation drawer - MAIN NAVIGATION HUB
    Gruvbox hacker-themed sidebar with all navigation items
    """

    def __init__(self, current_page='main'):
        self.current_page = current_page

    def create_drawer(self):
        """Create enhanced side navigation drawer with hacker aesthetic"""
        with ui.left_drawer(fixed=True, bordered=False).props('width=260').classes('left-drawer') as drawer:
            with ui.column().classes('drawer-content'):
                # Sidebar Header - Compact
                with ui.row().classes('items-center q-px-md q-pt-md q-pb-sm'):
                    ui.icon('terminal').classes('text-lg')
                    ui.label('NAV').classes('text-xl font-bold q-ml-2')
                
                ui.separator()

                # Main Navigation Items - Compact layout
                with ui.column().classes('q-px-md'):
                    self._create_drawer_item('💬 Chat', '/', 'chat', 'chat')
                    self._create_drawer_item('🤖 Ollama', '/ollama', 'ollama', 'psychology')
                    self._create_drawer_item('📊 Logs', '/logs', 'logs', 'description')
                    self._create_drawer_item('⚙️ Settings', '/settings', 'settings', 'settings')

                ui.separator().classes('q-mt-auto')

                # Compact Footer Section
                with ui.column().classes('q-px-md q-pb-md'):
                    ui.markdown('[easyAGI](https://rage.pythai.net)').classes('text-xs opacity-60')

        return drawer

    def _create_drawer_item(self, label, target, page_id, icon):
        """Create enhanced drawer navigation item with active state - compact version"""
        is_active = self.current_page == page_id

        with ui.item(on_click=lambda t=target: ui.navigate.to(t)).props('clickable').classes(
            'cursor-pointer drawer-nav-item' + (' active' if is_active else '')
        ):
            with ui.item_section():
                ui.icon(icon).classes('text-md')
            with ui.item_section():
                ui.label(label).classes('font-mono text-sm')

    def _clear_history(self):
        """Clear chat history (placeholder - implement actual functionality)"""
        ui.notify('History cleared', type='info')


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
        with ui.row().classes('items-center gap-2 p-2'):
            ui.icon('home').classes('text-sm')

            for i, (label, url) in enumerate(self.path_items):
                if i > 0:
                    ui.icon('chevron_right').classes('text-xs opacity-60')

                if i == len(self.path_items) - 1:
                    # Current page - no link
                    ui.label(label).classes('font-bold text-sm')
                else:
                    # Link to previous pages
                    with ui.link(target=url):
                        ui.label(label).classes('text-sm hover:underline cursor-pointer')


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
