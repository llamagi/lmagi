# html_head.py

def add_head_html(ui, server_settings=None):
    ui.add_head_html('<link rel="preconnect" href="https://fonts.googleapis.com">')
    ui.add_head_html('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">')
    
    # CRITICAL FIX #1: Inline critical CSS to prevent flash - load full CSS after
    # This ensures theme colors are available immediately, preventing white flash
    ui.add_head_html('<link rel="stylesheet" href="/gfx/easystyle.css">')
    
    # NEW: Seed localStorage from server settings BEFORE blocking theme init
    if server_settings:
        try:
            # Build script that writes server settings into localStorage synchronously
            seed_js = (
                "<script>(function(){try{"
                f"localStorage.setItem('ui-theme','{server_settings.get('ui-theme','everforest')}');"
                f"localStorage.setItem('theme','{server_settings.get('theme','dark')}');"
                f"localStorage.setItem('autonomous-reasoning','{server_settings.get('autonomous-reasoning','false')}');"
                f"localStorage.setItem('sidebar-width','{server_settings.get('sidebar-width','260')}');"
                f"localStorage.setItem('footer-height','{server_settings.get('footer-height','160')}');"
                "}catch(e){}})();</script>"
            )
            ui.add_head_html(seed_js)
        except Exception:
            # Fail silently; blocking script below still applies defaults
            pass
        
        # EXTRA: Inject minimal server-rendered critical CSS using persisted theme
        try:
            theme_name = server_settings.get('ui-theme', 'everforest')
            dark_mode = (server_settings.get('theme', 'dark') == 'dark')
            theme_colors = {
                'everforest': {
                    'bg0': '#2d353b', 'bg1': '#343f44', 'bg2': '#3d484d',
                    'bgl0': '#fdf6e3', 'bgl1': '#efead4', 'bgl2': '#dfd9c2',
                    'fgd': '#d3c6aa', 'fgl': '#5c6a72', 'blue': '#7fbbb3', 'green': '#a7c080'
                },
                'gruvbox': {
                    'bg0': '#282828', 'bg1': '#3c3836', 'bg2': '#504945',
                    'bgl0': '#fbf1c7', 'bgl1': '#ebdbb2', 'bgl2': '#d5c4a1',
                    'fgd': '#ebdbb2', 'fgl': '#3c3836', 'blue': '#458588', 'green': '#689d6a'
                },
                'nord': {
                    'bg0': '#2e3440', 'bg1': '#3b4252', 'bg2': '#434c5e',
                    'bgl0': '#eceff4', 'bgl1': '#e5e9f0', 'bgl2': '#d8dee9',
                    'fgd': '#eceff4', 'fgl': '#3b4252', 'blue': '#5e81ac', 'green': '#a3be8c'
                },
                'catppuccin': {
                    'bg0': '#1e1e2e', 'bg1': '#181825', 'bg2': '#313244',
                    'bgl0': '#eff1f5', 'bgl1': '#e6e9ef', 'bgl2': '#dce0e8',
                    'fgd': '#cdd6f4', 'fgl': '#4c4f69', 'blue': '#89b4fa', 'green': '#a6e3a1'
                },
            }
            c = theme_colors.get(theme_name, theme_colors['everforest'])
            bg0 = c['bg0'] if dark_mode else c['bgl0']
            bg1 = c['bg1'] if dark_mode else c['bgl1']
            fg = c['fgd'] if dark_mode else c['fgl']
            blue = c['blue']
            critical_css = (
                '<style id="server-theme-critical">'
                f'html,body{{background-color:{bg0} !important;color:{fg} !important;}}'
                f'.left-drawer{{background-color:{bg1} !important;border-right:1px solid {blue} !important;}}'
                f'.app-header{{background-color:{bg1} !important;border-bottom:1px solid {blue} !important;}}'
                f'.terminal-footer{{background-color:{bg1} !important;border-top:1px solid {blue} !important;}}'
                f'.q-card{{background-color:{bg1} !important;color:{fg} !important;}}'
                f'.q-btn:not(.q-fab){{background-color:{bg1} !important;color:{fg} !important;}}'
                '</style>'
            )
            ui.add_head_html(critical_css)
        except Exception:
            pass
    
    # CRITICAL: Block rendering until theme is applied - prevents flash
    ui.add_head_html('''
    <script>
        // BLOCKING SCRIPT: Must run synchronously before any rendering
        // This prevents FOUC (Flash of Unstyled Content)
        (function() {
            // Prevent rendering until theme is ready
            document.documentElement.style.display = 'none';
            
            // Store themeColors globally
            window.themeColors = {
                'gruvbox': {
                    '--bg-0': '#282828', '--bg-1': '#3c3836', '--bg-2': '#504945',
                    '--bg-light-0': '#fbf1c7', '--bg-light-1': '#ebdbb2', '--bg-light-2': '#d5c4a1',
                    '--fg-0': '#ebdbb2', '--fg-light-0': '#3c3836', '--gray': '#928374',
                    '--accent-blue': '#458588', '--accent-green': '#98971a', '--accent-purple': '#b16286',
                    '--accent-red': '#cc241d', '--accent-yellow': '#d79921', '--terminal-green': '#689d6a',
                    '--terminal-blue': '#458588'
                },
                'everforest': {
                    '--bg-0': '#2d353b', '--bg-1': '#343f44', '--bg-2': '#3d484d',
                    '--bg-light-0': '#fdf6e3', '--bg-light-1': '#efead4', '--bg-light-2': '#dfd9c2',
                    '--fg-0': '#d3c6aa', '--fg-light-0': '#5c6a72', '--gray': '#859289',
                    '--accent-blue': '#7fbbb3', '--accent-green': '#a7c080', '--accent-purple': '#d699b6',
                    '--accent-red': '#e67e80', '--accent-yellow': '#dbbc7f', '--terminal-green': '#a7c080',
                    '--terminal-blue': '#7fbbb3'
                },
                'nord': {
                    '--bg-0': '#2e3440', '--bg-1': '#3b4252', '--bg-2': '#434c5e',
                    '--bg-light-0': '#eceff4', '--bg-light-1': '#e5e9f0', '--bg-light-2': '#d8dee9',
                    '--fg-0': '#eceff4', '--fg-light-0': '#3b4252', '--gray': '#616e88',
                    '--accent-blue': '#5e81ac', '--accent-green': '#a3be8c', '--accent-purple': '#b48ead',
                    '--accent-red': '#bf616a', '--accent-yellow': '#ebcb8b', '--terminal-green': '#a3be8c',
                    '--terminal-blue': '#5e81ac'
                },
                'catppuccin': {
                    '--bg-0': '#1e1e2e', '--bg-1': '#181825', '--bg-2': '#313244',
                    '--bg-light-0': '#eff1f5', '--bg-light-1': '#e6e9ef', '--bg-light-2': '#dce0e8',
                    '--fg-0': '#cdd6f4', '--fg-light-0': '#4c4f69', '--gray': '#6c7086',
                    '--accent-blue': '#89b4fa', '--accent-green': '#a6e3a1', '--accent-purple': '#cba6f7',
                    '--accent-red': '#f38ba8', '--accent-yellow': '#f9e2af', '--terminal-green': '#a6e3a1',
                    '--terminal-blue': '#89b4fa'
                }
            };
            
            // CRITICAL FIX #2: Unified theme application - syncs both data-theme and body--dark
            function applyThemeImmediately() {
                try {
                    // Read from localStorage with fallback to defaults
                    const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
                    const savedThemeMode = localStorage.getItem('theme');
                    // Default to dark mode if not set (matches SettingsManager default)
                    const savedDarkMode = savedThemeMode === null ? true : savedThemeMode === 'dark';
                    
                    const colors = window.themeColors[savedTheme] || window.themeColors['everforest'];
                    const bgColor = savedDarkMode ? colors['--bg-0'] : colors['--bg-light-0'];
                    const bg1Color = savedDarkMode ? colors['--bg-1'] : colors['--bg-light-1'];
                    const fgColor = savedDarkMode ? colors['--fg-0'] : colors['--fg-light-0'];
                    const accentBlue = colors['--accent-blue'];
                    
                    // CRITICAL: Set body background color immediately to prevent white flash
                    // Apply to body directly before anything else
                    if (document.body) {
                        document.body.style.backgroundColor = bgColor;
                        document.body.style.color = fgColor;
                    }
                    
                    // CRITICAL: Apply theme to sidebar, header, and footer immediately
                    // These elements render early and need immediate styling
                    function applyThemeToElements() {
                        const drawer = document.querySelector('.left-drawer');
                        if (drawer) {
                            drawer.style.setProperty('background-color', bg1Color, 'important');
                            drawer.style.setProperty('border-right-color', accentBlue, 'important');
                        }
                        const header = document.querySelector('.app-header');
                        if (header) {
                            header.style.setProperty('background-color', bg1Color, 'important');
                            header.style.setProperty('border-bottom-color', accentBlue, 'important');
                        }
                        const footer = document.querySelector('.terminal-footer');
                        if (footer) {
                            footer.style.setProperty('background-color', bg1Color, 'important');
                            footer.style.setProperty('border-top-color', accentBlue, 'important');
                        }
                    }
                    
                    // Try to apply immediately
                    applyThemeToElements();
                    
                    // CRITICAL: Also apply theme to NiceGUI components as they appear
                    function applyThemeToNiceGUIComponents() {
                        const bg2Color = savedDarkMode ? colors['--bg-2'] : colors['--bg-light-2'];
                        
                        // Apply to all buttons
                        document.querySelectorAll('.q-btn:not(.q-fab)').forEach(btn => {
                            btn.style.setProperty('background-color', bg1Color, 'important');
                            btn.style.setProperty('color', fgColor, 'important');
                        });
                        
                        // Apply to all cards
                        document.querySelectorAll('.q-card').forEach(card => {
                            card.style.setProperty('background-color', bg1Color, 'important');
                            card.style.setProperty('color', fgColor, 'important');
                        });
                        
                        // Apply to all items/labels
                        document.querySelectorAll('.q-item, .q-item__label, label, .q-field__label').forEach(el => {
                            el.style.setProperty('color', fgColor, 'important');
                        });
                    }
                    
                    // Apply immediately and watch for new elements
                    applyThemeToNiceGUIComponents();
                    
                    // Also watch for elements to appear
                    if (document.body) {
                        const elementObserver = new MutationObserver(function() {
                            applyThemeToElements();
                            applyThemeToNiceGUIComponents();
                        });
                        elementObserver.observe(document.body, { childList: true, subtree: true });
                        // Disconnect after a short time to avoid performance issues
                        setTimeout(function() { elementObserver.disconnect(); }, 3000);
                    }
                    
                    // Apply data-theme attribute
                    if (document.documentElement) {
                        document.documentElement.setAttribute('data-theme', savedTheme);
                    }
                    if (document.body) {
                        document.body.setAttribute('data-theme', savedTheme);
                        // CRITICAL: Sync body--dark class with saved dark mode preference
                        // This must be done correctly for dark mode to persist
                        if (savedDarkMode) {
                            document.body.classList.add('body--dark');
                        } else {
                            document.body.classList.remove('body--dark');
                        }
                    }
                    
                    // Inject inline CSS with theme variables - CRITICAL for preventing flash
                    let cssVars = ':root, html[data-theme="' + savedTheme + '"], body[data-theme="' + savedTheme + '"], html[data-theme="' + savedTheme + '"] body, body[data-theme="' + savedTheme + '"] * {';
                    for (const [key, value] of Object.entries(colors)) {
                        cssVars += key + ':' + value + ' !important;';
                    }
                    cssVars += '}';
                    
                    // Add critical element styles to prevent flash
                    cssVars += ' body { background-color: ' + bgColor + ' !important; color: ' + fgColor + ' !important; }';
                    cssVars += ' .left-drawer { background-color: ' + bg1Color + ' !important; border-right-color: ' + accentBlue + ' !important; }';
                    cssVars += ' .app-header { background-color: ' + bg1Color + ' !important; border-bottom-color: ' + accentBlue + ' !important; }';
                    cssVars += ' .terminal-footer { background-color: ' + bg1Color + ' !important; border-top-color: ' + accentBlue + ' !important; }';
                    
                    // CRITICAL: Apply theme to NiceGUI components immediately to prevent flashing
                    const bg2Color = savedDarkMode ? colors['--bg-2'] : colors['--bg-light-2'];
                    const accentGreen = colors['--accent-green'];
                    
                    // Buttons - apply theme immediately
                    cssVars += ' .q-btn:not([class*="q-btn--"]):not(.q-fab), button.q-btn { background-color: ' + bg1Color + ' !important; color: ' + fgColor + ' !important; border-color: ' + accentBlue + ' !important; }';
                    cssVars += ' .q-btn:hover { background-color: ' + bg2Color + ' !important; color: ' + accentGreen + ' !important; }';
                    
                    // Cards - apply theme immediately
                    cssVars += ' .q-card { background-color: ' + bg1Color + ' !important; color: ' + fgColor + ' !important; border-color: ' + accentBlue + ' !important; }';
                    
                    // Items/Labels - apply theme immediately
                    cssVars += ' .q-item, .q-item__section { background-color: transparent !important; color: ' + fgColor + ' !important; }';
                    cssVars += ' .q-item:hover { background-color: ' + bg2Color + ' !important; }';
                    
                    // Labels/Text - apply theme immediately
                    cssVars += ' .q-item__label, label, .q-field__label { color: ' + fgColor + ' !important; }';
                    
                    // Input fields - apply theme immediately
                    cssVars += ' input, textarea, .q-field__control { background-color: ' + bg2Color + ' !important; color: ' + fgColor + ' !important; border-color: ' + accentBlue + ' !important; }';
                    
                    // Menu items - apply theme immediately
                    cssVars += ' .q-menu, .q-menu-item { background-color: ' + bg1Color + ' !important; color: ' + fgColor + ' !important; border-color: ' + accentBlue + ' !important; }';
                    cssVars += ' .q-menu-item:hover { background-color: ' + bg2Color + ' !important; }';
                    
                    // Switches/Toggles - apply theme immediately
                    cssVars += ' .q-switch { color: ' + accentBlue + ' !important; }';
                    
                    // FAB buttons - apply theme immediately
                    cssVars += ' .q-fab { background-color: ' + accentBlue + ' !important; color: ' + fgColor + ' !important; }';
                    
                    // Select/Dropdown - apply theme immediately
                    cssVars += ' .q-select, .q-field { background-color: ' + bg1Color + ' !important; color: ' + fgColor + ' !important; }';
                    
                    // Remove existing theme style if present
                    const existingStyle = document.getElementById('theme-inline');
                    if (existingStyle) {
                        existingStyle.remove();
                    }
                    
                    // Create and insert new style at the very beginning of head
                    const style = document.createElement('style');
                    style.id = 'theme-inline';
                    style.textContent = cssVars;
                    
                    if (document.head) {
                        document.head.insertBefore(style, document.head.firstChild);
                    } else {
                        // If head doesn't exist yet, wait for it
                        const observer = new MutationObserver(function(mutations, obs) {
                            if (document.head) {
                                document.head.insertBefore(style, document.head.firstChild);
                                obs.disconnect();
                            }
                        });
                        observer.observe(document.documentElement, { childList: true });
                    }
                    
                    // Re-enable rendering
                    document.documentElement.style.display = '';
                    
                    return savedTheme;
                } catch(e) {
                    console.error('Theme init error:', e);
                    document.documentElement.style.display = '';
                    return 'everforest';
                }
            }
            
            // Apply theme immediately
            applyThemeImmediately();
            
            // Restore autonomous reasoning state before page renders
            const savedAutonomous = localStorage.getItem('autonomous-reasoning');
            if (savedAutonomous === 'true') {
                window.restoredAutonomousState = true;
            } else {
                window.restoredAutonomousState = false;
            }
            
            // Unified theme update function - syncs both systems
            window.updateThemeStyle = function(theme) {
                const existingStyle = document.getElementById('theme-inline');
                if (window.themeColors) {
                    const savedDarkMode = localStorage.getItem('theme') === 'dark';
                    const colors = window.themeColors[theme] || window.themeColors['everforest'];
                    let cssVars = ':root, html[data-theme="' + theme + '"], body[data-theme="' + theme + '"], html[data-theme="' + theme + '"] body, body[data-theme="' + theme + '"] * {';
                    for (const [key, value] of Object.entries(colors)) {
                        cssVars += key + ':' + value + ' !important;';
                    }
                    cssVars += '}';
                    
                    const bgColor = savedDarkMode ? colors['--bg-0'] : colors['--bg-light-0'];
                    const bg1Color = savedDarkMode ? colors['--bg-1'] : colors['--bg-light-1'];
                    const fgColor = savedDarkMode ? colors['--fg-0'] : colors['--fg-light-0'];
                    const accentBlue = colors['--accent-blue'];
                    
                    // Add critical element styles to prevent flash
                    cssVars += ' body { background-color: ' + bgColor + ' !important; color: ' + fgColor + ' !important; }';
                    cssVars += ' .left-drawer { background-color: ' + bg1Color + ' !important; border-right-color: ' + accentBlue + ' !important; }';
                    cssVars += ' .app-header { background-color: ' + bg1Color + ' !important; border-bottom-color: ' + accentBlue + ' !important; }';
                    cssVars += ' .terminal-footer { background-color: ' + bg1Color + ' !important; border-top-color: ' + accentBlue + ' !important; }';
                    
                    // CRITICAL: Apply theme to NiceGUI components immediately to prevent flashing
                    const bg2Color = savedDarkMode ? colors['--bg-2'] : colors['--bg-light-2'];
                    const accentGreen = colors['--accent-green'];
                    
                    // Buttons - apply theme immediately
                    cssVars += ' .q-btn:not([class*="q-btn--"]):not(.q-fab), button.q-btn { background-color: ' + bg1Color + ' !important; color: ' + fgColor + ' !important; border-color: ' + accentBlue + ' !important; }';
                    cssVars += ' .q-btn:hover { background-color: ' + bg2Color + ' !important; color: ' + accentGreen + ' !important; }';
                    
                    // Cards - apply theme immediately
                    cssVars += ' .q-card { background-color: ' + bg1Color + ' !important; color: ' + fgColor + ' !important; border-color: ' + accentBlue + ' !important; }';
                    
                    // Items/Labels - apply theme immediately
                    cssVars += ' .q-item, .q-item__section { background-color: transparent !important; color: ' + fgColor + ' !important; }';
                    cssVars += ' .q-item:hover { background-color: ' + bg2Color + ' !important; }';
                    
                    // Labels/Text - apply theme immediately
                    cssVars += ' .q-item__label, label, .q-field__label { color: ' + fgColor + ' !important; }';
                    
                    // Input fields - apply theme immediately
                    cssVars += ' input, textarea, .q-field__control { background-color: ' + bg2Color + ' !important; color: ' + fgColor + ' !important; border-color: ' + accentBlue + ' !important; }';
                    
                    // Menu items - apply theme immediately
                    cssVars += ' .q-menu, .q-menu-item { background-color: ' + bg1Color + ' !important; color: ' + fgColor + ' !important; border-color: ' + accentBlue + ' !important; }';
                    cssVars += ' .q-menu-item:hover { background-color: ' + bg2Color + ' !important; }';
                    
                    // Switches/Toggles - apply theme immediately
                    cssVars += ' .q-switch { color: ' + accentBlue + ' !important; }';
                    
                    // FAB buttons - apply theme immediately
                    cssVars += ' .q-fab { background-color: ' + accentBlue + ' !important; color: ' + fgColor + ' !important; }';
                    
                    // Select/Dropdown - apply theme immediately
                    cssVars += ' .q-select, .q-field { background-color: ' + bg1Color + ' !important; color: ' + fgColor + ' !important; }';
                    
                    if (existingStyle) {
                        existingStyle.textContent = cssVars;
                    } else {
                        const style = document.createElement('style');
                        style.id = 'theme-inline';
                        style.textContent = cssVars;
                        if (document.head) {
                            document.head.insertBefore(style, document.head.firstChild);
                        }
                    }
                    
                    // Apply background colors immediately to prevent flash
                    if (document.body) {
                        document.body.style.backgroundColor = bgColor;
                        document.body.style.color = fgColor;
                    }
                    const drawer = document.querySelector('.left-drawer');
                    if (drawer) {
                        drawer.style.backgroundColor = bg1Color;
                    }
                    const header = document.querySelector('.app-header');
                    if (header) {
                        header.style.backgroundColor = bg1Color;
                    }
                    const footer = document.querySelector('.terminal-footer');
                    if (footer) {
                        footer.style.backgroundColor = bg1Color;
                    }
                    
                    // CRITICAL: Apply theme to NiceGUI components immediately when they appear
                    function applyThemeToNiceGUIComponents() {
                        const bg2Color = savedDarkMode ? colors['--bg-2'] : colors['--bg-light-2'];
                        const accentGreen = colors['--accent-green'];
                        
                        // Apply to all buttons
                        document.querySelectorAll('.q-btn:not(.q-fab)').forEach(btn => {
                            btn.style.setProperty('background-color', bg1Color, 'important');
                            btn.style.setProperty('color', fgColor, 'important');
                        });
                        
                        // Apply to all cards
                        document.querySelectorAll('.q-card').forEach(card => {
                            card.style.setProperty('background-color', bg1Color, 'important');
                            card.style.setProperty('color', fgColor, 'important');
                        });
                        
                        // Apply to all items/labels
                        document.querySelectorAll('.q-item, .q-item__label, label').forEach(el => {
                            el.style.setProperty('color', fgColor, 'important');
                        });
                    }
                    
                    // Apply immediately and watch for new elements
                    applyThemeToNiceGUIComponents();
                    const niceGUIObserver = new MutationObserver(function() {
                        applyThemeToNiceGUIComponents();
                    });
                    if (document.body) {
                        niceGUIObserver.observe(document.body, { childList: true, subtree: true });
                        // Disconnect after initial render to avoid performance issues
                        setTimeout(function() { niceGUIObserver.disconnect(); }, 3000);
                    }
                }
            };
            
            // Unified theme application function - syncs both data-theme and body--dark
            window.applyTheme = function(theme, darkMode) {
                // Use saved dark mode if not provided
                if (darkMode === undefined) {
                    darkMode = localStorage.getItem('theme') === 'dark';
                }
                
                // CRITICAL: Persist theme to localStorage FIRST
                localStorage.setItem('ui-theme', theme);
                localStorage.setItem('theme', darkMode ? 'dark' : 'light');
                
                // Apply data-theme attribute
                if (document.documentElement) {
                    document.documentElement.setAttribute('data-theme', theme);
                }
                if (document.body) {
                    document.body.setAttribute('data-theme', theme);
                    // CRITICAL: Sync body--dark class - must be done correctly
                    if (darkMode) {
                        document.body.classList.add('body--dark');
                    } else {
                        document.body.classList.remove('body--dark');
                    }
                    
                    // Apply background color immediately to prevent flash
                    if (window.themeColors && window.themeColors[theme]) {
                        const colors = window.themeColors[theme];
                        const bgColor = darkMode ? colors['--bg-0'] : colors['--bg-light-0'];
                        const bg1Color = darkMode ? colors['--bg-1'] : colors['--bg-light-1'];
                        const fgColor = darkMode ? colors['--fg-0'] : colors['--fg-light-0'];
                        const accentBlue = colors['--accent-blue'];
                        
                        document.body.style.backgroundColor = bgColor;
                        document.body.style.color = fgColor;
                        
                        // Apply to sidebar, header, footer immediately
                        const drawer = document.querySelector('.left-drawer');
                        if (drawer) {
                            drawer.style.backgroundColor = bg1Color;
                            drawer.style.borderRightColor = accentBlue;
                        }
                        const header = document.querySelector('.app-header');
                        if (header) {
                            header.style.backgroundColor = bg1Color;
                            header.style.borderBottomColor = accentBlue;
                        }
                        const footer = document.querySelector('.terminal-footer');
                        if (footer) {
                            footer.style.backgroundColor = bg1Color;
                            footer.style.borderTopColor = accentBlue;
                        }
                        
                        // CRITICAL: Apply theme to NiceGUI components immediately
                        const bg2Color = darkMode ? colors['--bg-2'] : colors['--bg-light-2'];
                        const accentGreen = colors['--accent-green'];
                        
                        // Apply to all buttons
                        document.querySelectorAll('.q-btn:not(.q-fab)').forEach(btn => {
                            btn.style.setProperty('background-color', bg1Color, 'important');
                            btn.style.setProperty('color', fgColor, 'important');
                        });
                        
                        // Apply to all cards
                        document.querySelectorAll('.q-card').forEach(card => {
                            card.style.setProperty('background-color', bg1Color, 'important');
                            card.style.setProperty('color', fgColor, 'important');
                        });
                        
                        // Apply to all items/labels
                        document.querySelectorAll('.q-item, .q-item__label, label').forEach(el => {
                            el.style.setProperty('color', fgColor, 'important');
                        });
                    }
                }
                
                // Update inline CSS
                if (window.updateThemeStyle) {
                    window.updateThemeStyle(theme);
                }
            };
        })();
    </script>
    ''')
    
    ui.add_head_html('<title>EasyAGI Augmented Generative Intelligence</title>')
    
    # CRITICAL FIX #4: Consolidated initialization - single DOMContentLoaded handler
    ui.add_head_html('''
    <script>
            // CRITICAL FIX #3: Single consolidated theme initialization function
        function initializeTheme() {
            const savedTheme = localStorage.getItem('ui-theme') || 'everforest';
            // CRITICAL: Read dark mode from localStorage - default to dark if not set
            const savedThemeMode = localStorage.getItem('theme');
            const savedDarkMode = savedThemeMode === null ? true : savedThemeMode === 'dark';
            
            // Apply theme using unified function
            if (window.applyTheme) {
                window.applyTheme(savedTheme, savedDarkMode);
            } else {
                // Fallback if applyTheme not available yet
                if (document.documentElement) {
                    document.documentElement.setAttribute('data-theme', savedTheme);
                }
                if (document.body) {
                    document.body.setAttribute('data-theme', savedTheme);
                    // CRITICAL: Ensure body--dark class is set correctly
                    if (savedDarkMode) {
                        document.body.classList.add('body--dark');
                    } else {
                        document.body.classList.remove('body--dark');
                    }
                    
                    // Apply colors immediately
                    if (window.themeColors && window.themeColors[savedTheme]) {
                        const colors = window.themeColors[savedTheme];
                        const bgColor = savedDarkMode ? colors['--bg-0'] : colors['--bg-light-0'];
                        const bg1Color = savedDarkMode ? colors['--bg-1'] : colors['--bg-light-1'];
                        const fgColor = savedDarkMode ? colors['--fg-0'] : colors['--fg-light-0'];
                        const accentBlue = colors['--accent-blue'];
                        
                        document.body.style.backgroundColor = bgColor;
                        document.body.style.color = fgColor;
                        
                        const drawer = document.querySelector('.left-drawer');
                        if (drawer) {
                            drawer.style.backgroundColor = bg1Color;
                            drawer.style.borderRightColor = accentBlue;
                        }
                        const header = document.querySelector('.app-header');
                        if (header) {
                            header.style.backgroundColor = bg1Color;
                            header.style.borderBottomColor = accentBlue;
                        }
                        const footer = document.querySelector('.terminal-footer');
                        if (footer) {
                            footer.style.backgroundColor = bg1Color;
                            footer.style.borderTopColor = accentBlue;
                        }
                    }
                }
                if (window.updateThemeStyle) {
                    window.updateThemeStyle(savedTheme);
                }
            }
        }
        
        // Helper functions for layout
        function updatePageContentHeight() {
            const footerEl = document.querySelector('.terminal-footer');
            const footerHeight = footerEl ? footerEl.offsetHeight : 160;
            // Expose footer height to CSS so .page-content can anchor to it
            document.documentElement.style.setProperty('--footer-height', footerHeight + 'px');

            const pageContent = document.querySelectorAll('.page-content');
            pageContent.forEach(el => {
                if (!el) return;
                // Anchor between header and footer using fixed positioning (CSS controls left/right)
                el.style.position = 'fixed';
                el.style.top = '60px';
                el.style.bottom = footerHeight + 'px';
                el.style.height = '';
                el.style.maxHeight = '';
                el.style.paddingBottom = '0';
                el.style.marginBottom = '0';
                el.style.overflow = 'hidden';
            });

            // Ensure chat-container scrolls properly within the fixed area
            const chatContainers = document.querySelectorAll('.chat-container');
            chatContainers.forEach(el => {
                if (!el) return;
                el.style.overflowY = 'auto';
                el.style.overflowX = 'hidden';
                el.style.flex = '1 1 0';
                el.style.minHeight = '0';
                el.style.maxHeight = '100%';
                el.style.marginTop = '0';
                el.style.marginBottom = '0';
                el.style.paddingBottom = '0';
            });
        }
        
        function updateLayoutForSidebar(width) {
            // Drive layout via CSS variable so width adjusts fluidly
            document.documentElement.style.setProperty('--sidebar-width', width + 'px');
            // Ensure page content uses left/right anchoring (no padding-left)
            const pageContent = document.querySelectorAll('.page-content');
            pageContent.forEach(el => { if (el) { el.style.paddingLeft = '0'; } });
            // Footer still needs left padding so its content clears the drawer
            const footerElements = document.querySelectorAll('.terminal-footer');
            footerElements.forEach(el => { if (el) { el.style.paddingLeft = width + 'px'; } });
        }

        // Collapsible sidebar controls
        window.setSidebarCollapsed = function(collapsed) {
            try {
                localStorage.setItem('sidebar-collapsed', collapsed ? 'true' : 'false');
                const drawer = document.querySelector('.left-drawer');
                const savedWidth = parseInt(localStorage.getItem('sidebar-width') || '260');
                const width = collapsed ? 0 : Math.min(Math.max(savedWidth || 260, 200), 500);
                if (drawer) {
                    if (collapsed) {
                        drawer.classList.add('collapsed');
                        document.body.classList.add('sidebar-collapsed');
                    } else {
                        drawer.classList.remove('collapsed');
                        drawer.style.width = width + 'px';
                        document.body.classList.remove('sidebar-collapsed');
                    }
                }
                updateLayoutForSidebar(width);
            } catch (e) { console.error(e); }
        };

        window.toggleSidebarCollapse = function() {
            const collapsed = localStorage.getItem('sidebar-collapsed') === 'true';
            window.setSidebarCollapsed(!collapsed);
        };

        // Auto-scroll helpers
        function scrollChatToBottom() {
            const chats = document.querySelectorAll('.chat-container');
            chats.forEach(el => {
                try { el.scrollTop = el.scrollHeight; } catch (e) {}
            });
        }

        function setupChatAutoScroll() {
            const chats = document.querySelectorAll('.chat-container');
            chats.forEach(el => {
                if (el.__autoScrollObserver) return;
                const obs = new MutationObserver(function(mutations) {
                    for (const m of mutations) {
                        if (m.type === 'childList' && m.addedNodes && m.addedNodes.length) {
                            scrollChatToBottom();
                            break;
                        }
                    }
                });
                obs.observe(el, { childList: true });
                el.__autoScrollObserver = obs;
            });
        }
        
        // CRITICAL FIX #6: Single consolidated DOMContentLoaded handler
        let initRan = false;
        function initOnReady() {
            if (initRan) return; // Prevent duplicate execution
            initRan = true;
            
            // Initialize theme
            initializeTheme();
            
            // Initialize sidebar width from localStorage
            const savedSidebarWidth = localStorage.getItem('sidebar-width');
            const initialCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';
            if (savedSidebarWidth) {
                const drawer = document.querySelector('.left-drawer');
                if (drawer) {
                    if (initialCollapsed) {
                        drawer.classList.add('collapsed');
                        document.body.classList.add('sidebar-collapsed');
                        updateLayoutForSidebar(0);
                    } else {
                        drawer.style.width = savedSidebarWidth + 'px';
                        document.body.classList.remove('sidebar-collapsed');
                        updateLayoutForSidebar(parseInt(savedSidebarWidth));
                    }
                } else {
                    // Wait for drawer to appear
                    const drawerObserver = new MutationObserver(function(mutations, obs) {
                        const drawer = document.querySelector('.left-drawer');
                        if (drawer) {
                            if (initialCollapsed) {
                                drawer.classList.add('collapsed');
                                document.body.classList.add('sidebar-collapsed');
                                updateLayoutForSidebar(0);
                            } else {
                                drawer.style.width = savedSidebarWidth + 'px';
                                document.body.classList.remove('sidebar-collapsed');
                                updateLayoutForSidebar(parseInt(savedSidebarWidth));
                            }
                            obs.disconnect();
                        }
                    });
                    drawerObserver.observe(document.body, { childList: true, subtree: true });
                }
            } else {
                // No saved width; still apply collapsed or default width to layout
                if (initialCollapsed) {
                    document.body.classList.add('sidebar-collapsed');
                    updateLayoutForSidebar(0);
                } else {
                    document.body.classList.remove('sidebar-collapsed');
                    updateLayoutForSidebar(260);
                }
            }
            
            // Initialize footer height and page layout
            const savedFooterHeight = localStorage.getItem('footer-height');
            const footerHeight = savedFooterHeight ? parseInt(savedFooterHeight) : 160;
            let footerEl = document.querySelector('.terminal-footer');
            if (footerEl) {
                // Ensure footer is visible and positioned correctly
                footerEl.style.position = 'fixed';
                footerEl.style.bottom = '0';
                footerEl.style.left = '0';
                footerEl.style.right = '0';
                footerEl.style.zIndex = '9999';
                footerEl.style.visibility = 'visible';
                footerEl.style.height = footerHeight + 'px';
                updatePageContentHeight();
            } else {
                // Wait for footer to appear
                const footerObserver = new MutationObserver(function(mutations, obs) {
                    footerEl = document.querySelector('.terminal-footer');
                    if (footerEl) {
                        // Ensure footer is visible and positioned correctly
                        footerEl.style.position = 'fixed';
                        footerEl.style.bottom = '0';
                        footerEl.style.left = '0';
                        footerEl.style.right = '0';
                        footerEl.style.zIndex = '9999';
                        footerEl.style.visibility = 'visible';
                        footerEl.style.height = footerHeight + 'px';
                        updatePageContentHeight();
                        // Set up resize observer once footer appears
                        const footerResizeObserver = new MutationObserver(function() {
                            updatePageContentHeight();
                        });
                        footerResizeObserver.observe(footerEl, {
                            attributes: true,
                            attributeFilter: ['style', 'height']
                        });
                        obs.disconnect();
                    }
                });
                footerObserver.observe(document.body, { childList: true, subtree: true });
            }
            
            // Listen for theme changes and sync both systems
            const themeObserver = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'attributes') {
                        if (mutation.attributeName === 'data-theme') {
                            const theme = document.body.getAttribute('data-theme');
                            localStorage.setItem('ui-theme', theme);
                            if (window.updateThemeStyle) {
                                window.updateThemeStyle(theme);
                            }
                        }
                        // Sync body--dark changes
                        if (mutation.attributeName === 'class') {
                            const isDark = document.body.classList.contains('body--dark');
                            localStorage.setItem('theme', isDark ? 'dark' : 'light');
                        }
                    }
                });
            });
            themeObserver.observe(document.body, { 
                attributes: true, 
                attributeFilter: ['data-theme', 'class'] 
            });

            // Resize handling: keep chat area responsive to window size
            let resizeTimeout;
            window.addEventListener('resize', function() {
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(function() {
                    updatePageContentHeight();
                    scrollChatToBottom();
                }, 50);
            });

            // Initialize chat autoscroll and ensure it starts at bottom
            setupChatAutoScroll();
            scrollChatToBottom();
            
            // Watch for footer resizing (only if footer already exists)
            if (footerEl) {
                const footerResizeObserver = new MutationObserver(function() {
                    updatePageContentHeight();
                });
                footerResizeObserver.observe(footerEl, {
                    attributes: true,
                    attributeFilter: ['style', 'height']
                });
            }
            
            // CRITICAL FIX #8: Improved navigation interception with debouncing
            const originalPushState = history.pushState;
            const originalReplaceState = history.replaceState;
            let navigationThemeTimeout;
            
            function applyThemeOnNavigation() {
                clearTimeout(navigationThemeTimeout);
                // Apply theme immediately before navigation
                initializeTheme();
                // Then apply again after DOM updates with debouncing
                navigationThemeTimeout = setTimeout(function() {
                    initializeTheme();
                    updatePageContentHeight();
                    scrollChatToBottom();
                }, 10);
            }
            
            history.pushState = function() {
                applyThemeOnNavigation();
                const result = originalPushState.apply(history, arguments);
                // Apply after a microtask to catch NiceGUI's DOM replacement
                Promise.resolve().then(function() {
                    initializeTheme();
                    updatePageContentHeight();
                    scrollChatToBottom();
                });
                return result;
            };
            
            history.replaceState = function() {
                applyThemeOnNavigation();
                const result = originalReplaceState.apply(history, arguments);
                // Apply after a microtask to catch NiceGUI's DOM replacement
                Promise.resolve().then(function() {
                    initializeTheme();
                    updatePageContentHeight();
                    scrollChatToBottom();
                });
                return result;
            };
            
            // Listen for navigation events
            window.addEventListener('popstate', function() {
                initializeTheme();
                updatePageContentHeight();
                scrollChatToBottom();
            });
            
            // CRITICAL FIX #7: Watch for #q-app changes (NiceGUI's root element) with debouncing
            // Only apply theme if it actually changed to avoid redundant calls
            let qAppThemeTimeout;
            let lastAppliedTheme = localStorage.getItem('ui-theme') || 'everforest';
            let lastAppliedDarkMode = localStorage.getItem('theme') === 'dark';
            
            const qAppObserver = new MutationObserver(function() {
                const currentTheme = localStorage.getItem('ui-theme') || 'everforest';
                const currentDarkMode = localStorage.getItem('theme') === 'dark';
                // Only apply if theme actually changed
                if (currentTheme !== lastAppliedTheme || currentDarkMode !== lastAppliedDarkMode) {
                    clearTimeout(qAppThemeTimeout);
                    qAppThemeTimeout = setTimeout(function() {
                        initializeTheme();
                        updatePageContentHeight();
                        setupChatAutoScroll();
                        scrollChatToBottom();
                        lastAppliedTheme = currentTheme;
                        lastAppliedDarkMode = currentDarkMode;
                    }, 10);
                }
            });
            
            const qApp = document.getElementById('q-app');
            if (qApp) {
                qAppObserver.observe(qApp, {
                    childList: true,
                    subtree: true
                });
            } else {
                // Wait for q-app to appear
                const qAppWaiter = new MutationObserver(function(mutations, obs) {
                    const qApp = document.getElementById('q-app');
                    if (qApp) {
                        qAppObserver.observe(qApp, {
                            childList: true,
                            subtree: true
                        });
                        obs.disconnect();
                    }
                });
                qAppWaiter.observe(document.body, { childList: true, subtree: true });
            }
            
            // Handle sidebar resizing
            let sidebarResizing = false;
            let sidebarStartX = 0;
            let sidebarStartWidth = 0;
            
            const drawer = document.querySelector('.left-drawer');
            if (drawer) {
                // Create resize handle if it doesn't exist
                let resizeHandle = drawer.querySelector('.sidebar-resize-handle');
                if (!resizeHandle) {
                    resizeHandle = document.createElement('div');
                    resizeHandle.className = 'sidebar-resize-handle';
                    drawer.style.position = 'relative';
                    drawer.appendChild(resizeHandle);
                }
                
                resizeHandle.addEventListener('mousedown', function(e) {
                    sidebarResizing = true;
                    sidebarStartX = e.clientX;
                    sidebarStartWidth = drawer.offsetWidth;
                    resizeHandle.classList.add('active');
                    e.preventDefault();
                    e.stopPropagation();
                });
                
                document.addEventListener('mousemove', function(e) {
                    if (sidebarResizing) {
                        const newWidth = sidebarStartWidth + (e.clientX - sidebarStartX);
                        if (newWidth >= 200 && newWidth <= 500) {
                            drawer.style.width = newWidth + 'px';
                            updateLayoutForSidebar(newWidth);
                            localStorage.setItem('sidebar-width', newWidth);
                        }
                    }
                });
                
                document.addEventListener('mouseup', function() {
                    if (sidebarResizing) {
                        sidebarResizing = false;
                        resizeHandle.classList.remove('active');
                    }
                });
            }
            
            // Handle footer resizing
            let footerResizing = false;
            let footerStartY = 0;
            let footerStartHeight = 0;
            
            // Reuse footerEl from above if it exists, otherwise query again
            if (!footerEl) {
                footerEl = document.querySelector('.terminal-footer');
            }
            if (footerEl) {
                let footerResizeHandle = footerEl.querySelector('.footer-resize-handle');
                if (!footerResizeHandle) {
                    footerResizeHandle = document.createElement('div');
                    footerResizeHandle.className = 'footer-resize-handle';
                    footerEl.style.position = 'relative';
                    footerEl.appendChild(footerResizeHandle);
                }
                
                footerResizeHandle.addEventListener('mousedown', function(e) {
                    footerResizing = true;
                    footerStartY = e.clientY;
                    footerStartHeight = footerEl.offsetHeight;
                    footerResizeHandle.classList.add('active');
                    e.preventDefault();
                    e.stopPropagation();
                });
                
                document.addEventListener('mousemove', function(e) {
                    if (footerResizing) {
                        const newHeight = footerStartHeight - (e.clientY - footerStartY);
                        if (newHeight >= 120 && newHeight <= 400) {
                            footerEl.style.height = newHeight + 'px';
                            localStorage.setItem('footer-height', newHeight);
                            updatePageContentHeight();
                        }
                    }
                });
                
                document.addEventListener('mouseup', function() {
                    if (footerResizing) {
                        footerResizing = false;
                        footerResizeHandle.classList.remove('active');
                    }
                });
            }
        }
        
        // Run initialization when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initOnReady);
        } else {
            initOnReady();
        }
    </script>
    ''')
    ui.add_head_html('''<meta name="description" content="easyAGI augmented generative intelligence for LLM">''')
    ui.add_head_html('''<meta name="keywords" content="EasyAGI Augmented Generative Intelligence">''')
    ui.add_head_html('''<meta name="author" content="Gregory L. Magnusson">''')
    ui.add_head_html('''<meta name="license" content="MIT">''')
    ui.add_head_html('<link rel="icon" type="image/x-icon" href="/gfx/fav/favicon.ico">')
    ui.add_head_html('''<meta name="viewport" content="width=device-width, initial-scale=1.0">''')
    ui.add_head_html('<link rel="apple-touch-icon" sizes="180x180" href="/gfx/fav/apple-touch-icon.png">')
    ui.add_head_html('<link rel="icon" type="image/png" sizes="32x32" href="/gfx/fav/favicon-32x32.png">')
    ui.add_head_html('<link rel="icon" type="image/png" sizes="16x16" href="/gfx/fav/favicon-16x16.png">')
    ui.add_head_html('<link rel="manifest" href="/site.webmanifest">')
