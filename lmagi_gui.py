#!/usr/bin/env python3
# lmagi_gui.py - Main entrypoint for lmagi application
# lmagi (c) Gregory L. Magnusson MIT license 2024
# 
# This is the preferred way to launch the application.
# Provides a native desktop application with embedded web view.
# Launches lmagi.py backend server in a subprocess.

import sys
import os
import signal
import logging
import subprocess
import threading
import time
import tempfile
from PyQt6.QtCore import QUrl, Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, QEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout, QWidget, QLabel, QSizePolicy, QGraphicsDropShadowEffect
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QMovie, QFont

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LmagiGUI(QMainWindow):
    """Main GUI window for lmagi application"""

    def __init__(self):
        super().__init__()
        self.server_process = None
        self.server_ready = False
        self.init_ui()
        self.start_backend()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('lmagi - easyAGI')
        # Larger default window size for better usability
        self.setGeometry(100, 100, 1600, 1000)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Get video path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        splash_video_path = os.path.join(base_dir, 'gfx', 'splash.mp4')
        logger.info(f"Splash video path: {splash_video_path}")
        
        # Create splash screen with native video player
        self.create_splash_screen(layout, splash_video_path)
        
        # Create web view (hidden initially, shown after splash)
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("about:blank"))
        self.browser.hide()  # Hide until backend is ready
        layout.addWidget(self.browser)

    def create_splash_screen(self, parent_layout, video_path):
        """Create splash screen with video background and overlay elements"""
        # Create a container widget for the splash screen
        splash_container = QWidget()
        splash_layout = QVBoxLayout(splash_container)
        splash_layout.setContentsMargins(0, 0, 0, 0)
        splash_layout.setSpacing(0)
        
        # Title - at the top
        self.title_label = QLabel("🧠 lmagi")
        title_font = QFont()
        title_font.setPointSize(48)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            color: white; 
            padding: 20px; 
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
        """)
        # Add drop shadow effect for better visibility
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(Qt.GlobalColor.black)
        shadow.setOffset(2, 2)
        self.title_label.setGraphicsEffect(shadow)
        splash_layout.addWidget(self.title_label)
        
        # Video player - fill available space but leave room for definitions
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        # Make video fill the space but maintain aspect ratio
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setSizePolicy(size_policy)
        # Set aspect ratio mode to maintain aspect ratio but scale to fit
        try:
            self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        except AttributeError:
            # If setAspectRatioMode doesn't exist, will use default behavior
            pass
        
        # Add video with stretch factor (takes most space but leaves room for definitions)
        splash_layout.addWidget(self.video_widget, stretch=2)
        
        # Definitions container - at the bottom with fixed size
        self.definitions_label = QLabel()
        self.definitions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.definitions_label.setStyleSheet("""
            color: white; 
            font-size: 14px; 
            padding: 15px 20px;
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            line-height: 1.5;
        """)
        self.definitions_label.setWordWrap(True)
        # Set fixed height to prevent resizing - increased to accommodate longer definitions
        self.definitions_label.setFixedHeight(200)
        self.definitions_label.setMinimumHeight(200)
        self.definitions_label.setMaximumHeight(200)
        # Add drop shadow effect for better visibility
        def_shadow = QGraphicsDropShadowEffect()
        def_shadow.setBlurRadius(10)
        def_shadow.setColor(Qt.GlobalColor.black)
        def_shadow.setOffset(2, 2)
        self.definitions_label.setGraphicsEffect(def_shadow)
        splash_layout.addWidget(self.definitions_label)
        
        # Store reference to splash container
        self.splash_container = splash_container
        
        # Create fade animation for definitions
        self.fade_animation = QPropertyAnimation(self.definitions_label, b"windowOpacity")
        self.fade_animation.setDuration(1000)  # 1 second fade in/out
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Add splash container to parent layout
        parent_layout.addWidget(splash_container)
        
        # Media player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0)  # Muted
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        self.media_player.setLoops(2)  # Loop exactly 2 times (not infinite)
        
        # Start video playback
        self.media_player.play()
        logger.info(f"Video playback started: {video_path}")
        
        # Video is set to loop exactly 2 times (set earlier)
        logger.info(f"Video loops setting: {self.media_player.loops()} (should be 2)")
        
        # Setup definition cycling
        self.current_definition = 0
        self.definition_index = 0
        self.definitions = [
            {
                'word': 'aug·ment·ed',
                'phonetic': '/ôɡˈmen(t)əd/',
                'type': 'adjective: augmented',
                'meaning': 'having been made greater in size or value'
            },
            {
                'word': 'gen·er·a·tive',
                'phonetic': '/ˈjen(ə)rədiv,ˈjenəˌrādiv/',
                'type': 'adjective: generative',
                'meaning': 'denoting an approach to any field of linguistics that involves applying a finite set of rules to linguistic input in order to produce all and only the well-formed items of a language\n\nrelating to or capable of production or reproduction'
            },
            {
                'word': 'in·tel·li·gence',
                'phonetic': '/inˈteləj(ə)ns/',
                'type': 'noun: intelligence',
                'meaning': 'the ability to acquire and apply knowledge and skills'
            }
        ]
        
        # Wait for video to complete at least TWO cycles before starting definitions
        # Using timer-based approach instead of position tracking for reliability
        self.video_cycle_count = 0  # For debug logging only
        self.target_cycles = 2  # Loop video twice before showing definitions
        self._definitions_started = False  # Flag to prevent multiple starts
        self.media_player.positionChanged.connect(self.on_video_position_changed)
        self.media_player.durationChanged.connect(self.on_video_duration_changed)
        
        # Fallback: start definitions after estimated two cycles + buffer
        # Will be updated when video duration is known
        # Use a longer initial timeout to ensure we wait
        self.fallback_timer = QTimer()
        self.fallback_timer.setSingleShot(True)
        self.fallback_timer.timeout.connect(self.start_definitions)
        self.fallback_timer.start(30000)  # 30 second fallback (will be updated when duration known)
        logger.info("Initial fallback timer set to 30 seconds")
        
        # Set background to black
        central_widget = self.centralWidget()
        central_widget.setStyleSheet("background-color: black;")
        
        # Center window on screen
        self.center_on_screen()

    def on_video_duration_changed(self, duration):
        """Called when video duration is known"""
        self.video_duration = duration
        logger.info(f"Video duration: {duration}ms ({duration/1000:.2f} seconds)")
        
        # Cancel any existing timers
        if hasattr(self, 'cycle_timer') and self.cycle_timer.isActive():
            self.cycle_timer.stop()
            logger.info("Stopped existing cycle timer")
        if hasattr(self, 'fallback_timer') and self.fallback_timer.isActive():
            self.fallback_timer.stop()
            logger.info("Stopped existing fallback timer")
        
        # Use timer-based approach: start definitions immediately, video plays for 2 cycles total
        # Duration is in milliseconds
        if duration > 0:
            # Calculate timing: all 3 definitions should cycle during 2 video cycles
            # Video total time: duration * 2
            # 3 definitions to show: divide time by 3
            # Each definition: fade in (1s) + display + fade out (1s)
            video_total_ms = duration * 2
            time_per_definition_ms = video_total_ms / 3  # Divide 2-cycle time by 3 definitions
            display_time_ms = time_per_definition_ms - 2000  # Subtract fade in/out time (1s each = 2s total)
            
            self.definition_display_time = max(int(display_time_ms), 2000)  # At least 2 seconds display
            definition_cycle_time = int(time_per_definition_ms)
            
            logger.info(f"Video: {duration/1000:.2f}s per cycle, {video_total_ms/1000:.2f}s total for 2 cycles")
            logger.info(f"Each definition: {definition_cycle_time/1000:.2f}s total (fade 1s + display {self.definition_display_time/1000:.2f}s + fade 1s)")
            
            # Start definitions immediately so they show during video
            # Store video duration for use in start_definitions
            self.video_duration = duration
            
            # Start definitions immediately (don't wait)
            def start_definitions_safe():
                try:
                    logger.info("Starting definitions...")
                    self.start_definitions()
                except Exception as e:
                    logger.error(f"Error starting definitions: {e}", exc_info=True)
            
            QTimer.singleShot(100, start_definitions_safe)
            
            # Stop video after exactly 2 cycles
            video_stop_time_ms = int(duration * 2 + 500)  # 2 cycles + small buffer
            self.video_stop_timer = QTimer()
            self.video_stop_timer.setSingleShot(True)
            self.video_stop_timer.timeout.connect(self.stop_video_after_2_cycles)
            self.video_stop_timer.start(video_stop_time_ms)
            logger.info(f"Video will stop after {video_stop_time_ms/1000:.2f} seconds (2 cycles)")
            
            # Update fallback timer
            fallback_time_ms = int(duration * 2.5 + 2000)  # 2.5 cycles + 2 sec buffer
            fallback_time_sec = fallback_time_ms / 1000
            self.fallback_timer.stop()
            self.fallback_timer.start(fallback_time_ms)
            logger.info(f"Fallback timer set to {fallback_time_ms}ms ({fallback_time_sec:.2f} seconds)")
        else:
            logger.error("Video duration is 0 or invalid!")
            # Fallback timing
            self.definition_display_time = 2000  # 2 seconds display

    def stop_video_after_2_cycles(self):
        """Stop video playback after exactly 2 cycles"""
        if hasattr(self, 'media_player'):
            self.media_player.stop()
            logger.info("Video stopped after 2 complete cycles")
        
        # Stop the definitions timer - definitions should only cycle during video
        if hasattr(self, 'definitions_timer') and self.definitions_timer.isActive():
            self.definitions_timer.stop()
            logger.info("Stopped definitions timer - video completed")
    
    def on_video_position_changed(self, position):
        """Called when video position changes - optional debug logging"""
        # Keep this for debug logging but use timer-based approach for cycle detection
        if hasattr(self, 'video_duration') and self.video_duration > 0:
            # Debug: log when we reach end of cycle (for verification)
            if not hasattr(self, 'last_logged_position'):
                self.last_logged_position = 0
            
            # Log when video completes a cycle (for debugging)
            if (position < 100 and self.last_logged_position > self.video_duration - 100):
                self.video_cycle_count += 1
                logger.info(f"Video loop detected - cycle {self.video_cycle_count} completed")
            
            self.last_logged_position = position

    def start_definitions(self):
        """Start cycling through definitions"""
        try:
            # Prevent multiple calls
            if hasattr(self, '_definitions_started') and self._definitions_started:
                logger.warning("start_definitions called multiple times, ignoring")
                return
            
            # Stop and cancel any timers
            if hasattr(self, 'cycle_timer') and self.cycle_timer.isActive():
                self.cycle_timer.stop()
                logger.info("Stopped cycle timer")
            if hasattr(self, 'fallback_timer') and self.fallback_timer.isActive():
                self.fallback_timer.stop()
                logger.info("Stopped fallback timer")
            
            # Mark as started
            self._definitions_started = True
            self._splash_start_time = time.time() * 1000  # Store start time in milliseconds
            
            # Log timing info
            if hasattr(self, 'video_duration') and self.video_duration > 0:
                elapsed = (self.video_duration * 2) / 1000
                logger.info(f"Starting definitions after ~{elapsed:.2f} seconds (2 video cycles)")
            else:
                logger.warning("Starting definitions but video duration was never detected!")
            
            # Show first definition with fade in
            self.show_definition(0)
            logger.info(f"Showing first definition: {self.definitions[0]['word']}")
            
            # Track how many definitions have been shown (start at 1 since we just showed the first)
            self.definitions_shown_count = 1
            
            # Cycle through definitions - timing calculated based on video duration
            # Use timing from video duration calculation if available
            if hasattr(self, 'definition_display_time') and self.definition_display_time:
                cycle_time = self.definition_display_time + 2000  # display + fade in (1s) + fade out (1s)
                logger.info(f"Using calculated cycle time from definition_display_time: {cycle_time/1000:.2f}s")
            else:
                # Fallback: calculate based on video duration if available
                if hasattr(self, 'video_duration') and self.video_duration > 0:
                    video_total_ms = self.video_duration * 2
                    cycle_time = int(video_total_ms / 3)  # Divide 2-cycle time by 3 definitions
                    logger.info(f"Using fallback cycle time from video_duration: {cycle_time/1000:.2f}s")
                else:
                    cycle_time = 4000  # Default: 4 seconds per definition
                    logger.info(f"Using default cycle time: {cycle_time/1000:.2f}s")
            
            self.definitions_timer = QTimer()
            self.definitions_timer.timeout.connect(self.cycle_definitions)
            self.definitions_timer.start(cycle_time)
            
            logger.info(f"Started definition cycling timer - will cycle every {cycle_time/1000:.2f}s")
            logger.info(f"Total definitions: {len(self.definitions)}")
            
            # After starting definitions, wait for video to complete 2 cycles total
            # The definitions should cycle during the video duration (2 cycles)
            # After video stops, wait a bit more for final definition to display, then switch
            if hasattr(self, 'video_duration') and self.video_duration > 0:
                # Video plays for 2 cycles
                video_time_ms = self.video_duration * 2
                # Add small buffer after video stops for final definition to be visible
                min_splash_time_ms = int(video_time_ms + 2000)  # 2 cycles + 2s buffer
                min_splash_time = min_splash_time_ms
                logger.info(f"Will switch to web app after {min_splash_time_ms/1000:.2f} seconds total (video: {video_time_ms/1000:.2f}s + 2s buffer)")
            else:
                min_splash_time = 15000  # 15 seconds fallback
            
            QTimer.singleShot(min_splash_time, self.check_and_switch_to_web_app)
            
            # Also set up a check that runs periodically to switch when backend is ready
            self._switch_check_timer = QTimer()
            self._switch_check_timer.timeout.connect(self.check_and_switch_to_web_app)
            self._switch_check_timer.start(1000)  # Check every second
        except Exception as e:
            logger.error(f"Error in start_definitions: {e}", exc_info=True)
            # Try to continue anyway - show error but don't crash
            try:
                self.show_error("Splash Screen Error", f"Error starting definitions: {e}")
            except:
                pass

    def show_definition(self, index):
        """Display a definition with fade in effect"""
        try:
            if index >= len(self.definitions):
                index = 0
            
            def_data = self.definitions[index]
            logger.info(f"Showing definition {index}: {def_data['word']}")
            
            # Format definition text with nicer styling
            definition_text = f"""
            <div style="text-align: center; line-height: 1.6;">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 8px; letter-spacing: 1px;">
                    {def_data['word']}
                </div>
                <div style="font-size: 13px; font-style: italic; margin-bottom: 6px; opacity: 0.85; color: #e0e0e0;">
                    {def_data['phonetic']}
                </div>
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; opacity: 0.75; color: #d0d0d0;">
                    {def_data['type']}
                </div>
                <div style="font-size: 13px; line-height: 1.5; max-width: 800px; margin: 0 auto; color: #f0f0f0;">
                    {def_data['meaning'].replace(chr(10), '<br>')}
                </div>
            </div>
            """
            
            # Set text first (hidden)
            self.definitions_label.setText(definition_text)
            self.definitions_label.setWindowOpacity(0.0)  # Start invisible
            self.definition_index = index
            
            # Stop any running animation and disconnect any handlers
            self.fade_animation.stop()
            try:
                self.fade_animation.finished.disconnect(self._fade_out_complete)
            except TypeError:
                pass  # No connection exists, that's fine
            
            # Fade in over 1 second (matches animation duration)
            self.fade_animation.setStartValue(0.0)
            self.fade_animation.setEndValue(1.0)
            self.fade_animation.start()
            logger.info(f"Started fade in animation for definition {index}")
        except Exception as e:
            logger.error(f"Error in show_definition: {e}", exc_info=True)

    def cycle_definitions(self):
        """Cycle to next definition with fade out then fade in"""
        try:
            logger.info(f"Cycling definitions - current index: {self.definition_index}, shown count: {getattr(self, 'definitions_shown_count', 0)}")
            
            # Check if we've shown all definitions once
            if hasattr(self, 'definitions_shown_count') and self.definitions_shown_count >= len(self.definitions):
                logger.info("All definitions shown once, stopping cycle timer")
                if hasattr(self, 'definitions_timer') and self.definitions_timer.isActive():
                    self.definitions_timer.stop()
                return
            
            # Increment to next definition
            self.definition_index = (self.definition_index + 1) % len(self.definitions)
            self.definitions_shown_count = getattr(self, 'definitions_shown_count', 0) + 1
            logger.info(f"Next definition index: {self.definition_index}, word: {self.definitions[self.definition_index]['word']}, shown count: {self.definitions_shown_count}")
            
            # First fade out current definition
            self.fade_animation.stop()
            
            # Disconnect finished signal if connected - use try/except to handle if not connected
            try:
                # Try to disconnect all connections first
                self.fade_animation.finished.disconnect()
            except TypeError:
                # If disconnect() fails, it means no connections exist - that's fine
                pass
            
            # Set up fade out animation
            self.fade_animation.setStartValue(1.0)
            self.fade_animation.setEndValue(0.0)
            self.fade_animation.finished.connect(self._fade_out_complete)
            self.fade_animation.start()
        except Exception as e:
            logger.error(f"Error in cycle_definitions: {e}", exc_info=True)

    def _fade_out_complete(self):
        """Called when fade out completes, then show next definition"""
        try:
            logger.info(f"Fade out complete, showing definition {self.definition_index}")
            
            # Disconnect the finished signal before connecting again
            try:
                self.fade_animation.finished.disconnect(self._fade_out_complete)
            except TypeError:
                # If disconnect fails, no connection exists - that's fine
                pass
            
            self.show_definition(self.definition_index)
        except Exception as e:
            logger.error(f"Error in _fade_out_complete: {e}", exc_info=True)

    def eventFilter(self, obj, event):
        """Handle resize events if needed"""
        # No longer needed since we're using normal layout instead of overlay
        return super().eventFilter(obj, event)

    def center_on_screen(self):
        """Center the window on the screen"""
        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        center_point = screen.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def start_backend(self):
        """Start the lmagi backend server in a separate thread"""
        def monitor_stream(stream, stream_name):
            """Monitor stdout or stderr stream"""
            try:
                for line in stream:
                    logger.info(f"Backend ({stream_name}): {line.strip()}")
                    if "NiceGUI ready to go" in line and not self.server_ready:
                        self.server_ready = True
                        # Load the web app after a short delay
                        QTimer.singleShot(500, self.load_web_app)
            except Exception as e:
                logger.error(f"Error monitoring {stream_name}: {e}")

        def run_server():
            try:
                logger.info("Starting lmagi backend server...")
                # Activate virtual environment and run lmagi.py
                # Cross-platform venv path detection
                if sys.platform == 'win32':
                    venv_python = os.path.join(os.path.dirname(__file__), 'venv', 'Scripts', 'python.exe')
                else:
                    venv_python = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python')
                
                # Fallback to system python if venv doesn't exist
                if not os.path.exists(venv_python):
                    logger.warning(f"Venv python not found at {venv_python}, trying system python")
                    venv_python = sys.executable
                
                lmagi_script = os.path.join(os.path.dirname(__file__), 'lmagi.py')

                # Set environment to prevent browser auto-open
                env = os.environ.copy()
                env['LMAGI_HEADLESS'] = '1'  # Custom flag to prevent browser opening

                self.server_process = subprocess.Popen(
                    [venv_python, lmagi_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env
                )

                # Monitor both stdout and stderr in separate threads
                stdout_thread = threading.Thread(
                    target=monitor_stream,
                    args=(self.server_process.stdout, "stdout"),
                    daemon=True
                )
                stderr_thread = threading.Thread(
                    target=monitor_stream,
                    args=(self.server_process.stderr, "stderr"),
                    daemon=True
                )

                stdout_thread.start()
                stderr_thread.start()

            except Exception as e:
                logger.error(f"Failed to start backend: {e}")
                QTimer.singleShot(100, lambda: self.show_error("Failed to start backend server", str(e)))

        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Set timeout to check if server started
        QTimer.singleShot(15000, self.check_server_timeout)

    def check_server_timeout(self):
        """Check if server failed to start within timeout"""
        if not self.server_ready:
            logger.error("Backend server failed to start within timeout")
            self.show_error(
                "Server Timeout",
                "The backend server failed to start. Please check the logs."
            )

    def load_web_app(self):
        """Load the web application in the browser"""
        logger.info("Backend ready - but waiting for splash screen to complete...")
        
        # Don't hide splash screen immediately - wait for it to complete
        # The splash screen will hide itself when ready via start_definitions
        # Just prepare the browser but keep it hidden
        self.browser.setUrl(QUrl("http://localhost:8080"))
        # Don't show browser yet - wait for splash to finish
        self.browser.hide()
        
        # Store that backend is ready
        self._backend_ready = True
        logger.info("Browser URL set, waiting for splash screen to finish...")

    def check_and_switch_to_web_app(self):
        """Check if we should switch to web app (backend ready and splash time elapsed)"""
        try:
            # Only switch if backend is ready and splash has been showing for reasonable time
            backend_ready = hasattr(self, '_backend_ready') and self._backend_ready
            splash_started = hasattr(self, '_definitions_started') and self._definitions_started
            
            if backend_ready and splash_started:
                # Check if minimum splash time has elapsed
                if hasattr(self, '_splash_start_time'):
                    elapsed_ms = (time.time() * 1000) - self._splash_start_time
                    
                    # Calculate minimum splash time based on video duration
                    if hasattr(self, 'video_duration') and self.video_duration > 0:
                        # Need: 2 video cycles + small buffer
                        video_time_ms = self.video_duration * 2
                        min_splash_time_ms = int(video_time_ms + 2000)  # 2 cycles + 2s buffer
                    else:
                        min_splash_time_ms = 15000  # 15 seconds fallback
                    
                    if elapsed_ms < min_splash_time_ms:
                        remaining_ms = min_splash_time_ms - elapsed_ms
                        logger.info(f"Splash screen: {elapsed_ms/1000:.1f}s elapsed, need {min_splash_time_ms/1000:.1f}s total. Waiting {remaining_ms/1000:.1f}s more...")
                        return  # Not enough time has passed yet
                
                # Stop the check timer
                if hasattr(self, '_switch_check_timer'):
                    self._switch_check_timer.stop()
                
                # Switch to web app
                self.switch_to_web_app()
        except Exception as e:
            logger.error(f"Error in check_and_switch_to_web_app: {e}", exc_info=True)

    def switch_to_web_app(self):
        """Switch from splash screen to web application"""
        try:
            # Prevent multiple switches
            if hasattr(self, '_switched_to_web') and self._switched_to_web:
                return
            
            self._switched_to_web = True
            logger.info("Switching from splash screen to web application")
            
            # Stop check timer if it exists
            if hasattr(self, '_switch_check_timer'):
                self._switch_check_timer.stop()
            
            # Stop definition timer if it exists
            if hasattr(self, 'definitions_timer'):
                self.definitions_timer.stop()
            
            # Hide splash screen elements
            if hasattr(self, 'splash_container'):
                self.splash_container.hide()
            if hasattr(self, 'video_widget'):
                self.video_widget.hide()
            if hasattr(self, 'definitions_label'):
                self.definitions_label.hide()
            if hasattr(self, 'title_label'):
                self.title_label.hide()
            
            # Stop video playback
            if hasattr(self, 'media_player'):
                self.media_player.stop()
                logger.info("Stopped video playback")
            
            # Show browser and load web app
            self.browser.show()
            logger.info("Browser shown, web app loaded")
        except Exception as e:
            logger.error(f"Error in switch_to_web_app: {e}", exc_info=True)
            # Try to show browser anyway
            try:
                self.browser.show()
            except:
                pass

    def show_error(self, title, message):
        """Show an error dialog"""
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event):
        """Handle window close event"""
        logger.info("Shutting down lmagi GUI...")

        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            'Confirm Exit',
            'Are you sure you want to quit lmagi?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Shutdown backend server and all child processes
            if self.server_process:
                logger.info("Terminating backend server...")
                try:
                    # Kill the entire process group
                    import psutil
                    parent = psutil.Process(self.server_process.pid)
                    for child in parent.children(recursive=True):
                        logger.info(f"Killing child process: {child.pid}")
                        child.kill()
                    parent.kill()
                    parent.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Backend didn't terminate gracefully, force killing...")
                    self.server_process.kill()
                except Exception as e:
                    logger.error(f"Error terminating backend: {e}")
                    # Force kill as fallback
                    self.server_process.kill()

            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point for the GUI application"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("lmagi")
    app.setOrganizationName("easyAGI")

    # Create and show main window
    window = LmagiGUI()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
