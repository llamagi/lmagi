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
from PyQt6.QtCore import QUrl, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView

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
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create web view
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("about:blank"))
        layout.addWidget(self.browser)

        # Show loading message
        self.browser.setHtml("""
            <html>
            <head>
                <style>
                    body {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    }
                    .container {
                        text-align: center;
                        color: white;
                    }
                    h1 {
                        font-size: 3em;
                        margin-bottom: 20px;
                    }
                    .spinner {
                        width: 50px;
                        height: 50px;
                        border: 5px solid rgba(255, 255, 255, 0.3);
                        border-top: 5px solid white;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin: 20px auto;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🧠 lmagi</h1>
                    <div class="spinner"></div>
                    <p>Starting easyAGI backend...</p>
                </div>
            </body>
            </html>
        """)

        # Center window on screen
        self.center_on_screen()

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
                venv_python = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python')
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
        logger.info("Loading web application...")
        self.browser.setUrl(QUrl("http://localhost:8080"))

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
