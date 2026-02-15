#!/usr/bin/env python3
"""
Quick launcher for the Chinese Character Tutor
Run this to start the application with one command
"""

import sys
import os
import subprocess
import threading
import json
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  中文 Chinese Character Tutor 🎨                          ║
║                                                          ║
║  Learn to write Chinese with real-time feedback          ║
║  Powered by MediaPipe hand detection & DTW matching      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies():
    """Check if all required packages are installed."""
    required = ['cv2', 'mediapipe', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Run: uv sync")
        return False
    
    print("All dependencies installed")
    return True

def check_camera():
    """Check if camera is accessible."""
    import cv2
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        print("macOS: Settings → Privacy & Security → Camera → Grant access")
        print("Windows: Check camera in Device Manager")
        cap.release()
        return False
    
    cap.release()
    print("Camera accessible")
    return True

def check_character_data():
    """Check if MakeMeAHanzi graphics.txt exists and characters module loads."""
    graphics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'makemeahanzi', 'graphics.txt')
    if not os.path.exists(graphics_path):
        print("❌ makemeahanzi/graphics.txt not found")
        return False

    try:
        from characters import CHARACTER_LIST
        print(f"Character database loaded ({len(CHARACTER_LIST)} characters)")
    except Exception as e:
        print(f"❌ Failed to load characters module: {e}")
        return False

    return True

# Global state for the tutor app
tutor_app_instance = None
socketio = None
ui_state = {
    "current_character": None,
    "mode": 1,  # 1=Teaching, 2=Pinyin, 3=Translation
}

def start_websocket_server():
    """Start a Flask-SocketIO server for real-time communication."""
    global socketio
    
    flask_app = Flask(__name__)
    flask_app.config['SECRET_KEY'] = 'chinese-tutor-secret'
    CORS(flask_app)
    socketio = SocketIO(flask_app, cors_allowed_origins="*")
    
    @flask_app.route("/", methods=["GET"])
    def index():
        return "WebSocket Server Running"
    
    @socketio.on('connect')
    def handle_connect():
        print("[WebSocket] Client connected")
        emit('connection_response', {'data': 'Connected to launcher'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print("[WebSocket] Client disconnected")
    
    @socketio.on('set_character')
    def handle_set_character(data):
        """Receive character change from web app."""
        global tutor_app_instance
        character = data.get('character')
        ui_state['current_character'] = character
        print(f"[WEB] Character set to: {character}")
        
        if tutor_app_instance:
            tutor_app_instance.set_character_remote(character)
        
        emit('character_updated', {'character': character}, broadcast=True)
    
    @socketio.on('set_mode')
    def handle_set_mode(data):
        """Receive mode change from web app."""
        global tutor_app_instance
        mode = data.get('mode')
        ui_state['mode'] = mode
        mode_names = {1: "Teaching", 2: "Pinyin Recognition", 3: "English Translation"}
        print(f"[WEB] Mode set to: {mode_names.get(mode, 'Unknown')}")
        
        if tutor_app_instance:
            tutor_app_instance.set_mode_remote(mode)
        
        emit('mode_updated', {'mode': mode}, broadcast=True)
    
    @socketio.on('trigger_action')
    def handle_trigger_action(data):
        """Receive action from web app."""
        global tutor_app_instance
        action = data.get('action')
        print(f"[WEB] Action triggered: {action}")
        
        if tutor_app_instance:
            tutor_app_instance.handle_action_remote(action)
        
        emit('action_completed', {'action': action}, broadcast=True)
    
    @socketio.on('get_state')
    def handle_get_state():
        """Send current state to client."""
        emit('state_update', ui_state)
    
    print("\n" + "=" * 60)
    print("WEBSOCKET SERVER STARTED")
    print("=" * 60)
    print("Local WebSocket: ws://localhost:5001")
    print("\nTo expose to web app:")
    print("  1. Install ngrok: https://ngrok.com/download")
    print("  2. Run: ngrok http 5001")
    print("  3. Use wss://your-ngrok-url in web app settings")
    print("=" * 60 + "\n")
    
    socketio.run(flask_app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)

def main():
    print_banner()
    
    print("\nPre-launch checks...")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Camera", check_camera),
        ("Character DB", check_character_data)
    ]
    
    all_passed = True
    for name, check_fn in checks:
        try:
            if not check_fn():
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  Some checks failed. Fix issues above and try again.")
        sys.exit(1)
    
    print("\nAll systems ready!")
    print("\nLaunching Chinese Character Tutor...\n")
    print("=" * 60)
    print("KEYBOARD SHORTCUTS:")
    print("  1 - Teaching Mode")
    print("  2 - Pinyin Recognition")
    print("  3 - English Translation")
    print()
    print("  SPACE - Submit/Next     C - Clear drawing")
    print("  M - Menu                Q - Quit")
    print("=" * 60)
    print()
    
    # Start the WebSocket server in a background thread
    server_thread = threading.Thread(target=start_websocket_server, daemon=True)
    server_thread.start()
    
    # Run the main app
    try:
        from main_app import TutorApp
        global tutor_app_instance
        tutor_app_instance = TutorApp()
        tutor_app_instance.run()
    except KeyboardInterrupt:
        print("\n\nApplication closed by user.")
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
