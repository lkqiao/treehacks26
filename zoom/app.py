import os
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zoom-tutor-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")

# Local launcher endpoint (set via environment variable)
# Example: "wss://abcd1234.ngrok.io" or "ws://localhost:5001" for local testing
LAUNCHER_WS_URL = os.getenv("LAUNCHER_WS_URL", "ws://localhost:5001")
LAUNCHER_URL = os.getenv("LAUNCHER_URL", "http://localhost:5001")

CHINESE_CHARACTERS = ["你", "好", "学", "习", "书"]  # your 5 demo characters

def get_s2s_token():
    url = "https://zoom.us/oauth/token"
    params = {"grant_type": "account_credentials", "account_id": ACCOUNT_ID}
    auth = (CLIENT_ID, CLIENT_SECRET)
    r = requests.post(url, params=params, auth=auth)
    r.raise_for_status()
    return r.json()["access_token"]

def create_meeting(token):
    url = "https://api.zoom.us/v2/users/me/meetings"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"topic": "Language Learning", "type": 1, "settings": {"join_before_host": True, "approval_type": 0,        # automatically approve participants
        "waiting_room": False }}
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", launcher_ws_url=LAUNCHER_WS_URL)

@app.route("/start-meeting", methods=["POST"])
def start_meeting():
    try:
        token = get_s2s_token()
        meeting = create_meeting(token)
        join_url = meeting.get("join_url")
        # show_chars=False: show the Zoom link first
        return render_template("index.html", join_url=join_url, show_chars=False, launcher_ws_url=LAUNCHER_WS_URL)
    except Exception as e:
        return render_template("index.html", error=str(e))

@app.route("/choose-characters", methods=["GET"])
def choose_characters():
    # show 5 characters
    return render_template("index.html", show_chars=True, characters=CHINESE_CHARACTERS, launcher_ws_url=LAUNCHER_WS_URL)

@app.route("/select-character", methods=["POST"])
def select_character():
    selected = request.form.get("character")
    return render_template("index.html", selected_char=selected, launcher_ws_url=LAUNCHER_WS_URL)

# ===== API ENDPOINTS TO COMMUNICATE WITH LOCAL LAUNCHER =====

@app.route("/api/send-character", methods=["POST"])
def send_character():
    """Send a character to the local launcher."""
    try:
        data = request.json
        character = data.get("character")
        
        response = requests.post(
            f"{LAUNCHER_URL}/api/character",
            json={"character": character},
            timeout=5
        )
        response.raise_for_status()
        return jsonify({"status": "ok", "message": f"Sent character: {character}"})
    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "error",
            "message": "Could not reach launcher. Is it running? Set LAUNCHER_URL env var."
        }), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/send-mode", methods=["POST"])
def send_mode():
    """Change mode on the local launcher."""
    try:
        data = request.json
        mode = data.get("mode")  # 1, 2, or 3
        
        response = requests.post(
            f"{LAUNCHER_URL}/api/mode",
            json={"mode": mode},
            timeout=5
        )
        response.raise_for_status()
        mode_names = {1: "Teaching", 2: "Pinyin Recognition", 3: "English Translation"}
        return jsonify({"status": "ok", "message": f"Mode set to: {mode_names.get(mode)}"})
    except requests.exceptions.ConnectionError:
        return jsonify({"status": "error", "message": "Could not reach launcher"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/send-action", methods=["POST"])
def send_action():
    """Trigger an action on the local launcher."""
    try:
        data = request.json
        action = data.get("action")  # "submit", "clear", "next", etc
        
        response = requests.post(
            f"{LAUNCHER_URL}/api/action",
            json={"action": action},
            timeout=5
        )
        response.raise_for_status()
        return jsonify({"status": "ok", "message": f"Action triggered: {action}"})
    except requests.exceptions.ConnectionError:
        return jsonify({"status": "error", "message": "Could not reach launcher"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/launcher-status", methods=["GET"])
def launcher_status():
    """Check if the launcher is running."""
    try:
        response = requests.get(f"{LAUNCHER_URL}/health", timeout=2)
        response.raise_for_status()
        return jsonify({"status": "online", "launcher": response.json()})
    except:
        return jsonify({"status": "offline", "launcher_url": LAUNCHER_URL})

# ===== WEBSOCKET ENDPOINTS =====

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection from web client."""
    print("[WebSocket] Web client connected")
    emit('connection_response', {'data': 'Connected to web app'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print("[WebSocket] Web client disconnected")

@socketio.on('send_character')
def handle_send_character(data):
    """Forward character change to launcher via WebSocket."""
    character = data.get('character')
    print(f"[WebSocket] Sending character: {character}")
    # This would connect to launcher's WebSocket and forward the message
    # For now, also keep REST API fallback
    socketio.emit('character_to_launcher', {'character': character})

@socketio.on('send_mode')
def handle_send_mode(data):
    """Forward mode change to launcher via WebSocket."""
    mode = data.get('mode')
    print(f"[WebSocket] Sending mode: {mode}")
    socketio.emit('mode_to_launcher', {'mode': mode})

@socketio.on('send_action')
def handle_send_action(data):
    """Forward action to launcher via WebSocket."""
    action = data.get('action')
    print(f"[WebSocket] Sending action: {action}")
    socketio.emit('action_to_launcher', {'action': action})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
