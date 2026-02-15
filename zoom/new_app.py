import os
import secrets
import requests
from urllib.parse import urlencode
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify
)
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# ─────────────── Flask & Socket Setup ───────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(32))
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

from dotenv import load_dotenv
load_dotenv()
# ─────────────── Zoom OAuth Config ───────────────
ZOOM_CLIENT_ID     = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_REDIRECT_URI  = os.getenv("ZOOM_REDIRECT_URI", "http://localhost:5050/zoom/callback")

ZOOM_AUTH_URL    = "https://zoom.us/oauth/authorize"
ZOOM_TOKEN_URL   = "https://zoom.us/oauth/token"
ZOOM_API_BASE    = "https://api.zoom.us/v2"
ZOOM_SCOPES      = "meeting:write:meeting user:read:user"

# ─────────────── Launcher Setup ───────────────
LAUNCHER_URL = os.getenv("LAUNCHER_URL", "http://localhost:5001")

# ─────────────── Characters & Modes ───────────────
CHARACTERS = [
    {"char": "你", "pinyin": "nǐ",    "english": "you"},
    {"char": "好", "pinyin": "hǎo",   "english": "good"},
    {"char": "学", "pinyin": "xué",   "english": "study"},
    {"char": "习", "pinyin": "xí",    "english": "practice"},
    {"char": "书", "pinyin": "shū",   "english": "book"},
]

MODES = {
    "teaching":  "Teaching",
    "pinyin":    "Pinyin Recognition",
    "english":   "English Translation",
    "free_draw": "Free Draw",
}

# ─────────────── Zoom OAuth Helpers ───────────────
def _zoom_token() -> str | None:
    return session.get("zoom_access_token")

def _zoom_post(path: str, payload: dict) -> dict:
    token = _zoom_token()
    if not token:
        raise Exception("Not logged in")
    r = requests.post(
        f"{ZOOM_API_BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def _logged_in() -> bool:
    return bool(_zoom_token())

# ─────────────── Pages ───────────────
@app.route("/")
def index():
    if not _logged_in():
        return render_template("login.html")
    return render_template(
        "index.html",
        user=session.get("zoom_user", {}),
        characters=CHARACTERS,
        modes=MODES,
        launcher_url=LAUNCHER_URL,
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ─────────────── Zoom OAuth Flow ───────────────
@app.route("/zoom/login")
def zoom_login():
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    params = {
        "response_type": "code",
        "client_id": ZOOM_CLIENT_ID,
        "redirect_uri": ZOOM_REDIRECT_URI,
        "scope": ZOOM_SCOPES,
        "state": state,
    }
    return redirect(f"{ZOOM_AUTH_URL}?{urlencode(params)}")

@app.route("/zoom/callback")
def zoom_callback():
    returned_state = request.args.get("state")
    stored_state = session.pop("oauth_state", None)
    if not stored_state or returned_state != stored_state:
        return "OAuth state mismatch", 400

    error = request.args.get("error")
    if error:
        return f"Zoom denied access: {error}", 400

    code = request.args.get("code")
    if not code:
        return "No authorization code returned", 400

    try:
        r = requests.post(
            ZOOM_TOKEN_URL,
            params={"grant_type": "authorization_code", "code": code, "redirect_uri": ZOOM_REDIRECT_URI},
            auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET),
            timeout=10,
        )
        r.raise_for_status()
        token_data = r.json()
        session["zoom_access_token"] = token_data["access_token"]
        session["zoom_refresh_token"] = token_data.get("refresh_token")
    except Exception as e:
        return f"Token exchange failed: {e}", 500

    # Optional: fetch user profile
    try:
        r = requests.get(
            f"{ZOOM_API_BASE}/users/me",
            headers={"Authorization": f"Bearer {session['zoom_access_token']}"},
            timeout=10
        )
        r.raise_for_status()
        user = r.json()
        session["zoom_user"] = {"name": f"{user.get('first_name','')} {user.get('last_name','')}".strip(),
                                "email": user.get("email","")}
    except:
        session["zoom_user"] = {"name": "Zoom User", "email": ""}

    return redirect(url_for("index"))

# ─────────────── Start Meeting ───────────────
@app.route("/start-meeting", methods=["POST"])
def start_meeting():
    if not _logged_in():
        return redirect(url_for("zoom_login"))
    try:
        meeting = _zoom_post("/users/me/meetings", {
            "topic": "Chinese Character Practice",
            "type": 1,
            "settings": {
                "join_before_host": True,
                "waiting_room": False,
                "approval_type": 0,
                "host_video": True,
                "participant_video": True
            }
        })
        return render_template("index.html", join_url=meeting.get("join_url"))
    except Exception as e:
        return render_template("index.html", error=str(e))

# ─────────────── Launcher API ───────────────
def _launcher_post(path: str, payload: dict):
    try:
        r = requests.post(f"{LAUNCHER_URL}{path}", json=payload, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

@app.route("/api/send-character", methods=["POST"])
def send_character():
    data, err = _launcher_post("/api/character", request.get_json(force=True) or {})
    if err:
        return jsonify({"status":"error","message":err}),503
    return jsonify({"status":"ok","launcher":data})

@app.route("/api/send-mode", methods=["POST"])
def send_mode():
    data, err = _launcher_post("/api/mode", request.get_json(force=True) or {})
    if err:
        return jsonify({"status":"error","message":err}),503
    return jsonify({"status":"ok","launcher":data})

@app.route("/api/send-action", methods=["POST"])
def send_action():
    data, err = _launcher_post("/api/action", request.get_json(force=True) or {})
    if err:
        return jsonify({"status":"error","message":err}),503
    return jsonify({"status":"ok","launcher":data})

@app.route("/api/launcher-status")
def launcher_status():
    try:
        r = requests.get(f"{LAUNCHER_URL}/health", timeout=2)
        r.raise_for_status()
        return jsonify({"status":"online","launcher":r.json()})
    except:
        return jsonify({"status":"offline","launcher_url":LAUNCHER_URL})

# ─────────────── WebSocket Passthrough ───────────────
@socketio.on("connect")
def ws_connect():
    emit("connection_response", {"data": "Connected"})

@socketio.on("send_character")
def ws_character(data):
    _launcher_post("/api/character", data)
    emit("character_to_launcher", data)

@socketio.on("send_mode")
def ws_mode(data):
    _launcher_post("/api/mode", data)
    emit("mode_to_launcher", data)

@socketio.on("send_action")
def ws_action(data):
    _launcher_post("/api/action", data)
    emit("action_to_launcher", data)

# ─────────────── Run App ───────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
