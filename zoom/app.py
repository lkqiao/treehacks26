import os
import requests
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")

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
    payload = {"topic": "Hackathon Demo", "type": 1, "settings": {"join_before_host": True}}
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/start-meeting", methods=["POST"])
def start_meeting():
    try:
        token = get_s2s_token()
        meeting = create_meeting(token)
        join_url = meeting.get("join_url")
        # show_chars=False: show the Zoom link first
        return render_template("index.html", join_url=join_url, show_chars=False)
    except Exception as e:
        return render_template("index.html", error=str(e))

@app.route("/choose-characters", methods=["GET"])
def choose_characters():
    # show 5 characters
    return render_template("index.html", show_chars=True, characters=CHINESE_CHARACTERS)

@app.route("/select-character", methods=["POST"])
def select_character():
    selected = request.form.get("character")
    return render_template("index.html", selected_char=selected)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
