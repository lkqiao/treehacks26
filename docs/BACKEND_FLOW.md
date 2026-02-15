# Backend flow: who talks to whom

```
┌─────────────────┐         ┌──────────────────────────────────────────┐
│  Browser        │         │  launcher.py (port 5001)                  │
│  (user)         │         │  ┌─────────────────────────────────────┐  │
│                 │  HTTP   │  │ Flask + SocketIO server             │  │
│  Opens          │ ──────► │  │  • GET /         → "WebSocket ..."  │  │
│  http://        │         │  │  • GET /api/*   → (see below)       │  │
│  localhost:5050 │         │  │  • SocketIO: set_character,         │  │
│                 │         │  │    set_mode, trigger_action         │  │
│  (page served   │         │  └──────────────┬──────────────────────┘  │
│   by zoom/app)  │         │                 │                          │
└────────┬────────┘         │                 ▼                          │
         │                  │  ┌─────────────────────────────────────┐  │
         │  SocketIO        │  │ tutor_app_instance = TutorApp()      │  │
         │  (to zoom:5050)  │  │   → set_character_remote(char)       │  │
         │                  │  │   → set_mode_remote(mode)            │  │
         ▼                  │  │   → handle_action_remote(action)     │  │
┌─────────────────┐         │  └─────────────────────────────────────┘  │
│  zoom/app.py    │  HTTP   │  (main_app.TutorApp — the OpenCV window) │
│  (port 5050)    │ ──────► │                                            │
│                 │  POST   │  Same process: WebSocket thread +         │
│  Serves         │  /api/  │  main thread running TutorApp.run()      │
│  index.html     │  ...    └──────────────────────────────────────────┘
└─────────────────┘
```

## Which file interacts with which

| File | Role | Interacts with |
|------|------|-----------------|
| **launcher.py** | Entry point. Starts WebSocket server (port 5001) in a thread and runs **main_app.TutorApp** in the main thread. | **main_app.py** (holds `tutor_app_instance` and calls its `set_character_remote`, `set_mode_remote`, `handle_action_remote`). |
| **main_app.py** | The actual tutor (camera, drawing, modes). Defines `TutorApp`. | No direct reference to launcher or zoom. It only reacts when launcher calls the three `*_remote` methods. |
| **zoom/app.py** | Web UI server (port 5050). Serves the page and tries to control the tutor. | **launcher.py** — by HTTP to `LAUNCHER_URL` (e.g. `http://localhost:5001`) at `/api/character`, `/api/mode`, `/api/action`, and `GET /health`. |

So: **launcher.py** is the only thing that talks to **main_app.py**. **zoom/app.py** talks to **launcher.py** (intended via HTTP or later WebSocket). **main_app.py** does not import or call launcher or zoom.

## Event flow (when it’s wired up)

1. User does something in the browser (e.g. picks a character).
2. Frontend sends to zoom/app (e.g. `POST /api/send-character` or SocketIO `send_character`).
3. zoom/app forwards to launcher (e.g. `POST http://localhost:5001/api/character`).
4. Launcher receives the request and calls e.g. `tutor_app_instance.set_character_remote(character)`.
5. TutorApp (main_app) updates its state and the next frame shows the new character.

## What was missing (and is fixed below)

- **main_app.py** had no `set_character_remote`, `set_mode_remote`, or `handle_action_remote`, so launcher would raise AttributeError when it tried to call them.
- **launcher.py** had no REST routes `/api/character`, `/api/mode`, `/api/action`, or `/health`, so zoom/app’s HTTP calls would 404.

Adding those in launcher and main_app makes the backend and zoom app work together.
