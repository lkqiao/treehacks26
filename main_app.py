"""
Chinese Character Tutor - Main Application

Three Modes:
1. Teaching Mode: Learn to write (shows stroke guide with animations)
2. Pinyin Recognition: See pinyin, recall character
3. English Translation: See English, recall character + gamification

Uses MakeMeAHanzi stroke data and median lines for curve-aware stroke scoring.
Each drawn segment is evaluated point-by-point against the reference median's
local tangent, and coloured green/red to give live directional feedback.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import sys
import time
import os
import requests
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from characters import (
    get_character,
    CharacterData,
    CHARACTER_LIST,
    get_random_character_info,
    makemeahanzi_to_display,
)

# ===============================
# Hand Detection Setup (Tasks API)
# ===============================

INDEX_FINGER_TIP = 8
THUMB_TIP = 4

# Pinch hysteresis: tighter threshold to start, wider to release
PINCH_ON_THRESHOLD = 0.06     # distance to START pinching (fingers close)
PINCH_OFF_THRESHOLD = 0.09    # distance to STOP pinching (must spread further apart)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker model...")
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH
        )
        print("Model downloaded.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# ===============================
# WiFi Haptic Feedback Integration
# ===============================

ESP_IP = "172.20.10.12"  # ESP8266 haptic feedback device IP
WIFI_ENABLED = True  # Toggle WiFi feedback on/off

def set_drawing_state(state: bool):
    """Send drawing state to haptic feedback device via WiFi."""
    if not WIFI_ENABLED:
        return
    
    url = f"http://{ESP_IP}/drawing"
    try:
        requests.post(url, json={"drawing": state}, timeout=2)
    except Exception as e:
        # Silently fail to not disrupt user experience
        pass

# ===============================
# Constants
# ===============================

MOVE_THRESHOLD = 5
CAMERA_INDEX = 0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
AUTO_ADVANCE_DELAY = 2.5

# --- Stroke scoring (curve-aware, from test_21) ---
MAX_DIR_ANGLE_DEG = 35       # Max angle between user and reference direction
INSTANT_WINDOW = 6           # Trailing points for instantaneous direction
MIN_DRAWN_DISP = 30.0        # Min distance in display units before evaluating
LIVE_CHECK_EVERY_N = 2       # Evaluate direction every N new points
REF_DENSE_N = 256            # Dense resampling count for reference lookup
ACCEPT_PCT = 60.0            # Min % of correct segments to accept a stroke
MIN_LENGTH_PCT = 75.0        # Min % of reference stroke length that must be drawn

# --- Colors (BGR) ---
COLOR_GUIDE_OUTLINE = (120, 120, 120)    # Gray for character outline
COLOR_COMPLETED = (0, 180, 0)            # Green for completed strokes
COLOR_CURRENT_MEDIAN = (0, 200, 255)     # Yellow for current stroke guide
COLOR_ARROW_ANIM = (255, 100, 0)         # Orange for animated stroke arrow

COL_OK = (30, 210, 30)                   # Green segment (correct direction)
COL_WRONG = (30, 30, 220)               # Red segment (wrong direction)
COL_NEUTRAL = (140, 140, 140)           # Gray segment (not yet evaluated)
COL_ARROW_EXPECTED = (255, 200, 0)      # Yellow arrow (expected direction)
COL_ARROW_USER = (255, 255, 255)        # White arrow (user direction)

COLOR_TEXT = (100, 255, 100)
COLOR_TEXT_DIM = (150, 150, 150)
COLOR_TEXT_TITLE = (150, 200, 255)

LINE_THICKNESS = 5

# --- Menu button layout ---
MENU_BUTTON_X = 40
MENU_BUTTON_W = 600
MENU_BUTTON_H = 90
MENU_BUTTON_Y_START = 160
MENU_BUTTON_SPACING = 120
MENU_LABELS = [
    ("Teaching Mode", "Learn stroke-by-stroke with guides"),
    ("Pinyin Recognition", "See pinyin, recall character"),
    ("English Translation", "See English, recall character"),
    ("Quit", "Exit application"),
]
MENU_ACTIONS = ["teaching", "pinyin", "english", "quit"]


# ===============================
# PIL Unicode Text Rendering
# ===============================

def _find_cjk_font() -> Optional[str]:
    """Find a CJK-capable TTF/TTC font on the system."""
    candidates = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei
        "C:/Windows/Fonts/msjh.ttc",       # Microsoft JhengHei
        "C:/Windows/Fonts/simsun.ttc",     # SimSun
        "C:/Windows/Fonts/simhei.ttf",     # SimHei
        "C:/Windows/Fonts/msyhbd.ttc",     # Microsoft YaHei Bold
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


CJK_FONT_PATH = _find_cjk_font()
if CJK_FONT_PATH:
    print(f"Using CJK font: {CJK_FONT_PATH}")
else:
    print("Warning: No CJK font found. Chinese characters may not render.")

_font_cache: dict = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Get a cached PIL font at the given pixel size."""
    if size not in _font_cache:
        if CJK_FONT_PATH:
            _font_cache[size] = ImageFont.truetype(CJK_FONT_PATH, size)
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def put_text(img: np.ndarray, text: str, pos: Tuple[int, int],
             font_size: int, color_bgr: Tuple[int, int, int]):
    """
    Draw Unicode text on an OpenCV BGR image (in-place).
    pos = (x, baseline_y), matching cv2.putText coordinate convention.
    """
    if not text:
        return
    font = _get_font(font_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    try:
        draw.text(pos, text, font=font, fill=color_rgb, anchor="ls")
    except Exception:
        # Fallback if anchor not supported (old Pillow)
        draw.text(pos, text, font=font, fill=color_rgb)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)


# ===============================
# Curve-Aware Geometry Helpers
# ===============================

def resample_uniform(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to *n* uniformly-spaced points by arc-length."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return (np.repeat(pts[:1], n, axis=0)
                if len(pts) else np.zeros((n, 2), dtype=np.float32))
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float32)
    j = 0
    for i, ti in enumerate(t):
        while j < len(s) - 2 and s[j + 1] < ti:
            j += 1
        t0, t1 = s[j], s[j + 1]
        a = 0.0 if (t1 - t0) < 1e-6 else (ti - t0) / (t1 - t0)
        out[i] = (1.0 - a) * pts[j] + a * pts[j + 1]
    return out


def build_tangents(pts: np.ndarray) -> np.ndarray:
    """Central-difference unit tangents for each point in *pts*."""
    n = len(pts)
    out = np.zeros_like(pts)
    for i in range(n):
        lo = max(i - 1, 0)
        hi = min(i + 1, n - 1)
        v = pts[hi] - pts[lo]
        nv = np.linalg.norm(v)
        out[i] = v / nv if nv > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)
    return out


def vec_unit(v: np.ndarray) -> Optional[np.ndarray]:
    """Unit vector, or None if magnitude is negligible."""
    n = np.linalg.norm(v)
    return None if n < 1e-8 else v / n


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    un, vn = vec_unit(u), vec_unit(v)
    if un is None or vn is None:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(un, vn), -1.0, 1.0))))


def instantaneous_direction(pts: np.ndarray, window: int) -> Optional[np.ndarray]:
    """Direction from the most recent *window* points (sum of segment vectors)."""
    pts = np.asarray(pts, dtype=np.float32)
    recent = pts[-window:] if len(pts) >= window else pts
    if len(recent) < 2:
        return None
    v = recent[1:] - recent[:-1]
    return vec_unit(v.sum(axis=0))


# ===============================
# RefStroke — precomputed reference
# ===============================

class RefStroke:
    """
    One reference median densely resampled in display space with precomputed
    tangents.  Nearest-point lookup and tangent queries happen in display space,
    so user points must be in the same coordinate system.
    """

    def __init__(self, display_pts: np.ndarray):
        """*display_pts*: median points after makemeahanzi_to_display (0-1024, y-down)."""
        self.dense = resample_uniform(display_pts, n=REF_DENSE_N)
        self.tangents = build_tangents(self.dense)
        # Total arc length of the reference median in display-space units
        diffs = np.diff(self.dense, axis=0)
        self.arc_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    def nearest(self, dx: float, dy: float) -> Tuple[int, float]:
        """Return (index, t ∈ [0,1]) of the closest dense point to (dx, dy)."""
        pt = np.array([dx, dy], dtype=np.float32)
        d = np.linalg.norm(self.dense - pt, axis=1)
        idx = int(np.argmin(d))
        t = idx / max(REF_DENSE_N - 1, 1)
        return idx, t

    def tangent_at(self, idx: int) -> np.ndarray:
        return self.tangents[idx]


# ===============================
# Per-point evaluation
# ===============================

def evaluate_point(
    new_pt: np.ndarray,
    history: np.ndarray,
    ref: RefStroke,
    total_drawn: float,
) -> Tuple[Optional[bool], float, float, np.ndarray, Optional[np.ndarray]]:
    """
    Evaluate direction at the latest drawn point (display space).

    Returns (ok, angle, t, ref_tangent, user_dir):
      ok          : True / False / None (None = not enough drawn yet)
      angle       : degrees between user direction and reference tangent
      t           : user's position on reference curve [0, 1]
      ref_tangent : expected direction unit vector (display space)
      user_dir    : actual direction unit vector (display space), or None
    """
    idx, t = ref.nearest(float(new_pt[0]), float(new_pt[1]))
    ref_tan = ref.tangent_at(idx)

    if total_drawn < MIN_DRAWN_DISP:
        return None, 0.0, t, ref_tan, None

    u_dir = instantaneous_direction(history, window=INSTANT_WINDOW)
    if u_dir is None:
        return None, 0.0, t, ref_tan, None

    ang = angle_deg(u_dir, ref_tan)
    ok = ang <= MAX_DIR_ANGLE_DEG

    return ok, ang, t, ref_tan, u_dir


# ===============================
# Application
# ===============================

class TutorApp:
    def __init__(self):
        self.mode = "mode_select"  # mode_select, teaching, pinyin, english

        # Character state
        self.char_info: Optional[dict] = None
        self.char_data: Optional[CharacterData] = None
        self.current_stroke_idx = 0
        self.ref_strokes: List[RefStroke] = []

        # Pinch state (hysteresis)
        self.pinch_active = False          # True while pinching (sticky)
        self.pinch_just_started = False    # True on the single frame pinch begins
        self.hovered_button_idx = -1       # menu button under fingertip (-1 = none)
        self.should_quit = False           # set True to exit main loop

        # Drawing state
        self.user_strokes: list = []           # completed: list of (pts_px, seg_colors)
        self.current_user_stroke: list = []    # in-progress pixel points
        self.drawing = False
        self.prev_point: Optional[Tuple[int, int]] = None

        # Per-point scoring state
        self.pts_display: list = []            # current stroke in display space
        self.seg_colors: list = []             # per-segment BGR color
        self.drawn_len = 0.0                   # arc-length in display units
        self.live_ok: Optional[bool] = None
        self.live_angle: Optional[float] = None
        self.live_ref_tan: Optional[np.ndarray] = None
        self.live_u_dir: Optional[np.ndarray] = None
        self.live_counter = 0
        self.tip_xy: Optional[Tuple[int, int]] = None

        # Display
        self.drawing_bbox = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.char_size = min(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Feedback & scoring
        self.feedback: List[str] = []
        self.score = 0
        self.completed_characters = 0
        self.character_complete = False
        self.complete_time: Optional[float] = None

        # Animation
        self.anim_start = time.time()

        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print("Error: Camera failed to initialize.")
            print("Check camera permissions in system settings.")
            sys.exit(1)

        self.frame_count = 0

    # ----------------------------
    # Coordinate helpers
    # ----------------------------

    def _compute_drawing_bbox(self, w: int, h: int):
        """Centered square bbox for character display (85 % of min dimension)."""
        size = int(min(w, h) * 0.85)
        x0 = (w - size) // 2
        y0 = (h - size) // 2
        self.drawing_bbox = (x0, y0, x0 + size, y0 + size)
        self.char_size = size

    def _pixel_to_display(self, x_px: int, y_px: int) -> Tuple[float, float]:
        """Convert pixel coords to display space (0-1024, y-down)."""
        bbox = self.drawing_bbox
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw == 0 or bh == 0:
            return 0.0, 0.0
        dx = (x_px - bbox[0]) * 1024.0 / bw
        dy = (y_px - bbox[1]) * 1024.0 / bh
        return dx, dy

    # ----------------------------
    # Character management
    # ----------------------------

    def _build_ref_strokes(self):
        """Build RefStroke objects for the current character's medians."""
        self.ref_strokes = []
        if not self.char_data:
            return
        medians = self.char_data.obj.get("medians", [])
        for median_raw in medians:
            pts_disp = makemeahanzi_to_display(
                np.array(median_raw, dtype=np.float32)
            )
            self.ref_strokes.append(RefStroke(pts_disp))

    def select_new_character(self):
        """Pick a new random character and reset all state."""
        self.char_info = get_random_character_info()
        self.char_data = get_character(self.char_info["char"])
        self._build_ref_strokes()
        self.current_stroke_idx = 0
        self.user_strokes = []
        self.current_user_stroke = []
        self._reset_live_state()
        self.character_complete = False
        self.complete_time = None
        self.feedback = []
        self.anim_start = time.time()

    def _reset_live_state(self):
        """Reset per-point scoring state for a new stroke."""
        self.pts_display = []
        self.seg_colors = []
        self.drawn_len = 0.0
        self.live_ok = None
        self.live_angle = None
        self.live_ref_tan = None
        self.live_u_dir = None
        self.live_counter = 0

    # ----------------------------
    # Stroke completion & scoring
    # ----------------------------

    def finish_stroke(self):
        """
        Complete the current stroke.
        Score by percentage of correctly-directed segments AND
        length coverage of the reference median.
        """
        if len(self.current_user_stroke) < 3:
            self.current_user_stroke = []
            self._reset_live_state()
            return

        stroke_px = self.current_user_stroke[:]
        colors = self.seg_colors[:]
        drawn_len = self.drawn_len          # save before reset
        self.current_user_stroke = []
        self._reset_live_state()

        if self.character_complete or not self.char_data:
            return
        if self.current_stroke_idx >= self.char_data.num_strokes:
            return
        if self.current_stroke_idx >= len(self.ref_strokes):
            return

        ref = self.ref_strokes[self.current_stroke_idx]

        # --- direction accuracy ---
        n_ok = sum(1 for c in colors if c == COL_OK)
        n_bad = sum(1 for c in colors if c == COL_WRONG)
        n_eval = n_ok + n_bad
        dir_pct = 100.0 * n_ok / n_eval if n_eval > 0 else 0.0

        # --- length coverage ---
        len_pct = 100.0 * drawn_len / ref.arc_length if ref.arc_length > 0 else 100.0

        accepted = dir_pct >= ACCEPT_PCT and len_pct >= MIN_LENGTH_PCT

        if accepted:
            # Stroke accepted
            self.user_strokes.append((stroke_px, colors))
            self.current_stroke_idx += 1
            self.anim_start = time.time()

            if self.current_stroke_idx >= self.char_data.num_strokes:
                self.character_complete = True
                self.complete_time = time.time()
                self.completed_characters += 1
                if self.mode == "pinyin":
                    self.score += 150
                elif self.mode == "english":
                    self.score += 200
                else:
                    self.score += 100
                self.feedback.append(
                    f"Character {self.char_info['char']} complete! +points"
                )
            else:
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx} correct! "
                    f"(dir {dir_pct:.0f}%, len {len_pct:.0f}%)"
                )
        else:
            if n_eval == 0:
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx + 1}: too short to evaluate."
                )
            else:
                reasons = []
                if dir_pct < ACCEPT_PCT:
                    reasons.append(f"dir {dir_pct:.0f}% (need {ACCEPT_PCT:.0f}%)")
                if len_pct < MIN_LENGTH_PCT:
                    reasons.append(f"len {len_pct:.0f}% (need {MIN_LENGTH_PCT:.0f}%)")
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx + 1}: {', '.join(reasons)}. "
                    f"Try again."
                )

    # ----------------------------
    # Hand detection & drawing input
    # ----------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect hand, compute pinch with hysteresis, delegate to mode handler."""
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        self._compute_drawing_bbox(w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.frame_count += 1
        timestamp_ms = self.frame_count * 33
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        prev_pinch = self.pinch_active
        self.pinch_just_started = False
        self.tip_xy = None

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            tip = hand_lms[INDEX_FINGER_TIP]
            thumb = hand_lms[THUMB_TIP]
            # Pinch location = midpoint of thumb tip and index tip
            x = int((tip.x + thumb.x) / 2 * w)
            y = int((tip.y + thumb.y) / 2 * h)
            self.tip_xy = (x, y)

            pinch_dist = np.hypot(tip.x - thumb.x, tip.y - thumb.y)

            # Hysteresis: tighter threshold to engage, wider to release
            if self.pinch_active:
                self.pinch_active = pinch_dist < PINCH_OFF_THRESHOLD
            else:
                self.pinch_active = pinch_dist < PINCH_ON_THRESHOLD

            if self.pinch_active and not prev_pinch:
                self.pinch_just_started = True
                set_drawing_state(True)  # Send haptic feedback: start drawing

            # Mode-specific input handling
            if self.mode == "mode_select":
                self._handle_menu_input(x, y)
            else:
                self._handle_drawing_input(x, y)

            # Draw hand landmarks
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
        else:
            self.pinch_active = False

        # End stroke if pinch released while drawing
        if not self.pinch_active and self.drawing:
            self.drawing = False
            self.prev_point = None
            self.finish_stroke()
            set_drawing_state(False)  # Send haptic feedback: stop drawing

        return frame

    def _handle_menu_input(self, x: int, y: int):
        """Handle hover detection and pinch-to-select on the mode_select screen."""
        self.hovered_button_idx = -1
        for i in range(len(MENU_ACTIONS)):
            by = MENU_BUTTON_Y_START + i * MENU_BUTTON_SPACING
            if (MENU_BUTTON_X <= x <= MENU_BUTTON_X + MENU_BUTTON_W
                    and by <= y <= by + MENU_BUTTON_H):
                self.hovered_button_idx = i
                break

        if self.pinch_just_started and self.hovered_button_idx >= 0:
            action = MENU_ACTIONS[self.hovered_button_idx]
            if action == "quit":
                self.should_quit = True
            else:
                self.mode = action
                self.select_new_character()

    def _handle_drawing_input(self, x: int, y: int):
        """Handle stroke drawing with pinch-to-draw (during teaching/recall modes)."""
        if self.pinch_active:
            if not self.drawing:
                # --- stroke start ---
                self.drawing = True
                self.prev_point = (x, y)
                self.current_user_stroke = [(x, y)]
                dx, dy = self._pixel_to_display(x, y)
                self.pts_display = [(dx, dy)]
                self.seg_colors = []
                self.drawn_len = 0.0
                self.live_ok = None
                self.live_angle = None
                self.live_ref_tan = None
                self.live_u_dir = None
                self.live_counter = 0
            else:
                # --- stroke continuation ---
                if self.prev_point:
                    dist = np.hypot(
                        x - self.prev_point[0], y - self.prev_point[1]
                    )
                    if dist > MOVE_THRESHOLD:
                        self.current_user_stroke.append((x, y))
                        self.prev_point = (x, y)

                        # Track in display space
                        dx, dy = self._pixel_to_display(x, y)
                        if not self.pts_display:
                            self.pts_display = [(dx, dy)]
                            return
                        prev_d = self.pts_display[-1]
                        seg_len = math.hypot(dx - prev_d[0], dy - prev_d[1])
                        self.drawn_len += seg_len
                        self.pts_display.append((dx, dy))

                        # Per-point direction check (throttled)
                        self.live_counter += 1
                        if (self.live_counter % LIVE_CHECK_EVERY_N == 0
                                and self.current_stroke_idx < len(self.ref_strokes)):
                            ok, ang, t, ref_tan, u_dir = evaluate_point(
                                new_pt=np.array([dx, dy]),
                                history=np.array(self.pts_display,
                                                 dtype=np.float32),
                                ref=self.ref_strokes[self.current_stroke_idx],
                                total_drawn=self.drawn_len,
                            )
                            self.seg_colors.append(
                                COL_NEUTRAL if ok is None
                                else (COL_OK if ok else COL_WRONG)
                            )
                            if ok is not None:
                                self.live_ok = ok
                                self.live_angle = ang
                            self.live_ref_tan = ref_tan
                            self.live_u_dir = u_dir
                        else:
                            self.seg_colors.append(
                                self.seg_colors[-1]
                                if self.seg_colors else COL_NEUTRAL
                            )

    # ----------------------------
    # Drawing helpers
    # ----------------------------

    def _draw_coloured_stroke(self, frame, pts_px, colors):
        """Draw a polyline with per-segment coloring."""
        for i in range(min(len(pts_px) - 1, len(colors))):
            p1 = (int(pts_px[i][0]), int(pts_px[i][1]))
            p2 = (int(pts_px[i + 1][0]), int(pts_px[i + 1][1]))
            cv2.line(frame, p1, p2, colors[i], LINE_THICKNESS, cv2.LINE_AA)

    def _draw_user_strokes(self, frame):
        """Draw completed strokes (with saved colors) and the in-progress stroke."""
        for stroke_px, colors in self.user_strokes:
            self._draw_coloured_stroke(frame, stroke_px, colors)

        if len(self.current_user_stroke) >= 2:
            self._draw_coloured_stroke(
                frame, self.current_user_stroke, self.seg_colors
            )

    def _draw_direction_arrows(self, frame):
        """Draw expected (yellow) and actual (white) direction arrows at fingertip."""
        if self.tip_xy is None or not self.drawing:
            return
        if self.current_stroke_idx >= len(self.ref_strokes):
            return

        ox, oy = self.tip_xy
        arrow_len_px = 40

        # Expected direction (yellow)
        if self.live_ref_tan is not None:
            dx = self.live_ref_tan[0] * arrow_len_px
            dy = self.live_ref_tan[1] * arrow_len_px
            end = (int(ox + dx), int(oy + dy))
            cv2.arrowedLine(frame, self.tip_xy, end, COL_ARROW_EXPECTED, 3,
                            cv2.LINE_AA, tipLength=0.35)

        # User direction (white)
        if self.live_u_dir is not None:
            dx = self.live_u_dir[0] * arrow_len_px
            dy = self.live_u_dir[1] * arrow_len_px
            end = (int(ox + dx), int(oy + dy))
            cv2.arrowedLine(frame, self.tip_xy, end, COL_ARROW_USER, 2,
                            cv2.LINE_AA, tipLength=0.35)

    def _draw_live_feedback(self, frame):
        """Draw live direction-angle text while drawing."""
        if not self.drawing or self.current_stroke_idx >= len(self.ref_strokes):
            return
        if self.live_ok is None:
            label = "direction: draw more ..."
            col = COL_NEUTRAL
        else:
            tag = "OK" if self.live_ok else "WRONG"
            label = (f"direction: {tag}  {self.live_angle:.1f}deg"
                     f"  (max {MAX_DIR_ANGLE_DEG}deg)")
            col = COL_OK if self.live_ok else COL_WRONG
        put_text(frame, label, (10, 30), 24, col)

    def _draw_animated_arrow(self, frame, median_pts, progress):
        """Draw an animated arrow traveling along the median polyline."""
        if len(median_pts) < 2:
            return
        pts = [(int(p[0]), int(p[1])) for p in median_pts]
        total_len = sum(
            np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )
        target_len = total_len * progress
        cumulative = 0.0
        for i in range(len(pts) - 1):
            seg_len = np.hypot(
                pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
            )
            if cumulative + seg_len >= target_len:
                alpha = ((target_len - cumulative) / seg_len
                         if seg_len > 0 else 0)
                p = (
                    int(pts[i][0] + alpha * (pts[i + 1][0] - pts[i][0])),
                    int(pts[i][1] + alpha * (pts[i + 1][1] - pts[i][1]))
                )
                dx = pts[i + 1][0] - pts[i][0]
                dy = pts[i + 1][1] - pts[i][1]
                mag = np.hypot(dx, dy)
                if mag > 0:
                    dx, dy = dx / mag, dy / mag
                else:
                    break
                arrow_len = 25
                end = (int(p[0] + arrow_len * dx), int(p[1] + arrow_len * dy))
                cv2.line(frame, p, end, COLOR_ARROW_ANIM, 3)
                angle = np.arctan2(dy, dx)
                for da in [-np.pi / 6, np.pi / 6]:
                    hx = int(end[0] - 12 * np.cos(angle + da))
                    hy = int(end[1] - 12 * np.sin(angle + da))
                    cv2.line(frame, end, (hx, hy), COLOR_ARROW_ANIM, 3)
                cv2.circle(frame, p, 8, COLOR_ARROW_ANIM, -1)
                break
            cumulative += seg_len

    def _draw_drawing_box(self, frame):
        """Faint reference box and crosshairs for the drawing area."""
        bbox = self.drawing_bbox
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        cv2.line(frame, (cx, bbox[1]), (cx, bbox[3]), (50, 50, 50), 1)
        cv2.line(frame, (bbox[0], cy), (bbox[2], cy), (50, 50, 50), 1)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (50, 50, 50), 1)

    def _draw_status_bar(self, frame, status_text):
        """Semi-transparent status bar at the bottom."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        put_text(frame, status_text, (20, h - 55), 28, COLOR_TEXT)
        if self.feedback:
            put_text(frame, self.feedback[-1], (20, h - 20), 20, COLOR_TEXT_DIM)

    def _draw_shortcuts(self, frame):
        h, w = frame.shape[:2]
        shortcuts = "C:Clear  N:Next  M:Menu  Q:Quit"
        put_text(frame, shortcuts, (w - 450, 55), 18, COLOR_TEXT_DIM)

    # ----------------------------
    # Mode renderers
    # ----------------------------

    def render_teaching_mode(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        if not self.char_data:
            self.select_new_character()

        bbox = self.drawing_bbox
        cd = self.char_data

        # Background guides
        self._draw_drawing_box(display)
        cd.draw_union(display, bbox, color=COLOR_GUIDE_OUTLINE, thickness=2)

        # Completed strokes (filled green)
        for i in range(self.current_stroke_idx):
            cd.draw_stroke(display, i, bbox, filled=True, color=COLOR_COMPLETED)

        # Current stroke median (yellow) + animated arrow
        if self.current_stroke_idx < cd.num_strokes:
            cd.draw_stroke_midline(
                display, self.current_stroke_idx, bbox,
                color=COLOR_CURRENT_MEDIAN, thickness=3
            )
            median_px = cd.get_stroke_midline(self.current_stroke_idx, bbox)
            if len(median_px) > 0:
                progress = ((time.time() - self.anim_start) % 2.0) / 2.0
                self._draw_animated_arrow(display, median_px, progress)

        # User strokes (coloured per-segment)
        self._draw_user_strokes(display)

        # Direction arrows + live feedback text (on top of everything)
        self._draw_direction_arrows(display)
        self._draw_live_feedback(display)

        # Title (below live feedback line)
        title = (
            f"Teaching: {self.char_info['char']}  "
            f"({self.char_info['pinyin']}) - {self.char_info['english']}"
        )
        put_text(display, title, (20, 55), 32, COLOR_TEXT)

        if self.character_complete:
            status = "Character complete! (auto-advancing...)"
        else:
            status = f"Stroke {self.current_stroke_idx + 1} / {cd.num_strokes}"

        self._draw_status_bar(display, status)
        self._draw_shortcuts(display)
        return display

    def render_recall_mode(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        h, w = display.shape[:2]

        if not self.char_data:
            self.select_new_character()

        if self.mode == "pinyin":
            mode_title = "Pinyin Mode"
            prompt = f"Write: {self.char_info['pinyin']}"
        else:
            mode_title = "Translation Mode"
            prompt = f"Write: {self.char_info['english'].upper()}"

        put_text(display, mode_title, (20, 55), 32, COLOR_TEXT_TITLE)
        put_text(display, prompt, (20, 95), 38, COLOR_TEXT_TITLE)
        put_text(display, f"Score: {self.score:05d}", (w - 250, 55), 32, COLOR_TEXT)

        bbox = self.drawing_bbox
        self._draw_drawing_box(display)
        self._draw_user_strokes(display)
        self._draw_direction_arrows(display)
        self._draw_live_feedback(display)

        # Reveal character on completion
        if self.character_complete:
            cd = self.char_data
            cd.draw_union(display, bbox, color=COLOR_COMPLETED, thickness=2)
            for i in range(cd.num_strokes):
                cd.draw_stroke(display, i, bbox, filled=True, color=COLOR_COMPLETED)

        if self.character_complete:
            status = (f"Correct! {self.char_info['char']} "
                      f"({self.char_info['pinyin']})")
        else:
            status = (f"Stroke {self.current_stroke_idx + 1} / "
                      f"{self.char_data.num_strokes}")

        self._draw_status_bar(display, status)
        self._draw_shortcuts(display)
        return display

    def render_mode_select(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        h, w = display.shape[:2]

        put_text(display, "Chinese Character Tutor",
                 (w // 2 - 300, 80), 60, (255, 200, 100))

        # Interactive buttons
        keys = ["1", "2", "3", "Q"]
        for i, ((title, desc), key) in enumerate(zip(MENU_LABELS, keys)):
            by = MENU_BUTTON_Y_START + i * MENU_BUTTON_SPACING
            bx = MENU_BUTTON_X
            bx2 = bx + MENU_BUTTON_W
            by2 = by + MENU_BUTTON_H
            hovered = (i == self.hovered_button_idx)

            # Button background
            if hovered:
                cv2.rectangle(display, (bx, by), (bx2, by2), (80, 80, 80), -1)
                cv2.rectangle(display, (bx, by), (bx2, by2), COLOR_TEXT_TITLE, 2)
            else:
                cv2.rectangle(display, (bx, by), (bx2, by2), (30, 30, 30), -1)
                cv2.rectangle(display, (bx, by), (bx2, by2), (70, 70, 70), 1)

            # Button text
            text_col = (255, 255, 255) if hovered else COLOR_TEXT_TITLE
            put_text(display, f"[{key}] {title}", (bx + 20, by + 38), 32, text_col)
            put_text(display, desc, (bx + 20, by + 68), 22, COLOR_TEXT_DIM)

        # Cursor at fingertip
        if self.tip_xy is not None:
            cx, cy = self.tip_xy
            col = (0, 255, 0) if self.pinch_active else (255, 255, 255)
            cv2.circle(display, (cx, cy), 15, col, 2)
            if self.pinch_active:
                cv2.circle(display, (cx, cy), 8, col, -1)

        put_text(display, "Pinch a button or press 1/2/3/Q",
                 (w // 2 - 260, h - 60), 28, COLOR_TEXT)
        return display

    # ----------------------------
    # Input handling
    # ----------------------------

    def handle_key(self, key: int) -> bool:
        if self.mode == "mode_select":
            if key == ord('1'):
                self.mode = "teaching"
                self.select_new_character()
            elif key == ord('2'):
                self.mode = "pinyin"
                self.select_new_character()
            elif key == ord('3'):
                self.mode = "english"
                self.select_new_character()
            elif key == ord('q'):
                return False
        else:
            if key == ord(' '):
                if self.character_complete:
                    self.select_new_character()
            elif key == ord('c'):
                self.current_stroke_idx = 0
                self.user_strokes = []
                self.current_user_stroke = []
                self._reset_live_state()
                self.character_complete = False
                self.complete_time = None
                self.feedback = []
                self.anim_start = time.time()
            elif key == ord('n'):
                self.select_new_character()
            elif key == ord('m'):
                self.mode = "mode_select"
            elif key == ord('q'):
                return False
        return True

    # ----------------------------
    # Main loop
    # ----------------------------

    def run(self):
        cv2.namedWindow("Chinese Character Tutor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chinese Character Tutor", WINDOW_WIDTH, WINDOW_HEIGHT)

        running = True
        while running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame = self.process_frame(frame)

            # Check if pinch-quit was triggered on the menu
            if self.should_quit:
                break

            if self.mode == "mode_select":
                display = self.render_mode_select(frame)
            elif self.mode == "teaching":
                display = self.render_teaching_mode(frame)
            elif self.mode in ("pinyin", "english"):
                display = self.render_recall_mode(frame)
            else:
                display = frame.copy()

            if self.complete_time is not None:
                if time.time() - self.complete_time >= AUTO_ADVANCE_DELAY:
                    self.select_new_character()

            cv2.imshow("Chinese Character Tutor", display)

            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                running = self.handle_key(key)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = TutorApp()
    app.run()
