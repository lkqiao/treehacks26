
"""
Chinese Character Tutor - Main Application
Three Modes:
  1. Teaching Mode: Learn to write (shows stroke guide with animations)
  2. Pinyin Recognition: See pinyin, recall character
  3. English Translation: See English, recall character + gamification
Uses MakeMeAHanzi stroke data and median lines for curve-aware stroke scoring.
Each drawn segment is evaluated point-by-point against the reference median's
local tangent, and coloured green/red to give live directional feedback.

HINT FEATURE (Pinyin / English modes):
  Press 'h' to reveal the CURRENT stroke only — it shows the same animated
  median guide used in Teaching Mode. The hint disappears as soon as you
  finish tracing (on pinch-release), regardless of whether the stroke was
  accepted or not.
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
import math
import requests
import threading
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from colour import Color
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
PINCH_ON_THRESHOLD  = 0.06   # distance to START pinching (fingers close)
PINCH_OFF_THRESHOLD = 0.09   # distance to STOP pinching (must spread further apart)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)             # Palm
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
ESP_IP        = "172.20.10.12"   # ESP8266 haptic feedback device IP
WIFI_ENABLED  = True             # Toggle WiFi feedback on/off

def set_drawing_state(state: bool):
    """Send drawing state to haptic feedback device via WiFi (non-blocking)."""
    if not WIFI_ENABLED:
        return
    def _post():
        url = f"http://{ESP_IP}/drawing"
        try:
            requests.post(url, json={"drawing": state}, timeout=2)
        except Exception:
            pass
    threading.Thread(target=_post, daemon=True).start()

# ===============================
# Constants
# ===============================
MOVE_THRESHOLD   = 5
CAMERA_INDEX     = 0
WINDOW_WIDTH     = 1280
WINDOW_HEIGHT    = 720
AUTO_ADVANCE_DELAY = 2.5

# --- Stroke scoring (curve-aware, from test_21) ---
MAX_DIR_ANGLE_DEG = 35    # Max angle between user and reference direction
INSTANT_WINDOW    = 3     # Trailing points for instantaneous direction
MIN_DRAWN_DISP    = 30.0  # Min distance in display units before evaluating
LIVE_CHECK_EVERY_N = 2    # Evaluate direction every N new points
REF_DENSE_N       = 256   # Dense resampling count for reference lookup
ACCEPT_PCT        = 60.0  # Min % of correct segments to accept a stroke
MIN_LENGTH_PCT    = 75.0  # Min % of reference stroke length that must be drawn

# --- Colors (BGR) ---
COLOR_GUIDE_OUTLINE   = (120, 120, 120)   # Gray for character outline
COLOR_COMPLETED       = (0, 180, 0)       # Green for completed strokes
COLOR_CURRENT_MEDIAN  = (0, 200, 255)     # Yellow for current stroke guide
COLOR_ARROW_ANIM      = (255, 100, 0)     # Orange for animated stroke arrow
COL_OK                = (30, 210, 30)     # Green segment (correct direction)
COL_WRONG             = (220, 30, 30)     # Red segment (wrong direction)
COL_NEUTRAL           = (30, 210, 30)     # Neutral segment (not yet evaluated)
COL_ARROW_EXPECTED    = (255, 200, 0)     # Yellow arrow (expected direction)
COL_ARROW_USER        = (255, 255, 255)   # White arrow (user direction)

# Hint stroke overlay colour (cyan-ish, distinct from teaching yellow)
COLOR_HINT_MEDIAN  = (0, 200, 255)   # same as teaching median — familiar feel
COLOR_HINT_OUTLINE = (180, 100, 255) # purple tint so it reads as "hint"

# --- Angle-to-color gradient (green → yellow → red) ---
_GRAD_COLORS = list(Color(rgb=[k/255 for k in COL_OK]).range_to(
    Color(rgb=[k/255 for k in COL_WRONG]), 181))   # 0°..180°

def angle_to_bgr(angle_deg: float) -> Tuple[int, int, int]:
    """Map an angle deviation (0°–180°) to a BGR color on a green→yellow→red gradient."""
    angle_deg = abs(angle_deg)
    angle_deg = math.sqrt(angle_deg / 180) * 180
    idx = max(0, min(180, int(round(angle_deg))))
    c = _GRAD_COLORS[idx]
    r, g, b = [int(round(x * 255)) for x in c.rgb]
    return (b, g, r)  # BGR for OpenCV

COLOR_TEXT       = (100, 220, 150)
COLOR_TEXT_DIM   = (120, 140, 160)
COLOR_TEXT_TITLE = (150, 200, 255)
LINE_THICKNESS   = 5

# --- Menu button layout ---
MENU_BUTTON_X        = 40
MENU_BUTTON_W        = 600
MENU_BUTTON_H        = 80
MENU_BUTTON_Y_START  = 130
MENU_BUTTON_SPACING  = 100
MENU_LABELS = [
    ("Teaching Mode",      "Learn stroke-by-stroke with guides"),
    ("Free Draw",          "Draw with index finger (no pinch)"),
    ("Pinyin Recognition", "See pinyin, recall character"),
    ("English Translation","See English, recall character"),
    ("Quit",               "Exit application"),
]
MENU_ACTIONS = ["teaching", "free_draw", "pinyin", "english", "quit"]

# --- In-game nav buttons (bottom-right) ---
NAV_BTN_W        = 100
NAV_BTN_H        = 45
NAV_BTN_GAP      = 12
NAV_BTN_MARGIN_R = 50
NAV_BTN_MARGIN_B = 100

# ===============================
# PIL Unicode Text Rendering
# ===============================

def _find_cjk_font() -> Optional[str]:
    """Find a CJK-capable TTF/TTC font on the system."""
    candidates = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyhbd.ttc",
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
    draw    = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    try:
        draw.text(pos, text, font=font, fill=color_rgb, anchor="ls")
    except Exception:
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
        return (np.repeat(pts[:1], n, axis=0) if len(pts)
                else np.zeros((n, 2), dtype=np.float32))
    seg   = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s     = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    t   = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float32)
    j   = 0
    for i, ti in enumerate(t):
        while j < len(s) - 2 and s[j + 1] < ti:
            j += 1
        t0, t1 = s[j], s[j + 1]
        a = 0.0 if (t1 - t0) < 1e-6 else (ti - t0) / (t1 - t0)
        out[i] = (1.0 - a) * pts[j] + a * pts[j + 1]
    return out

def build_tangents(pts: np.ndarray) -> np.ndarray:
    """Central-difference unit tangents for each point in *pts*."""
    n   = len(pts)
    out = np.zeros_like(pts)
    for i in range(n):
        lo = max(i - 1, 0)
        hi = min(i + 1, n - 1)
        v  = pts[hi] - pts[lo]
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
    pts    = np.asarray(pts, dtype=np.float32)
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
        self.dense    = resample_uniform(display_pts, n=REF_DENSE_N)
        self.tangents = build_tangents(self.dense)
        diffs = np.diff(self.dense, axis=0)
        self.arc_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    def nearest(self, dx: float, dy: float) -> Tuple[int, float]:
        """Return (index, t ∈ [0,1]) of the closest dense point to (dx, dy)."""
        pt  = np.array([dx, dy], dtype=np.float32)
        d   = np.linalg.norm(self.dense - pt, axis=1)
        idx = int(np.argmin(d))
        t   = idx / max(REF_DENSE_N - 1, 1)
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
    idx, t   = ref.nearest(float(new_pt[0]), float(new_pt[1]))
    ref_tan  = ref.tangent_at(idx)
    if total_drawn < MIN_DRAWN_DISP:
        return None, 0.0, t, ref_tan, None
    u_dir = instantaneous_direction(history, window=INSTANT_WINDOW)
    if u_dir is None:
        return None, 0.0, t, ref_tan, None
    ang = angle_deg(u_dir, ref_tan)
    ok  = ang <= MAX_DIR_ANGLE_DEG
    return ok, ang, t, ref_tan, u_dir

# ===============================
# Application
# ===============================

class TutorApp:
    def __init__(self):
        self.mode = "mode_select"   # mode_select, teaching, pinyin, english

        # Character state
        self.char_info: Optional[dict]        = None
        self.char_data: Optional[CharacterData] = None
        self.current_stroke_idx               = 0
        self.ref_strokes: List[RefStroke]     = []

        # Pinch state (hysteresis)
        self.pinch_active       = False   # True while pinching (sticky)
        self.pinch_just_started = False   # True on the single frame pinch begins
        self.hovered_button_idx = -1      # menu button under fingertip (-1 = none)
        self.hovered_nav_idx    = -1      # in-game nav button under fingertip
        self.should_quit        = False   # set True to exit main loop
        self._last_frame_shape  = (WINDOW_HEIGHT, WINDOW_WIDTH)
        self.free_draw_lost_frames = 0

        # Drawing state
        self.user_strokes: list        = []   # completed: list of (pts_px, seg_colors)
        self.current_user_stroke: list = []   # in-progress pixel points
        self.drawing                   = False
        self.prev_point: Optional[Tuple[int, int]] = None

        # Per-point scoring state
        self.pts_display: list         = []   # current stroke in display space
        self.seg_colors: list          = []   # per-segment BGR color
        self.seg_angles: list          = []   # per-segment angle deviation (degrees)
        self.drawn_len                 = 0.0  # arc-length in display units
        self.live_ok: Optional[bool]   = None
        self.live_angle: Optional[float] = None
        self.live_ref_tan: Optional[np.ndarray] = None
        self.live_u_dir: Optional[np.ndarray]   = None
        self.live_counter              = 0
        self.tip_xy: Optional[Tuple[int, int]]  = None

        # Display / bounding-box calibration
        self.drawing_bbox  = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.char_size     = min(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.calibrating   = False
        self.calib_corners: list = []
        self.custom_bbox: Optional[Tuple[int, int, int, int]] = None

        # Feedback
        self.feedback: List[str]    = []
        self.completed_characters   = 0
        self.character_complete     = False
        self.complete_time: Optional[float] = None

        # Animation
        self.anim_start = time.time()

        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print("Error: Camera failed to initialize.")
            sys.exit(1)
        ret, test_frame = self.cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            sys.exit(1)
        print(f"Camera initialized. Frame shape: {test_frame.shape}")
        self.frame_count = 0

        # -----------------------------------------------
        # Hint state
        # -----------------------------------------------
        # hint_active      : whether a hint stroke is currently visible
        # hint_stroke_idx  : which stroke index the hint is showing
        # hint_was_active  : True for the duration of the stroke where 'h' was
        #                    pressed; cleared in finish_stroke / on pinch-release
        self.hint_active     = False
        self.hint_stroke_idx = 0
        self.hint_was_active = False   # NEW: tracks that THIS stroke started with hint

    # ----------------------------
    # Coordinate helpers
    # ----------------------------

    def _compute_drawing_bbox(self, w: int, h: int):
        if self.custom_bbox is not None:
            self.drawing_bbox = self.custom_bbox
            self.char_size = min(self.custom_bbox[2] - self.custom_bbox[0],
                                 self.custom_bbox[3] - self.custom_bbox[1])
            return
        size = int(min(w, h) * 0.85)
        x0   = (w - size) // 2
        y0   = (h - size) // 2
        self.drawing_bbox = (x0, y0, x0 + size, y0 + size)
        self.char_size    = size

    def _pixel_to_display(self, x_px: int, y_px: int) -> Tuple[float, float]:
        bbox = self.drawing_bbox
        bw   = bbox[2] - bbox[0]
        bh   = bbox[3] - bbox[1]
        if bw == 0 or bh == 0:
            return 0.0, 0.0
        dx = (x_px - bbox[0]) * 1024.0 / bw
        dy = (y_px - bbox[1]) * 1024.0 / bh
        return dx, dy

    # ----------------------------
    # Character management
    # ----------------------------

    def _build_ref_strokes(self):
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
        self.char_info  = get_random_character_info()
        self.char_data  = get_character(self.char_info["char"])
        self._build_ref_strokes()
        self.current_stroke_idx   = 0
        self.user_strokes         = []
        self.current_user_stroke  = []
        self._reset_live_state()
        self.character_complete   = False
        self.complete_time        = None
        self.feedback             = []
        self.anim_start           = time.time()
        # Clear hint when moving to a new character
        self.hint_active          = False
        self.hint_stroke_idx      = 0
        self.hint_was_active      = False

    def _reset_live_state(self):
        self.pts_display  = []
        self.seg_colors   = []
        self.seg_angles   = []
        self.drawn_len    = 0.0
        self.live_ok      = None
        self.live_angle   = None
        self.live_ref_tan = None
        self.live_u_dir   = None
        self.live_counter = 0

    # ----------------------------
    # Stroke completion & scoring
    # ----------------------------

    def finish_stroke(self):
        """
        Complete the current stroke and score it.

        HINT BEHAVIOUR: if a hint was active for this stroke (hint_was_active),
        deactivate it now — regardless of whether the stroke was accepted.
        """
        if len(self.current_user_stroke) < 3:
            self.current_user_stroke = []
            self._reset_live_state()
            # Still dismiss hint even on an empty stroke attempt
            if self.hint_was_active:
                self.hint_active     = False
                self.hint_was_active = False
            return

        stroke_px  = self.current_user_stroke[:]
        colors     = self.seg_colors[:]
        angles     = self.seg_angles[:]
        drawn_len  = self.drawn_len

        self.current_user_stroke = []
        self._reset_live_state()

        # --- Dismiss hint now that the user has attempted the stroke ---
        if self.hint_was_active:
            self.hint_active     = False
            self.hint_was_active = False

        if self.character_complete or not self.char_data:
            return
        if self.current_stroke_idx >= self.char_data.num_strokes:
            return
        if self.current_stroke_idx >= len(self.ref_strokes):
            return

        ref = self.ref_strokes[self.current_stroke_idx]

        # --- direction accuracy ---
        evaluated = [a for a in angles if a is not None]
        n_eval    = len(evaluated)
        n_ok      = sum(1 for a in evaluated if a <= MAX_DIR_ANGLE_DEG)
        dir_pct   = 100.0 * n_ok / n_eval if n_eval > 0 else 0.0

        # --- length coverage ---
        len_pct   = 100.0 * drawn_len / ref.arc_length if ref.arc_length > 0 else 100.0

        accepted  = dir_pct >= ACCEPT_PCT and len_pct >= MIN_LENGTH_PCT

        if accepted:
            if self.mode == "teaching":
                self.user_strokes = []
            else:
                self.user_strokes.append((stroke_px, colors))
            self.current_stroke_idx += 1
            self.anim_start = time.time()
            if self.current_stroke_idx >= self.char_data.num_strokes:
                self.character_complete = True
                self.complete_time      = time.time()
                self.completed_characters += 1
                self.feedback.append(
                    f"Character {self.char_info['char']} complete!")
            else:
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx} correct! "
                    f"(dir {dir_pct:.0f}%, len {len_pct:.0f}%)")
        else:
            if n_eval == 0:
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx + 1}: too short to evaluate.")
            else:
                reasons = []
                if dir_pct < ACCEPT_PCT:
                    reasons.append(f"dir {dir_pct:.0f}% (need {ACCEPT_PCT:.0f}%)")
                if len_pct < MIN_LENGTH_PCT:
                    reasons.append(f"len {len_pct:.0f}% (need {MIN_LENGTH_PCT:.0f}%)")
                self.feedback.append(
                    f"Stroke {self.current_stroke_idx + 1}: {', '.join(reasons)}. "
                    f"Try again.")

    # ----------------------------
    # Hand detection & drawing input
    # ----------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        self._last_frame_shape = (h, w)
        self._compute_drawing_bbox(w, h)

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_count += 1
        timestamp_ms = self.frame_count * 33
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        prev_pinch              = self.pinch_active
        self.pinch_just_started = False
        self.tip_xy             = None

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            tip      = hand_lms[INDEX_FINGER_TIP]
            thumb    = hand_lms[THUMB_TIP]
            is_free  = (self.mode == "free_draw")

            if is_free:
                x = int(tip.x * w)
                y = int(tip.y * h)
            else:
                x = int((tip.x + thumb.x) / 2 * w)
                y = int((tip.y + thumb.y) / 2 * h)

            self.tip_xy   = (x, y)
            pinch_dist    = np.hypot(tip.x - thumb.x, tip.y - thumb.y)

            if self.pinch_active:
                self.pinch_active = pinch_dist < PINCH_OFF_THRESHOLD
            else:
                self.pinch_active = pinch_dist < PINCH_ON_THRESHOLD

            if self.pinch_active and not prev_pinch:
                self.pinch_just_started = True
                self.hint_active = False
                set_drawing_state(True)

            free_draw_active = is_free and not self.character_complete

            if self.mode == "mode_select":
                self._handle_menu_input(x, y)
            else:
                self._check_nav_hover(x, y)
                if self.hovered_nav_idx >= 0:
                    self._handle_nav_pinch()
                elif self.calibrating:
                    self._handle_calibration_pinch(x, y)
                elif free_draw_active:
                    self._handle_free_draw_input(x, y)
                else:
                    self._handle_drawing_input(x, y)

            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
        else:
            self.pinch_active    = False
            self.hovered_nav_idx = -1

        # End stroke on pinch release (normal modes)
        if not self.pinch_active and self.drawing and self.mode != "free_draw":
            self.drawing     = False
            self.prev_point  = None
            self.finish_stroke()
            set_drawing_state(False)

        # Free draw grace period
        if self.mode == "free_draw" and self.drawing:
            if result.hand_landmarks:
                self.free_draw_lost_frames = 0
            else:
                self.free_draw_lost_frames += 1
                if self.free_draw_lost_frames > 10:
                    self.drawing    = False
                    self.prev_point = None
                    self.finish_stroke()
                    self.free_draw_lost_frames = 0

        return frame

    def _handle_menu_input(self, x: int, y: int):
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
        """Handle stroke drawing with pinch-to-draw."""
        if self.pinch_active:
            if not self.drawing:
                # --- stroke start ---
                # If hint is active, mark that this stroke began with a hint.
                # The hint outline stays visible while drawing.
                if self.hint_active:
                    self.hint_was_active = True

                self.drawing    = True
                self.prev_point = (x, y)
                self.current_user_stroke  = [(x, y)]
                dx, dy = self._pixel_to_display(x, y)
                self.pts_display  = [(dx, dy)]
                self.seg_colors   = []
                self.seg_angles   = []
                self.drawn_len    = 0.0
                self.live_ok      = None
                self.live_angle   = None
                self.live_ref_tan = None
                self.live_u_dir   = None
                self.live_counter = 0
            else:
                # --- stroke continuation ---
                if self.prev_point:
                    dist = np.hypot(x - self.prev_point[0],
                                    y - self.prev_point[1])
                    if dist > MOVE_THRESHOLD:
                        self.current_user_stroke.append((x, y))
                        self.prev_point = (x, y)

                        dx, dy = self._pixel_to_display(x, y)
                        if not self.pts_display:
                            self.pts_display = [(dx, dy)]
                            return
                        prev_d  = self.pts_display[-1]
                        seg_len = math.hypot(dx - prev_d[0], dy - prev_d[1])
                        self.drawn_len += seg_len
                        self.pts_display.append((dx, dy))

                        self.live_counter += 1
                        if (self.live_counter % LIVE_CHECK_EVERY_N == 0
                                and self.current_stroke_idx < len(self.ref_strokes)):
                            ok, ang, t, ref_tan, u_dir = evaluate_point(
                                new_pt=np.array([dx, dy]),
                                history=np.array(self.pts_display, dtype=np.float32),
                                ref=self.ref_strokes[self.current_stroke_idx],
                                total_drawn=self.drawn_len,
                            )
                            if ok is None:
                                self.seg_colors.append(COL_NEUTRAL)
                                self.seg_angles.append(None)
                            else:
                                self.seg_colors.append(angle_to_bgr(ang))
                                self.seg_angles.append(ang)
                            self.live_ok      = ok
                            self.live_angle   = ang
                            self.live_ref_tan = ref_tan
                            self.live_u_dir   = u_dir
                        else:
                            self.seg_colors.append(
                                self.seg_colors[-1] if self.seg_colors else COL_NEUTRAL)
                            self.seg_angles.append(
                                self.seg_angles[-1] if self.seg_angles else None)

    def _handle_free_draw_input(self, x: int, y: int):
        """Handle stroke drawing in free_draw mode — index finger always active."""
        if not self.drawing:
            self.drawing    = True
            self.prev_point = (x, y)
            self.current_user_stroke  = [(x, y)]
            dx, dy = self._pixel_to_display(x, y)
            self.pts_display  = [(dx, dy)]
            self.seg_colors   = []
            self.seg_angles   = []
            self.drawn_len    = 0.0
            self.live_ok      = None
            self.live_angle   = None
            self.live_ref_tan = None
            self.live_u_dir   = None
            self.live_counter = 0
        else:
            if self.prev_point:
                diff = np.array([x - self.prev_point[0],
                                 y - self.prev_point[1]])
                dist = np.linalg.norm(diff)
                if dist > MOVE_THRESHOLD:
                    self.current_user_stroke.append((x, y))
                    self.prev_point = (x, y)
                    dx, dy = self._pixel_to_display(x, y)
                    if not self.pts_display:
                        self.pts_display = [(dx, dy)]
                        return
                    stroke_done = self.char_data.update_partial_stroke(
                        self.current_stroke_idx, diff)
                    if stroke_done:
                        self.drawing    = False
                        self.prev_point = None
                        self.current_user_stroke = []
                        self._reset_live_state()
                        self.current_stroke_idx += 1
                        self.anim_start = time.time()
                        if self.current_stroke_idx >= self.char_data.num_strokes:
                            self.character_complete = True
                            self.complete_time      = time.time()
                            self.completed_characters += 1
                            self.feedback.append(
                                f"Character {self.char_info['char']} complete!")
                        else:
                            self.feedback.append(
                                f"Stroke {self.current_stroke_idx} done!")
                    else:
                        self.seg_colors.append(
                            self.seg_colors[-1] if self.seg_colors else COL_NEUTRAL)
                        self.seg_angles.append(
                            self.seg_angles[-1] if self.seg_angles else None)

    # ----------------------------
    # Drawing helpers
    # ----------------------------

    def _draw_coloured_stroke(self, frame, pts_px, colors):
        for i in range(min(len(pts_px) - 1, len(colors))):
            p1 = (int(pts_px[i][0]),     int(pts_px[i][1]))
            p2 = (int(pts_px[i + 1][0]), int(pts_px[i + 1][1]))
            cv2.line(frame, p1, p2, colors[i], LINE_THICKNESS, cv2.LINE_AA)

    def _draw_user_strokes(self, frame):
        for stroke_px, colors in self.user_strokes:
            self._draw_coloured_stroke(frame, stroke_px, colors)
        if len(self.current_user_stroke) >= 2:
            self._draw_coloured_stroke(
                frame, self.current_user_stroke, self.seg_colors)

    def _draw_direction_arrows(self, frame):
        if self.tip_xy is None or not self.drawing:
            return
        if self.current_stroke_idx >= len(self.ref_strokes):
            return
        ox, oy     = self.tip_xy
        arrow_len_px = 40

        if self.live_ref_tan is not None:
            dx  = self.live_ref_tan[0] * arrow_len_px
            dy  = self.live_ref_tan[1] * arrow_len_px
            end = (int(ox + dx), int(oy + dy))
            cv2.arrowedLine(frame, self.tip_xy, end, COL_ARROW_EXPECTED, 3,
                            cv2.LINE_AA, tipLength=0.35)

        if self.live_u_dir is not None:
            dx  = self.live_u_dir[0] * arrow_len_px
            dy  = self.live_u_dir[1] * arrow_len_px
            end = (int(ox + dx), int(oy + dy))
            cv2.arrowedLine(frame, self.tip_xy, end, COL_ARROW_USER, 2,
                            cv2.LINE_AA, tipLength=0.35)

    def _draw_live_feedback(self, frame):
        if not self.drawing or self.current_stroke_idx >= len(self.ref_strokes):
            return
        if self.live_ok is None:
            label = "direction: draw more ..."
            col   = COL_NEUTRAL
        else:
            tag   = "✓ OK" if self.live_ok else "✗ WRONG"
            label = f"direction: {tag} {self.live_angle:.1f}°"
            col   = angle_to_bgr(self.live_angle)

        h, w = frame.shape[:2]
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        panel_width  = text_size[0] + 24
        panel_height = 80
        panel_x      = 15
        panel_y      = (h - panel_height) // 2
        panel_x2     = panel_x + panel_width
        panel_y2     = panel_y + panel_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x2, panel_y2), (20, 25, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame,    (panel_x, panel_y), (panel_x2, panel_y2), col, 2)

        text_x = panel_x + (panel_width  - text_size[0]) // 2
        text_y = panel_y + (panel_height + text_size[1]) // 2
        put_text(frame, label, (text_x, text_y), 20, col)

    def _draw_animated_arrow(self, frame, median_pts, progress):
        if len(median_pts) < 2:
            return
        pts       = [(int(p[0]), int(p[1])) for p in median_pts]
        total_len = sum(
            np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )
        target_len = total_len * progress
        cumulative = 0.0
        for i in range(len(pts) - 1):
            seg_len = np.hypot(pts[i + 1][0] - pts[i][0],
                               pts[i + 1][1] - pts[i][1])
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
                end       = (int(p[0] + arrow_len * dx), int(p[1] + arrow_len * dy))
                cv2.line(frame, p, end, COLOR_ARROW_ANIM, 3)
                angle = np.arctan2(dy, dx)
                for da in [-np.pi / 6, np.pi / 6]:
                    hx = int(end[0] - 12 * np.cos(angle + da))
                    hy = int(end[1] - 12 * np.sin(angle + da))
                    cv2.line(frame, end, (hx, hy), COLOR_ARROW_ANIM, 3)
                cv2.circle(frame, p, 8, COLOR_ARROW_ANIM, -1)
                break
            cumulative += seg_len

    def _draw_hint_stroke(self, frame):
        """
        Draw the hint for the current stroke in recall modes (pinyin / english).

        Renders exactly like the Teaching Mode current-stroke guide:
          • The stroke outline (semi-transparent purple tint)
          • The median guideline (cyan)
          • An animated arrow travelling along the median

        The hint is only drawn when hint_active is True AND we are NOT yet
        drawing (as soon as the user lifts the pen the hint is gone).
        We still show it while drawing so the user can trace over it.
        """
        if not self.hint_active or self.character_complete:
            return
        if self.char_data is None:
            return

        idx = self.hint_stroke_idx
        if idx >= self.char_data.num_strokes:
            return

        bbox = self.drawing_bbox

        # 1. Stroke outline with a purple tint (distinguishes hint from completed strokes)
        self.char_data.draw_stroke(
            frame, idx, bbox,
            filled=True,
            color=COLOR_HINT_OUTLINE,
        )

        # 2. Median guideline (same cyan used in teaching mode)
        self.char_data.draw_stroke_midline(
            frame, idx, bbox,
            color=COLOR_HINT_MEDIAN,
            thickness=3,
        )

        # 3. Animated arrow along the median
        median_px = self.char_data.get_stroke_midline(idx, bbox)
        if len(median_px) > 0:
            progress = ((time.time() - self.anim_start) % 2.0) / 2.0
            self._draw_animated_arrow(frame, median_px, progress)

        # 4. Small "HINT" label near the top-left of the bbox so user knows it's a hint
        lx = bbox[0] + 8
        ly = bbox[1] + 30
        put_text(frame, "HINT — trace to dismiss", (lx, ly), 20, COLOR_HINT_OUTLINE)

    def _draw_drawing_box(self, frame):
        bbox = self.drawing_bbox
        cx   = (bbox[0] + bbox[2]) // 2
        cy   = (bbox[1] + bbox[3]) // 2
        cv2.line(frame,      (cx, bbox[1]), (cx, bbox[3]), (60, 70, 90), 1)
        cv2.line(frame,      (bbox[0], cy), (bbox[2], cy), (60, 70, 90), 1)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (70, 85, 110), 1)

    def _draw_status_bar(self, frame, status_text):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (15, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, h - 90), (w, h - 90), (80, 100, 130), 2)
        put_text(frame, status_text, (20, h - 55), 28, COLOR_TEXT)
        if self.feedback:
            put_text(frame, self.feedback[-1], (20, h - 20), 20, COLOR_TEXT_DIM)

    def _draw_shortcuts(self, frame):
        h, w = frame.shape[:2]
        if self.mode in ("pinyin", "english"):
            shortcuts = "H:Hint  C:Clear  N:Next  M:Menu  Q:Quit"
        else:
            shortcuts = "C:Clear  N:Next  M:Menu  Q:Quit"
        put_text(frame, shortcuts, (w - 520, 55), 18, (100, 130, 160))

    # --- In-game nav buttons ---

    def _nav_btn_rects_from_shape(self, h: int, w: int) -> list:
        quit_x2 = w - NAV_BTN_MARGIN_R
        quit_x1 = quit_x2 - NAV_BTN_W
        btn_y2  = h - NAV_BTN_MARGIN_B
        btn_y1  = btn_y2 - NAV_BTN_H
        menu_x2 = quit_x1 - NAV_BTN_GAP
        menu_x1 = menu_x2 - NAV_BTN_W
        adj_x2  = menu_x1 - NAV_BTN_GAP
        adj_x1  = adj_x2  - NAV_BTN_W
        return [(adj_x1, btn_y1, adj_x2, btn_y2),
                (menu_x1, btn_y1, menu_x2, btn_y2),
                (quit_x1, btn_y1, quit_x2, btn_y2)]

    def _nav_btn_rects(self, frame) -> list:
        h, w = frame.shape[:2]
        return self._nav_btn_rects_from_shape(h, w)

    def _draw_nav_buttons(self, frame):
        rects  = self._nav_btn_rects(frame)
        labels = ["Adjust", "Menu", "Quit"]
        for i, ((x1, y1, x2, y2), label) in enumerate(zip(rects, labels)):
            hovered = (self.hovered_nav_idx == i)
            if i == 0 and self.calibrating:
                bg     = (60, 40, 100) if not hovered else (100, 60, 140)
                border = (200, 120, 255)
            elif hovered:
                bg     = (70, 100, 150)
                border = (150, 200, 255)
            else:
                bg     = (45, 45, 55)
                border = (100, 100, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border,
                          2 if hovered or (i == 0 and self.calibrating) else 1)
            text_col  = (255, 255, 255) if hovered else COLOR_TEXT_DIM
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            tx = x1 + (NAV_BTN_W - text_size[0]) // 2
            ty = y1 + (NAV_BTN_H + text_size[1]) // 2 + 2
            put_text(frame, label, (tx, ty), 22, text_col)

        if self.hovered_nav_idx >= 0 and self.tip_xy is not None:
            cx, cy = self.tip_xy
            col    = (0, 200, 100) if self.pinch_active else (200, 200, 200)
            cv2.circle(frame, (cx, cy), 15, col, 2)
            if self.pinch_active:
                cv2.circle(frame, (cx, cy), 8, col, -1)

    def _check_nav_hover(self, x: int, y: int):
        h, w  = self._last_frame_shape
        rects = self._nav_btn_rects_from_shape(h, w)
        self.hovered_nav_idx = -1
        for i, (x1, y1, x2, y2) in enumerate(rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.hovered_nav_idx = i
                break

    def _handle_nav_pinch(self):
        if not self.pinch_just_started:
            return
        if self.hovered_nav_idx == 0:
            self.calibrating = True
            self.calib_corners = []
            self.drawing = False
            self.current_user_stroke = []
            self._reset_live_state()
            self.feedback.append("Pinch top-left corner of drawing area...")
        elif self.hovered_nav_idx == 1:
            self.calibrating = False
            self.mode = "mode_select"
            self.drawing = False
            self.current_user_stroke = []
            self._reset_live_state()
        elif self.hovered_nav_idx == 2:
            self.should_quit = True

    def _handle_calibration_pinch(self, x: int, y: int):
        if not self.pinch_just_started:
            return
        self.calib_corners.append((x, y))
        if len(self.calib_corners) == 1:
            self.feedback.append(
                f"Corner 1 set at ({x}, {y}). Now pinch bottom-right corner...")
        elif len(self.calib_corners) >= 2:
            (x0, y0), (x1, y1) = self.calib_corners[0], self.calib_corners[1]
            bx0  = min(x0, x1)
            by0  = min(y0, y1)
            bx1  = max(x0, x1)
            by1  = max(y0, y1)
            side = min(bx1 - bx0, by1 - by0)
            if side < 50:
                self.feedback.append("Box too small — try again.")
                self.calib_corners = []
                return
            bx1 = bx0 + side
            by1 = by0 + side
            self.custom_bbox  = (bx0, by0, bx1, by1)
            self.drawing_bbox = self.custom_bbox
            self.char_size    = side
            self.calibrating  = False
            self.calib_corners = []
            self._build_ref_strokes()
            self.user_strokes        = []
            self.current_user_stroke = []
            self._reset_live_state()
            self.feedback.append(
                f"Drawing area set! ({bx0},{by0}) to ({bx1},{by1})")

    def _draw_calibration_overlay(self, frame):
        if not self.calibrating:
            return
        h, w = frame.shape[:2]
        n    = len(self.calib_corners)
        msg  = ("CALIBRATE: Pinch at TOP-LEFT corner"
                if n == 0 else "CALIBRATE: Pinch at BOTTOM-RIGHT corner")

        text_size, _  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        panel_width   = text_size[0] + 24
        panel_height  = 80
        panel_x       = 15
        feedback_y_end = (h - 80) // 2 + 80 + 20
        panel_y  = feedback_y_end
        panel_x2 = panel_x + panel_width
        panel_y2 = panel_y + panel_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x2, panel_y2), (40, 30, 60), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x2, panel_y2), (200, 120, 255), 2)

        text_x = panel_x + (panel_width  - text_size[0]) // 2
        text_y = panel_y + (panel_height + text_size[1]) // 2
        put_text(frame, msg, (text_x, text_y), 20, (200, 120, 255))

        for i, (cx, cy) in enumerate(self.calib_corners):
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 14, (255, 0, 255), 2)
            put_text(frame, f"C{i+1}", (cx + 16, cy - 4), 20, (255, 0, 255))

        if n >= 1:
            cx, cy = self.calib_corners[0]
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 0, 255), 1)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 0, 255), 1)

        if n == 1 and self.tip_xy is not None:
            cx, cy = self.calib_corners[0]
            tx, ty = self.tip_xy
            cv2.rectangle(frame, (cx, cy), (tx, ty), (255, 255, 0), 2)

    # ----------------------------
    # Mode renderers
    # ----------------------------

    def render_teaching_mode(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        if not self.char_data:
            self.select_new_character()
        bbox = self.drawing_bbox
        cd   = self.char_data

        self._draw_drawing_box(display)
        cd.draw_union(display, bbox, color=COLOR_GUIDE_OUTLINE, thickness=2)

        for i in range(self.current_stroke_idx):
            cd.draw_stroke(display, i, bbox, filled=True, color=COLOR_COMPLETED)

        if self.current_stroke_idx < cd.num_strokes:
            cd.draw_stroke_midline(display, self.current_stroke_idx, bbox,
                                   color=COLOR_CURRENT_MEDIAN, thickness=3)
            median_px = cd.get_stroke_midline(self.current_stroke_idx, bbox)
            if len(median_px) > 0:
                progress = ((time.time() - self.anim_start) % 2.0) / 2.0
                self._draw_animated_arrow(display, median_px, progress)

        self._draw_user_strokes(display)
        self._draw_direction_arrows(display)
        self._draw_live_feedback(display)

        title = (f"Teaching: {self.char_info['char']} "
                 f"({self.char_info['pinyin']}) - {self.char_info['english']}")
        put_text(display, title, (20, 55), 32, COLOR_TEXT)

        status = ("Character complete!" if self.character_complete
                  else f"Stroke {self.current_stroke_idx + 1} / {cd.num_strokes}")
        self._draw_status_bar(display, status)
        self._draw_shortcuts(display)
        self._draw_nav_buttons(display)
        self._draw_calibration_overlay(display)
        return display

    def render_free_draw_mode(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        if not self.char_data:
            self.select_new_character()
        bbox = self.drawing_bbox
        cd   = self.char_data

        self._draw_drawing_box(display)

        for i in range(self.current_stroke_idx):
            cd.draw_stroke(display, i, bbox, filled=True, color=COLOR_COMPLETED)
        cd.draw_partial_stroke(display, bbox)

        self._draw_user_strokes(display)
        self._draw_direction_arrows(display)
        self._draw_live_feedback(display)

        title = (f"Free Draw: {self.char_info['char']} "
                 f"({self.char_info['pinyin']}) - {self.char_info['english']}")
        put_text(display, title, (20, 55), 32, COLOR_TEXT)

        status = ("Character complete!" if self.character_complete
                  else f"Stroke {self.current_stroke_idx + 1} / {cd.num_strokes}")
        self._draw_status_bar(display, status)
        self._draw_shortcuts(display)
        self._draw_nav_buttons(display)
        self._draw_calibration_overlay(display)
        return display

    def render_recall_mode(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        h, w    = display.shape[:2]

        if not self.char_data:
            self.select_new_character()

        if self.mode == "pinyin":
            mode_title = "Pinyin Mode"
            prompt     = f"Write: {self.char_info['pinyin']}"
        else:
            mode_title = "Translation Mode"
            prompt     = f"Write: {self.char_info['english'].upper()}"

        put_text(display, mode_title, (20, 55),  32, COLOR_TEXT_TITLE)
        put_text(display, prompt,     (20, 95),  38, COLOR_TEXT_TITLE)

        bbox = self.drawing_bbox
        self._draw_drawing_box(display)

        # ---- HINT OVERLAY (new) ----
        # Draw BEFORE the user strokes so the user can see their ink on top.
        self._draw_hint_stroke(display)
        # ----------------------------

        # Previously the hint logic was an if/else that drew a plain filled stroke;
        # that block is now fully replaced by _draw_hint_stroke above.

        self._draw_user_strokes(display)
        self._draw_direction_arrows(display)
        self._draw_live_feedback(display)

        # Reveal character on completion
        if self.character_complete:
            cd = self.char_data
            cd.draw_union(display, bbox, color=COLOR_COMPLETED, thickness=2)
            for i in range(cd.num_strokes):
                cd.draw_stroke(display, i, bbox, filled=True, color=COLOR_COMPLETED)

        status = (
            f"Correct! {self.char_info['char']} ({self.char_info['pinyin']})"
            if self.character_complete else
            f"Stroke {self.current_stroke_idx + 1} / {self.char_data.num_strokes}"
        )
        self._draw_status_bar(display, status)
        self._draw_shortcuts(display)
        self._draw_nav_buttons(display)
        self._draw_calibration_overlay(display)
        return display

    def render_mode_select(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (20, 25, 40), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        h, w = display.shape[:2]
        put_text(display, "Chinese Character Tutor",
                 (w // 2 - 300, 80), 60, (150, 200, 255))

        keys = ["1", "2", "3", "4", "Q"]
        for i, ((title, desc), key) in enumerate(zip(MENU_LABELS, keys)):
            by  = MENU_BUTTON_Y_START + i * MENU_BUTTON_SPACING
            bx  = MENU_BUTTON_X
            bx2 = bx + MENU_BUTTON_W
            by2 = by + MENU_BUTTON_H
            hovered = (i == self.hovered_button_idx)

            if hovered:
                cv2.rectangle(display, (bx, by), (bx2, by2), (100, 150, 200), -1)
                cv2.rectangle(display, (bx, by), (bx2, by2), (150, 200, 255), 3)
            else:
                cv2.rectangle(display, (bx, by), (bx2, by2), (50, 60, 80), -1)
                cv2.rectangle(display, (bx, by), (bx2, by2), (100, 120, 150), 2)

            text_col   = (255, 255, 255) if hovered else (180, 200, 220)
            title_text = f"[{key}] {title}"
            title_size, _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            put_text(display, title_text,
                     (bx + (MENU_BUTTON_W - title_size[0]) // 2, by + 45), 32, text_col)
            desc_size, _ = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            put_text(display, desc,
                     (bx + (MENU_BUTTON_W - desc_size[0]) // 2, by + 70), 22, (150, 170, 190))

        if self.tip_xy is not None:
            cx, cy = self.tip_xy
            col    = (0, 200, 100) if self.pinch_active else (200, 200, 200)
            cv2.circle(display, (cx, cy), 15, col, 2)
            if self.pinch_active:
                cv2.circle(display, (cx, cy), 8, col, -1)

        put_text(display, "Pinch a button or press 1/2/3/4/Q",
                 (w // 2 - 280, h - 60), 28, COLOR_TEXT)
        return display

    # ----------------------------
    # Input handling
    # ----------------------------

    def handle_key(self, key: int) -> bool:
        if self.mode == "mode_select":
            if key == ord('1'):
                self.mode = "teaching";  self.select_new_character()
            elif key == ord('2'):
                self.mode = "free_draw"; self.select_new_character()
            elif key == ord('3'):
                self.mode = "pinyin";    self.select_new_character()
            elif key == ord('4'):
                self.mode = "english";   self.select_new_character()
            elif key == ord('q'):
                return False
        else:
            if key == ord(' '):
                if self.character_complete:
                    self.select_new_character()
            elif key == ord('c'):
                self.current_stroke_idx  = 0
                self.user_strokes        = []
                self.current_user_stroke = []
                self._reset_live_state()
                self.character_complete  = False
                self.complete_time       = None
                self.feedback            = []
                self.anim_start          = time.time()
                self.hint_active         = False
                self.hint_was_active     = False
            elif key == ord('n'):
                self.select_new_character()
            elif key == ord('m'):
                self.mode = "mode_select"
            elif key == ord('q'):
                return False
            elif key == ord('h'):
                # ---- HINT KEY ----
                # Only available in recall modes, not while a stroke is
                # already in progress, and not once the character is complete.
                if (self.mode in ("pinyin", "english")
                        and not self.character_complete
                        and not self.drawing):
                    self.hint_active     = True
                    self.hint_stroke_idx = self.current_stroke_idx
                    self.anim_start      = time.time()   # restart arrow animation
                    self.hint_was_active = False          # reset; set again on pinch
                # ------------------
            else:
                if self.character_complete:
                    self.select_new_character()
        return True

    # ----------------------------
    # Main loop
    # ----------------------------

    def run(self):
        cv2.namedWindow("Chinese Character Tutor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chinese Character Tutor", WINDOW_WIDTH, WINDOW_HEIGHT)
        print("Window created. Press 1/2/3/4 to select mode or Q to quit.")
        print(f"Camera: {CAMERA_INDEX}, Resolution: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        running     = True
        frame_count = 0
        while running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            frame   = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame   = self.process_frame(frame)

            if self.should_quit:
                break

            if self.mode == "mode_select":
                display = self.render_mode_select(frame)
            elif self.mode == "teaching":
                display = self.render_teaching_mode(frame)
            elif self.mode == "free_draw":
                display = self.render_free_draw_mode(frame)
            elif self.mode in ("pinyin", "english"):
                display = self.render_recall_mode(frame)
            else:
                display = frame.copy()

            if self.complete_time is not None:
                if time.time() - self.complete_time >= AUTO_ADVANCE_DELAY:
                    self.select_new_character()

            cv2.imshow("Chinese Character Tutor", display)
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"Frame {frame_count} displayed. Mode: {self.mode}")

            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                running = self.handle_key(key)

        print("Closing application...")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = TutorApp()
    app.run()
