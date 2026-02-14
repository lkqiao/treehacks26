"""
Chinese Character Tutor - Main Application
Three Modes:
1. Teaching Mode: Learn to write (shows stroke guide with animations)
2. Pinyin Recognition: See pinyin, recall character
3. English Translation: See English, recall character + gamification
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
import time
import random
import os
from typing import List, Tuple, Optional, Dict

from stroke_engine import CharacterDatabase, match_stroke_to_template, validate_stroke_order
from ui_renderer import UIRenderer, AnimationManager

# ===============================
# Hand Detection Setup (Tasks API)
# ===============================

# Index finger tip landmark index (21-point hand model)
INDEX_FINGER_TIP = 8

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

# Download hand_landmarker model if needed
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker model...")
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH
        )
        print("✓ Model downloaded.")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("Download manually from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
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
# Constants
# ===============================

MOVE_THRESHOLD = 5
Z_THRESHOLD = -0.05
CAMERA_INDEX = 0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Stroke rendering
STROKE_COLOR = (100, 200, 100)  # BGR
STROKE_THICKNESS = 8
STROKE_ALPHA = 0.4  # Translucent

# ===============================
# Application State
# ===============================

class TutorApp:
    def __init__(self):
        self.mode = "mode_select"  # mode_select, teaching, pinyin, english
        self.character_db = CharacterDatabase("characters.json")
        self.renderer = UIRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.animator = AnimationManager()
        
        # Game state
        self.current_character = None
        self.current_stroke_idx = 0
        self.user_strokes = []
        self.current_user_stroke = []
        self.feedback = []
        self.score = 0
        self.character_history = []
        self.completed_characters = 0
        self.incorrect_strokes = []
        
        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print("Error: Camera failed to initialize.")
            sys.exit(1)
        
        self.drawing = False
        self.prev_point = None
        
        # UI state
        self.mode_selected = False
        self.character_submitted = False
        self.character_correct = False
        self.frame_count = 0
        
        # Auto-advance timer
        self.complete_time = None
        self.auto_advance_delay = 2.0  # seconds before auto-advancing
    
    def select_new_character(self):
        """Select a new character for the current mode."""
        self.current_character = self.character_db.get_random_character()
        self.user_strokes = []
        self.current_stroke_idx = 0
        self.character_submitted = False
        self.character_correct = False
        self.incorrect_strokes = []
        self.animator.reset()
        self.feedback = []
    
    def reset_for_next_character(self):
        """Reset state for next character."""
        self.select_new_character()
        self.complete_time = None  # Reset auto-advance timer
        if self.character_correct:
            self.completed_characters += 1
            if self.mode == "english" or self.mode == "pinyin":
                self.score += 100  # Base points
    
    
    def _normalize_to_template_space(self, pixel_stroke: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to normalized 0-1 space (template space)."""
        if not pixel_stroke:
            return []
        
        # Use actual frame dimensions
        # Assume full frame is drawing area (no padding needed with camera feed)
        w, h = WINDOW_WIDTH, WINDOW_HEIGHT
        
        normalized = []
        for px, py in pixel_stroke:
            norm_x = px / w if w > 0 else 0
            norm_y = py / h if h > 0 else 0
            # Clamp to 0-1
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            normalized.append((norm_x, norm_y))
        
        return normalized
    
    def _normalize_to_template_space(self, pixel_stroke: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to normalized 0-1 space (template space)."""
        if not pixel_stroke:
            return []
        
        # Use actual frame dimensions
        # Assume full frame is drawing area (no padding needed with camera feed)
        w, h = WINDOW_WIDTH, WINDOW_HEIGHT
        
        normalized = []
        for px, py in pixel_stroke:
            norm_x = px / w if w > 0 else 0
            norm_y = py / h if h > 0 else 0
            # Clamp to 0-1
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            normalized.append((norm_x, norm_y))
        
        return normalized
    
    def add_point_to_current_stroke(self, x: int, y: int):
        """Add a point to the current stroke being drawn."""
        self.current_user_stroke.append((x, y))
    
    def finish_stroke(self):
        """Finish the current stroke."""
        if len(self.current_user_stroke) >= 3:
            # Normalize stroke from pixel to 0-1
            normalized_stroke = self._normalize_to_template_space(self.current_user_stroke)
            self.user_strokes.append(normalized_stroke)
            
            # Validate this stroke
            if self.mode == "teaching":
                self._validate_teaching_stroke()
            elif self.mode == "pinyin" or self.mode == "english":
                # Auto-check on finish for recall modes
                self._check_stroke_progress()
            
            self.current_user_stroke = []
    
    def _validate_teaching_stroke(self):
        """Validate stroke in teaching mode."""
        if not self.current_character:
            return
        
        template_strokes = self.current_character['strokes']
        
        if self.current_stroke_idx < len(template_strokes):
            template = template_strokes[self.current_stroke_idx]
            last_user_stroke = self.user_strokes[-1]
            
            result = match_stroke_to_template([last_user_stroke], [template], threshold=0.3)
            
            if result['accuracy'] >= 0.7:
                self.feedback.append(f"✓ Stroke {self.current_stroke_idx + 1} correct!")
                self.current_stroke_idx += 1
                
                # Check if character is complete
                if self.current_stroke_idx >= len(template_strokes):
                    self.character_correct = True
                    self.character_submitted = True
                    self.complete_time = time.time()
                    self.feedback.append(f"🎉 Character {self.current_character['char']} completed!")
            else:
                self.feedback.append(f"✗ Stroke {self.current_stroke_idx + 1} incorrect. Try again!")
    
    def _check_stroke_progress(self):
        """Check if we have enough strokes to validate character (for recall modes)."""
        if not self.current_character:
            return
        
        template_strokes = self.current_character['strokes']
        
        # If user has drawn all strokes, auto-validate
        if len(self.user_strokes) >= len(template_strokes):
            result = match_stroke_to_template(self.user_strokes, template_strokes, threshold=0.25)
            
            if result['matched']:
                self.character_correct = True
                self.character_submitted = True
                self.complete_time = time.time()
                self.feedback.append(f"✓ Correct! {self.current_character['char']}")
                
                # Scoring
                if self.mode == "pinyin":
                    self.score += 150
                elif self.mode == "english":
                    self.score += 200
                
                self.completed_characters += 1
    
    def submit_character(self):
        """Submit character drawing for evaluation."""
        if not self.current_character or self.character_submitted:
            return
        
        template_strokes = self.current_character['strokes']
        result = match_stroke_to_template(self.user_strokes, template_strokes, threshold=0.25)
        
        if result['matched']:
            self.character_correct = True
            self.feedback.append(f"✓ Character correct! {self.current_character['char']}")
            
            # Scoring
            if self.mode == "pinyin":
                self.score += 150
            elif self.mode == "english":
                self.score += 200
            
            self.completed_characters += 1
        else:
            self.character_correct = False
            accuracy_pct = int(result['accuracy'] * 100)
            self.feedback.append(f"✗ Only {accuracy_pct}% correct. Try again!")
            
            # Show which strokes were wrong
            for wrong in result['wrong_strokes']:
                idx = wrong['stroke_idx']
                self.feedback.append(f"  Stroke {idx + 1}: {wrong['reason']}")
                self.incorrect_strokes.append(idx)
        
        self.character_submitted = True
        
        # Start auto-advance timer if character is correct
        if self.character_correct:
            self.complete_time = time.time()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a camera frame and detect hand."""
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        self.frame_count += 1
        timestamp_ms = self.frame_count * 33
        
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        finger_detected = False
        hand_landmarks = None
        
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            tip = hand_landmarks[INDEX_FINGER_TIP]
            x, y, z = int(tip.x * w), int(tip.y * h), tip.z
            
            if z < Z_THRESHOLD:
                finger_detected = True
                
                if not self.drawing:
                    self.drawing = True
                    self.prev_point = (x, y)
                    self.current_user_stroke = [(x, y)]
                else:
                    if self.prev_point:
                        dist = np.linalg.norm(np.array(self.prev_point) - np.array((x, y)))
                        if dist > MOVE_THRESHOLD:
                            self.current_user_stroke.append((x, y))
                            self.prev_point = (x, y)
        
        if not finger_detected and self.drawing:
            self.drawing = False
            self.prev_point = None
            self.finish_stroke()
        
        # Draw hand landmarks
        if result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for (px, py) in pts:
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
        
        return frame
    
    def render_teaching_mode(self, frame: np.ndarray) -> np.ndarray:
        """Render teaching mode interface on camera feed."""
        # Darken background for readability
        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        
        if not self.current_character:
            self.select_new_character()
        
        char = self.current_character
        template_strokes = char['strokes']
        
        h, w = display.shape[:2]
        
        # Title
        cv2.putText(display, f"Teaching Mode - {char['char']} ({char['english']})", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2)
        cv2.putText(display, f"Pinyin: {char['pinyin']}", 
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
        
        # Draw completed strokes (semi-transparent green)
        for i in range(self.current_stroke_idx):
            stroke_px = self._normalize_to_pixel_coords(template_strokes[i])
            overlay = display.copy()
            for j in range(len(stroke_px) - 1):
                cv2.line(overlay, stroke_px[j], stroke_px[j+1], (0, 200, 100), STROKE_THICKNESS)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        
        # Draw current stroke guide with animation
        if self.current_stroke_idx < len(template_strokes):
            current_template = template_strokes[self.current_stroke_idx]
            
            # Draw animated arrow
            progress = self.animator.get_progress(2.0)
            self._draw_animated_arrow_on_frame(display, current_template, progress)
            
            # Draw semi-transparent guide
            stroke_px = self._normalize_to_pixel_coords(current_template)
            overlay = display.copy()
            for j in range(len(stroke_px) - 1):
                cv2.line(overlay, stroke_px[j], stroke_px[j+1], (150, 255, 150), STROKE_THICKNESS - 2)
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
        
        # Draw user's current stroke
        if self.current_user_stroke:
            overlay = display.copy()
            for i in range(len(self.current_user_stroke) - 1):
                p1 = self.current_user_stroke[i]
                p2 = self.current_user_stroke[i+1]
                cv2.line(overlay, p1, p2, STROKE_COLOR, STROKE_THICKNESS)
            cv2.addWeighted(overlay, STROKE_ALPHA, display, 1 - STROKE_ALPHA, 0, display)
        
        # Draw already-drawn user strokes
        for stroke in self.user_strokes:
            stroke_px = self._normalize_to_pixel_coords(stroke)
            overlay = display.copy()
            for i in range(len(stroke_px) - 1):
                cv2.line(overlay, stroke_px[i], stroke_px[i+1], (0, 255, 100), STROKE_THICKNESS)
            cv2.addWeighted(overlay, STROKE_ALPHA, display, 1 - STROKE_ALPHA, 0, display)
        
        # Status
        status = f"Stroke {self.current_stroke_idx + 1} / {len(template_strokes)}"
        if self.character_submitted:
            if self.character_correct:
                status = "✓ Character Complete! (auto-advancing...)"
            else:
                status = "Try again or press SPACE for next"
        
        # Feedback box
        cv2.putText(display, status, (20, h - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)
        
        for i, line in enumerate(self.feedback[-2:]):
            cv2.putText(display, line, (20, h - 30 - i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return display
    
    def _normalize_to_pixel_coords(self, normalized_stroke: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Convert normalized 0-1 coordinates to pixel coordinates."""
        w, h = WINDOW_WIDTH, WINDOW_HEIGHT
        return [(int(x * w), int(y * h)) for x, y in normalized_stroke]
    
    def _draw_animated_arrow_on_frame(self, frame: np.ndarray, stroke: List[Tuple[float, float]], progress: float):
        """Draw animated arrow showing stroke direction on frame."""
        if len(stroke) < 2:
            return
        
        # Convert to pixel coords
        pts = self._normalize_to_pixel_coords(stroke)
        
        # Get segment based on progress
        total_len = sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i])) for i in range(len(pts)-1))
        target_len = total_len * progress
        
        cumulative = 0.0
        arrow_start = None
        arrow_end = None
        
        for i in range(len(pts) - 1):
            seg_len = np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i]))
            if cumulative + seg_len >= target_len:
                alpha = (target_len - cumulative) / seg_len if seg_len > 0 else 0
                p_start = np.array(pts[i]) + alpha * (np.array(pts[i+1]) - np.array(pts[i]))
                
                look_ahead = 40
                if i + 1 < len(pts) - 1:
                    next_seg_len = np.linalg.norm(np.array(pts[i+2]) - np.array(pts[i+1]))
                    remaining = seg_len - alpha * seg_len
                    if remaining + next_seg_len >= look_ahead:
                        beta = look_ahead / (remaining + next_seg_len) if (remaining + next_seg_len) > 0 else 0
                        p_end = np.array(pts[i+1]) + beta * (np.array(pts[i+2]) - np.array(pts[i+1]))
                    else:
                        direction = (np.array(pts[i+1]) - np.array(pts[i])) / (seg_len + 1e-6)
                        p_end = p_start + look_ahead * direction
                else:
                    direction = (np.array(pts[i+1]) - np.array(pts[i])) / (seg_len + 1e-6)
                    p_end = p_start + look_ahead * direction
                
                arrow_start = tuple(map(int, p_start))
                arrow_end = tuple(map(int, p_end))
                break
            cumulative += seg_len
        
        if arrow_start and arrow_end:
            cv2.line(frame, arrow_start, arrow_end, (255, 100, 0), 4)
            
            direction = np.array(arrow_end) - np.array(arrow_start)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            angle = np.arctan2(direction[1], direction[0])
            
            arrow_size = 15
            pt1 = arrow_end - arrow_size * np.array([np.cos(angle - np.pi/6), np.sin(angle - np.pi/6)])
            pt2 = arrow_end - arrow_size * np.array([np.cos(angle + np.pi/6), np.sin(angle + np.pi/6)])
            
            cv2.line(frame, arrow_end, tuple(map(int, pt1)), (255, 100, 0), 3)
            cv2.line(frame, arrow_end, tuple(map(int, pt2)), (255, 100, 0), 3)
    
    def render_pinyin_mode(self, frame: np.ndarray) -> np.ndarray:
        """Render Pinyin recognition mode on camera feed."""
        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        
        if not self.current_character:
            self.select_new_character()
        
        char = self.current_character
        h, w = display.shape[:2]
        
        # Title
        cv2.putText(display, "Pinyin Mode", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 200, 255), 2)
        cv2.putText(display, f"Write the character that sounds like: {char['pinyin']}", 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 200, 255), 2)
        
        # Draw user strokes
        for stroke in self.user_strokes:
            stroke_px = self._normalize_to_pixel_coords(stroke)
            overlay_tmp = display.copy()
            for i in range(len(stroke_px) - 1):
                cv2.line(overlay_tmp, stroke_px[i], stroke_px[i+1], (0, 200, 100), STROKE_THICKNESS)
            cv2.addWeighted(overlay_tmp, STROKE_ALPHA, display, 1 - STROKE_ALPHA, 0, display)
        
        if self.current_user_stroke:
            overlay_tmp = display.copy()
            for i in range(len(self.current_user_stroke) - 1):
                cv2.line(overlay_tmp, self.current_user_stroke[i], self.current_user_stroke[i+1], 
                        STROKE_COLOR, STROKE_THICKNESS)
            cv2.addWeighted(overlay_tmp, STROKE_ALPHA, display, 1 - STROKE_ALPHA, 0, display)
        
        # Status
        status = ""
        if self.character_submitted:
            if self.character_correct:
                status = f"✓ Correct! {char['char']}"
            else:
                status = "✗ Incorrect. Try again!"
        
        # Score and feedback
        cv2.putText(display, f"Score: {self.score:05d}", (w - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        
        if status:
            cv2.putText(display, status, (20, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        
        for i, line in enumerate(self.feedback[-2:]):
            cv2.putText(display, line, (20, h - 40 - i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return display
    
    def render_english_mode(self, frame: np.ndarray) -> np.ndarray:
        """Render English translation mode on camera feed."""
        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        
        if not self.current_character:
            self.select_new_character()
        
        char = self.current_character
        h, w = display.shape[:2]
        
        # Title
        cv2.putText(display, "Translation Mode", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 200, 255), 2)
        cv2.putText(display, f"Write the character for: {char['english'].upper()}", 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 200, 255), 2)
        
        # Draw user strokes
        for stroke in self.user_strokes:
            stroke_px = self._normalize_to_pixel_coords(stroke)
            overlay_tmp = display.copy()
            for i in range(len(stroke_px) - 1):
                cv2.line(overlay_tmp, stroke_px[i], stroke_px[i+1], (0, 200, 100), STROKE_THICKNESS)
            cv2.addWeighted(overlay_tmp, STROKE_ALPHA, display, 1 - STROKE_ALPHA, 0, display)
        
        if self.current_user_stroke:
            overlay_tmp = display.copy()
            for i in range(len(self.current_user_stroke) - 1):
                cv2.line(overlay_tmp, self.current_user_stroke[i], self.current_user_stroke[i+1],
                        STROKE_COLOR, STROKE_THICKNESS)
            cv2.addWeighted(overlay_tmp, STROKE_ALPHA, display, 1 - STROKE_ALPHA, 0, display)
        
        # Status
        status = ""
        bonus_text = ""
        if self.character_submitted:
            if self.character_correct:
                status = f"✓ Correct! {char['char']}"
                bonus_text = "+200 Points!"
            else:
                status = "✗ Incorrect. Try again!"
        
        # Score and stats
        cv2.putText(display, f"Score: {self.score:05d}", (w - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        cv2.putText(display, f"Completed: {self.completed_characters}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 200, 255), 1)
        
        if status:
            cv2.putText(display, status, (20, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        if bonus_text:
            cv2.putText(display, bonus_text, (20, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 150), 2)
        
        for i, line in enumerate(self.feedback[-2:]):
            cv2.putText(display, line, (20, h - 10 - i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return display
    
    def render_mode_select(self, frame: np.ndarray) -> np.ndarray:
        """Render mode selection screen on camera feed."""
        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        
        h, w = display.shape[:2]
        
        # Title
        cv2.putText(display, "Chinese Character Tutor", (w//2 - 300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 200, 100), 3)
        
        # Mode options
        modes = [
            ("1", "Teaching Mode", "Learn stroke-by-stroke with guides"),
            ("2", "Pinyin Recognition", "See pinyin, recall character"),
            ("3", "English Translation", "See English, recall character (Gamified)"),
            ("Q", "Quit", "Exit application")
        ]
        
        y = 200
        for key, title, desc in modes:
            cv2.putText(display, f"[{key}] {title}", (60, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 200, 255), 2)
            cv2.putText(display, f"    {desc}", (100, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
            y += 120
        
        cv2.putText(display, "Press a key to select...", (w//2 - 200, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        
        return display
    
    def handle_key(self, key: int):
        """Handle keyboard input."""
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
            if key == ord(' '):  # Space to submit
                if not self.character_submitted:
                    self.submit_character()
                else:
                    self.reset_for_next_character()
            elif key == ord('c'):  # C to clear current drawing
                self.current_user_stroke = []
                self.user_strokes = []
                self.incorrect_strokes = []
                self.character_submitted = False
                self.character_correct = False
                self.feedback = []
            elif key == ord('m'):  # M to return to mode select
                self.mode = "mode_select"
            elif key == ord('q'):
                return False
        
        return True
    
    def run(self):
        """Main application loop."""
        cv2.namedWindow("Chinese Character Tutor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chinese Character Tutor", WINDOW_WIDTH, WINDOW_HEIGHT)
        
        running = True
        while running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Resize and process frame
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            
            # Process hand detection
            frame = self.process_frame(frame)
            
            # Render current mode
            if self.mode == "mode_select":
                display_canvas = self.render_mode_select(frame)
            elif self.mode == "teaching":
                display_canvas = self.render_teaching_mode(frame)
            elif self.mode == "pinyin":
                display_canvas = self.render_pinyin_mode(frame)
            elif self.mode == "english":
                display_canvas = self.render_english_mode(frame)
            else:
                display_canvas = frame.copy()
            
            # Check for auto-advance on character completion
            if self.complete_time is not None:
                time_since_complete = time.time() - self.complete_time
                if time_since_complete >= self.auto_advance_delay:
                    self.reset_for_next_character()
            
            # Display
            cv2.imshow("Chinese Character Tutor", display_canvas)
            
            # Handle input
            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                running = self.handle_key(key)
            
            # Update animation
            self.animator.update()
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = TutorApp()
    app.run()
