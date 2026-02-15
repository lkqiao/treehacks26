import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import sys
import urllib.request

# -------------------------------
# Config
# -------------------------------
CAMERA_INDEX = 0

INDEX_TIP = 8
MIDDLE_TIP = 4

TWO_FINGER_THRESH_NORM = 0.06   # distance in normalized coords between index & middle tips
MOVE_THRESHOLD_PX = 5
LINE_THICKNESS = 5

# -------------------------------
# Download model if needed
# -------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker model...")
    try:
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH
        )
        print("Model downloaded.")
    except Exception as e:
        print("Failed to download model:", e)
        sys.exit(1)

# -------------------------------
# MediaPipe Tasks HandLandmarker
# -------------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)

# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Camera failed to initialize.")
    sys.exit(1)

canvas = None
drawing = False
prev_point = None
strokes = []

frame_count = 0

with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_count += 1
        timestamp_ms = frame_count * 33  # ~30fps

        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        finger_down = False
        tip_xy = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            idx = hand[INDEX_TIP]
            mid = hand[MIDDLE_TIP]

            tip_xy = (int(idx.x * w), int(idx.y * h))

            # two-finger distance (normalized)
            two_finger_dist = math.hypot(idx.x - mid.x, idx.y - mid.y)
            finger_down = (two_finger_dist < TWO_FINGER_THRESH_NORM)

            # lightweight viz (no mp.solutions): draw a couple points + line
            p_idx = (int(idx.x * w), int(idx.y * h))
            p_mid = (int(mid.x * w), int(mid.y * h))
            cv2.circle(frame, p_idx, 6, (0, 255, 255), -1)
            cv2.circle(frame, p_mid, 6, (0, 255, 255), -1)
            cv2.line(frame, p_idx, p_mid, (0, 255, 255), 2)
            cv2.putText(frame, f"dist={two_finger_dist:.3f}",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # ---------------- stroke capture ----------------
        if finger_down and tip_xy is not None:
            x, y = tip_xy
            if not drawing:
                drawing = True
                prev_point = (x, y)
                strokes.append([(x, y)])
            else:
                dist = np.linalg.norm(np.array(prev_point) - np.array((x, y)))
                if dist > MOVE_THRESHOLD_PX:
                    cv2.line(canvas, prev_point, (x, y), (0, 255, 0), LINE_THICKNESS)
                    strokes[-1].append((x, y))
                    prev_point = (x, y)

        # ---------------- stroke end ----------------
        if (not finger_down) and drawing:
            drawing = False
            prev_point = None
            print(f"Stroke finished. Total strokes: {len(strokes)}")

        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow("Two Finger Draw (MediaPipe Tasks)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)
            strokes = []
            drawing = False
            prev_point = None
            print("Cleared")

cap.release()
cv2.destroyAllWindows()
