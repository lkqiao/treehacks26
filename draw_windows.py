import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
import os

# Index finger tip landmark index (21-point hand model)
INDEX_FINGER_TIP = 8

# Hand connections for drawing (replaces mp.solutions.drawing_utils)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

def draw_hand_landmarks(frame, hand_landmarks_list, w, h):
    """Draw hand landmarks and connections on frame."""
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

# -------------------------------
# Download hand_landmarker model if needed
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker model...")
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH
        )
        print("Model downloaded.")
    except Exception as e:
        print("Failed to download model:", e)
        print("Download manually from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        sys.exit(1)

# -------------------------------
# MediaPipe Hand Setup (Tasks API) – GPU when available
# -------------------------------
def _create_hand_landmarker():
    """Create HandLandmarker, using GPU when available (Linux/macOS), else CPU."""
    for use_gpu in (True, False):
        try:
            base_opts = python.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
            )
            opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                running_mode=vision.RunningMode.VIDEO
            )
            lm = vision.HandLandmarker.create_from_options(opts)
            print("MediaPipe: Using", "GPU" if use_gpu else "CPU")
            return lm
        except Exception as e:
            if use_gpu:
                continue
            raise
    raise RuntimeError("Failed to create HandLandmarker")

# -------------------------------
# Webcam Setup
# -------------------------------
CAMERA_INDEX = 0
# Use default backend (works on Windows/Linux); CAP_AVFOUNDATION is macOS-only
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Camera failed to initialize.")
    print("Check System Settings → Privacy & Security → Camera")
    print("Make sure Terminal / IDE has camera access.")
    sys.exit(1)

# -------------------------------
# Drawing Setup
# -------------------------------
canvas = None
strokes = []
drawing = False
prev_point = None

MOVE_THRESHOLD = 5       # Minimum movement to add a point
Z_THRESHOLD = -0.05      # Max z for "finger close to screen" (adjust experimentally)

# -------------------------------
# Main Loop
# -------------------------------
frame_count = 0

with _create_hand_landmarker() as hand_landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp in ms (required for VIDEO mode; ~30 fps)
        frame_count += 1
        timestamp_ms = frame_count * 33

        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        finger_detected = False
        hand_landmarks = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            tip = hand_landmarks[INDEX_FINGER_TIP]
            x, y, z = int(tip.x * w), int(tip.y * h), tip.z

            if z < Z_THRESHOLD:
                finger_detected = True

                if not drawing:
                    drawing = True
                    prev_point = (x, y)
                    strokes.append([(x, y)])
                else:
                    if not strokes:
                        strokes.append([(x, y)])
                    dist = np.linalg.norm(np.array(prev_point) - np.array((x, y)))
                    if dist > MOVE_THRESHOLD:
                        cv2.line(canvas, prev_point, (x, y), (0, 255, 0), 5)
                        strokes[-1].append((x, y))
                        prev_point = (x, y)
            else:
                finger_detected = False

        if not finger_detected and drawing:
            drawing = False
            prev_point = None
            print(f"Stroke finished. Total strokes: {len(strokes)}")

        if result.hand_landmarks:
            draw_hand_landmarks(frame, result.hand_landmarks, w, h)

        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow("Finger Drawing MVP", combined)

        key = cv2.waitKey(1)
        if key == 27:  # ESC → quit
            break
        elif key == ord("c"):  # Clear canvas
            canvas = np.zeros_like(frame)
            strokes = []
            drawing = False
            prev_point = None
            print("Canvas cleared")

cap.release()
cv2.destroyAllWindows()
