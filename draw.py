import cv2
import mediapipe as mp
import numpy as np
import sys

# -------------------------------
# MediaPipe Hand Setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Webcam Setup (macOS built-in camera)
# -------------------------------
CAMERA_INDEX = 0  # Adjust if needed
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

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
Z_THRESHOLD = -0.05      # Max z for “finger close to screen” (adjust experimentally)

# -------------------------------
# Main Loop
# -------------------------------
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
    results = hands.process(rgb)

    finger_detected = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Index fingertip
        tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y, z = int(tip.x * w), int(tip.y * h), tip.z

        # Only draw if finger is “close enough”
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
            # Finger far → treat as pen lift
            finger_detected = False

    # Finger lifted → end stroke
    if not finger_detected and drawing:
        drawing = False
        prev_point = None
        print(f"Stroke finished. Total strokes: {len(strokes)}")

    # Draw hand landmarks for feedback
    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay canvas on webcam feed
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
