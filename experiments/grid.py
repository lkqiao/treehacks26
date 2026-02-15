import cv2
import numpy as np
import os
import json

# ==============================
# CONFIG
# ==============================
TARGET_CHAR = "仫"
DATA_DIR = "../makemeahanzi"   # must contain graphics.txt
WINDOW_SIZE = 700


# ==============================
# Load Hanzi
# ==============================
def load_hanzi(char, data_dir):
    path = os.path.join(data_dir, "graphics.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("character") == char:
                return data

    raise ValueError(f"Character '{char}' not found in graphics.txt")


# ==============================
# Draw Hanzi 1024 Grid
# ==============================
def draw_grid(canvas):
    h, w = canvas.shape[:2]

    # outer border
    cv2.rectangle(canvas, (0, 0), (w-1, h-1), (255,255,255), 2)

    # center lines (512 in Hanzi space)
    cv2.line(canvas, (w//2, 0), (w//2, h), (200,200,200), 1)
    cv2.line(canvas, (0, h//2), (w, h//2), (200,200,200), 1)

    # thirds grid
    cv2.line(canvas, (w//3, 0), (w//3, h), (120,120,120), 1)
    cv2.line(canvas, (2*w//3, 0), (2*w//3, h), (120,120,120), 1)
    cv2.line(canvas, (0, h//3), (w, h//3), (120,120,120), 1)
    cv2.line(canvas, (0, 2*h//3), (w, 2*h//3), (120,120,120), 1)


# ==============================
# Draw Median Strokes
# ==============================
def draw_medians(canvas, medians):
    size = canvas.shape[0]

    # normalize full character bounding box
    all_pts = np.vstack([np.asarray(s, np.float32) for s in medians if len(s) > 0])
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    span = mx - mn
    span[span < 1e-6] = 1.0
    scale = float(max(span[0], span[1]))

    def map_pts(stroke):
        pts = (np.asarray(stroke, np.float32) - mn) / scale
        pts[:,1] = 1.0 - pts[:,1]   # ← FLIP Y (normalized space)
        pts_px = (pts * (size * 0.8) + size * 0.1).astype(np.int32)
        return pts_px

    for idx, stroke in enumerate(medians):
        if len(stroke) < 2:
            continue

        pts_px = map_pts(stroke)

        # draw stroke line
        for i in range(len(pts_px) - 1):
            cv2.line(canvas, tuple(pts_px[i]), tuple(pts_px[i+1]),
                     (255,255,255), 3)

        # start point (yellow)
        cv2.circle(canvas, tuple(pts_px[0]), 8, (0,255,255), -1)

        # end point (orange)
        cv2.circle(canvas, tuple(pts_px[-1]), 8, (0,165,255), -1)

        # stroke number
        cv2.putText(canvas, str(idx+1),
                    tuple(pts_px[0] + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,255), 2)


# ==============================
# MAIN
# ==============================
def main():
    hanzi = load_hanzi(TARGET_CHAR, DATA_DIR)
    medians = hanzi.get("medians", [])

    print(f"Loaded '{TARGET_CHAR}' with {len(medians)} strokes.")

    canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

    draw_grid(canvas)
    draw_medians(canvas, medians)

    cv2.putText(canvas, f"Character: {TARGET_CHAR}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255,255,255), 2)

    cv2.imshow("Hanzi Median Viewer", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
