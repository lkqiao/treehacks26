"""
Character data module for Chinese Character Tutor.

Loads MakeMeAHanzi graphics.txt and provides:
- CharacterData class for rendering strokes/outlines
- Curated CHARACTER_LIST with pinyin/english metadata
- Median-based stroke data for matching
"""
import math
import cv2
import json
import os
import sys
import random
import shapely.geometry as sg
from shapely.ops import unary_union, split
from shapely.geometry import LineString
import numpy as np
from svg.path import parse_path
from xml.dom import minidom


# ==============================
# SVG / Path Helpers
# ==============================

def get_point_at(path, distance, scale, offset):
    pos = path.point(distance)
    pos += offset
    pos *= scale
    return pos.real, pos.imag


def points_from_path(path, density, scale, offset):
    step = int(path.length() * density)
    last_step = step - 1

    if last_step == 0:
        yield get_point_at(path, 0, scale, offset)
        return

    for distance in range(step):
        yield get_point_at(
            path, distance / last_step, scale, offset)


def points_from_doc(doc, density=5, scale=1, offset=0):
    offset = offset[0] + offset[1] * 1j
    points = []
    for element in doc.getElementsByTagName("path"):
        for path in parse_path(element.getAttribute("d")):
            points.extend(points_from_path(
                path, density, scale, offset))
    return points


# ==============================
# Coordinate Transforms
# ==============================

def makemeahanzi_to_display(pts):
    """
    Apply MakeMeAHanzi display transform: scale(1,-1) translate(0,-900).
    Result: (x, 900 - y) for standard y-increases-downward display.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    out = pts.copy()
    out[:, 1] = 900.0 - pts[:, 1]
    return out

def normalize_makemeahanzi(pts):
    #normalize makemeahanzi coordinates to 0-1 floating
    return makemeahanzi_to_display(pts).astype(np.float32)/1024

def resize_to_box(pts, bbox_px):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    out = pts.copy()
    out[:, 0] = pts[:, 0] * (bbox_px[2] - bbox_px[0]) + bbox_px[0]
    out[:, 1] = pts[:, 1] * (bbox_px[3] - bbox_px[1]) + bbox_px[1]
    return out

def makemeahanzi_to_box_px(pts, bbox_px):
    """
    Apply MakeMeAHanzi transform and map to bbox pixels.
    Maps 0-1024 coordinate range to bbox pixel range.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    out = makemeahanzi_to_display(pts)
    bw = bbox_px[2] - bbox_px[0]
    bh = bbox_px[3] - bbox_px[1]
    out[:, 0] = bbox_px[0] + (out[:, 0] * bw) / 1024
    out[:, 1] = bbox_px[1] + (out[:, 1] * bh) / 1024
    return out


def points_from_stroke(path_d, density=1):
    """Get points for a single stroke (SVG path d string)."""
    path = parse_path(path_d)
    pts = []
    for segment in path:
        seg_pts = list(points_from_path(segment, density, 1, 0j))
        pts.extend(seg_pts)
    return pts


# ==============================
# Data Loading
# ==============================

def load_graphics(path: str):
    """Load graphics.txt; return dict char -> {strokes, medians}."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            char = obj.get("character")
            if char:
                data[char] = obj
    return data

def load_dictionary(path: str):
    """Load dictionary.txt; return dict char -> {pinyin, english}."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            
            char = obj.get("character")
            pinyin = obj.get("pinyin")
            english = obj.get("definition")
            if pinyin and english:
                data[char] = {"pinyin": pinyin, "english": english}
    return data

def character_to_stroke_svgs(char: str, obj: dict, viewbox: str = "0 0 1024 1024") -> list:
    strokes = obj.get("strokes", [])
    stroke_svgs = []
    for d in strokes:
        path = f'    <path d="{d}" fill="none" stroke="black" stroke-width="4"/>'
        stroke_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">
  <g transform="scale(1, -1) translate(0, -900)">
{path}
  </g>
</svg>
'''
        stroke_svgs.append(stroke_svg)
    return stroke_svgs


# Load data (path relative to this file)
_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = load_graphics(os.path.join(_DIR, "makemeahanzi", "graphics.txt"))
DICTIONARY = load_dictionary(os.path.join(_DIR, "makemeahanzi", "dictionary.txt"))
cached_character_data = {}


# ==============================
# CharacterData Class
# ==============================

class CharacterData:
    def __init__(self, char: str):
        self.char = char
        self.obj = DATA[char]
        self.pinyin = DICTIONARY[char]["pinyin"]
        self.english = DICTIONARY[char]["english"]
        self.stroke_svgs = character_to_stroke_svgs(char, self.obj)
        self.stroke_medians = [normalize_makemeahanzi(np.array(s)) for s in self.obj.get("medians", [])]
        self.stroke_polygons = []
        self.num_strokes = len(self.stroke_svgs)
        for i, stroke_svg in enumerate(self.stroke_svgs):
            doc = minidom.parseString(stroke_svg)
            pts = np.array(points_from_doc(doc, density=0.5, scale=1, offset=(0, 0)))
            pts = normalize_makemeahanzi(pts)
            self.stroke_polygons.append(sg.Polygon(pts))
        

        self.union = unary_union(self.stroke_polygons)
        self.partial_stroke_polygon = None
        self.partial_stroke_progress = 0

    def get_stroke_midline(self, stroke_idx: int, bbox_px):
        """Get median points for a stroke, transformed to pixel coords."""
        if stroke_idx < 0 or stroke_idx >= len(self.stroke_medians):
            return np.array([])
        pts = np.array(self.stroke_medians[stroke_idx])
        pts = resize_to_box(pts, bbox_px)
        return pts

    def draw_stroke(self, frame, stroke_idx: int, bbox_px, filled=False, color=(0, 0, 255)):
        """Draw a single stroke outline or filled."""
        if stroke_idx >= self.num_strokes or stroke_idx < 0:
            return
        pts = self.stroke_polygons[stroke_idx].exterior.coords
        pts = resize_to_box(pts, bbox_px).astype(np.int32)
        if filled:
            cv2.fillPoly(frame, [pts], color)
        else:
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    def draw_stroke_midline(self, frame, stroke_idx: int, bbox_px, color=(0, 0, 255), thickness=2):
        """Draw the median line of a stroke."""
        if stroke_idx < 0 or stroke_idx >= len(self.stroke_medians):
            return
        pts = resize_to_box(self.get_stroke_midline(stroke_idx, bbox_px), bbox_px).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness)

    def draw_union(self, frame, bbox_px, color=(0, 0, 255), thickness=2):
        """Draw the union outline of all strokes."""
        if isinstance(self.union, sg.MultiPolygon):
            for polygon in self.union.geoms:
                for interior in polygon.interiors:
                    interior_pts = interior.coords
                    interior_pts = resize_to_box(interior_pts, bbox_px).astype(np.int32)
                    cv2.polylines(frame, [interior_pts], isClosed=True, color=color, thickness=thickness)
                pts = polygon.exterior.coords
                pts = resize_to_box(pts, bbox_px).astype(np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            for interior in self.union.interiors:
                interior_pts = interior.coords
                interior_pts = resize_to_box(interior_pts, bbox_px).astype(np.int32)
                cv2.polylines(frame, [interior_pts], isClosed=True, color=color, thickness=thickness)
            pts = self.union.exterior.coords
            pts = resize_to_box(pts, bbox_px).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    def draw_full_character(self, frame, bbox_px, current_stroke_idx: int = -1):
        """Draw full character with current stroke median highlighted."""
        self.draw_union(frame, bbox_px)
        if current_stroke_idx >= 0:
            self.draw_stroke_midline(frame, current_stroke_idx, bbox_px)
        if current_stroke_idx > 0:
            for i in range(min(current_stroke_idx, self.num_strokes)):
                self.draw_stroke(frame, i, bbox_px, filled=True)
    

    def update_partial_stroke(self, current_stroke_idx, movement):
        """Update the partial stroke of the character.

        Progress is a float where the integer part is the segment index and
        the fractional part is the interpolation factor between that median
        point and the next.  The splitting position is linearly interpolated
        so the reveal slides smoothly along the median instead of jumping.
        """
        if current_stroke_idx < 0:
            return False
        move_norm = np.linalg.norm(movement)
        if move_norm < 1e-8:
            return False
        move_dir = movement / move_norm
        current_medians = self.stroke_medians[current_stroke_idx]
        n = len(current_medians)
        if n < 2:
            return True  # degenerate stroke

        d_dir = None
        done = False
        min_progress = math.floor(self.partial_stroke_progress)
        for _ in range(5):
            pp_int = int(self.partial_stroke_progress)
            if pp_int + 1 >= n:
                done = True
                break
            d_next = current_medians[pp_int + 1] - current_medians[pp_int]
            d_len = np.linalg.norm(d_next)
            if d_len < 1e-8:
                self.partial_stroke_progress = float(pp_int + 1)
                continue
            d_dir = d_next / d_len
            prog = np.dot(d_dir, movement) - 0.2 * move_norm
            self.partial_stroke_progress += prog / d_len / 1500
            if prog < 0:
                break

        # Clamp progress
        self.partial_stroke_progress = max(self.partial_stroke_progress, 0)

        if d_dir is not None:
            # Interpolate position between median[pp_int] and median[pp_int+1]
            pp_int = int(self.partial_stroke_progress)
            frac = self.partial_stroke_progress - pp_int
            if pp_int + 1 < n:
                cur_pos = (1.0 - frac) * current_medians[pp_int] + frac * current_medians[pp_int + 1]
                # Direction for the splitting line
                d_seg = current_medians[pp_int + 1] - current_medians[pp_int]
                seg_len = np.linalg.norm(d_seg)
                if seg_len > 1e-8:
                    d_dir = d_seg / seg_len
            else:
                cur_pos = current_medians[-1]

            perpendicular_dir = np.array([-d_dir[1], d_dir[0]])
            line_length = 0.08
            splitting_line = LineString([
                cur_pos - perpendicular_dir * line_length,
                cur_pos + perpendicular_dir * line_length
            ])
            polygons = split(self.stroke_polygons[current_stroke_idx], splitting_line)
            if len(polygons.geoms) > 1:
                for polygon in polygons.geoms:
                    if sg.Point(current_medians[0]).within(polygon) and not sg.Point(current_medians[-1]).within(polygon):
                        self.partial_stroke_polygon = polygon
                        break
            else:
                self.partial_stroke_polygon = None

        if done:
            self.partial_stroke_progress = 0
            self.partial_stroke_polygon = None
        return done

        
        

    def draw_partial_stroke(self, frame, bbox_px, color = (0, 0, 255), filled = True):
        """Draw a partial stroke of the character."""
        if self.partial_stroke_polygon is not None:
            pts = self.partial_stroke_polygon.exterior.coords
            pts = resize_to_box(pts, bbox_px).astype(np.int32)
            if filled:
                cv2.fillPoly(frame, [pts], color)
            else:
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)



def get_character(char: str) -> "CharacterData":
    """Get or create cached CharacterData for a character."""
    if char not in cached_character_data:
        cached_character_data[char] = CharacterData(char)
    return cached_character_data[char]


# ==============================
# Curated Character List
# ==============================

CHARACTER_LIST = [ "新", "年", "快", "乐", "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "人",
    "大",
    "小",
    "上",
    "下",
    "中",
    "山",
    "水",
    "火",
    "木",
    "日",
    "月",
    "口",
    "手",
    "目",
    "耳",
    "田",
    "土",
    "云",
    
]

# Filter to only characters present in MakeMeAHanzi data
CHARACTER_LIST = [c for c in CHARACTER_LIST if c in DATA]

cur_ind = 0
def get_random_character_info() -> dict:
    """Get a random character entry from the curated list."""
    # global cur_ind
    # char = CHARACTER_LIST[cur_ind]
    # cur_ind = (cur_ind + 1) % len(CHARACTER_LIST)
    char = random.choice(CHARACTER_LIST)
    return char


# ==============================
# Standalone Demo
# ==============================

if __name__ == "__main__":
    character = "田"
    character_data = get_character(character)
    CAMERA_INDEX = 0
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Camera failed to initialize.")
        sys.exit(1)
    cur_stroke = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        character_data.draw_full_character(frame, (0, 0, w, h), current_stroke_idx=cur_stroke)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("w"):
            cur_stroke += 1
        elif key == ord("s"):
            cur_stroke -= 1
    cap.release()
    cv2.destroyAllWindows()
