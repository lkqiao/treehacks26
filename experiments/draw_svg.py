"""
Generate SVG file(s) from MakeMeAHanzi graphics.txt.

Usage:
  python draw_svg.py                    # Output 十.svg (default char)
  python draw_svg.py 永                 # Output 永.svg
  python draw_svg.py --all              # Output all characters to svg_output/
  python draw_svg.py 人 -o output.svg   # Custom output path
"""

import cv2
import json
import os
import sys
import argparse
import shapely.geometry as sg
from shapely.ops import unary_union
import numpy as np
from svg.path import parse_path
from xml.dom import minidom



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


def makemeahanzi_to_display(pts):
    """
    Apply MakeMeAHanzi display transform: scale(1,-1) translate(0,-900).
    Coords: upper-left (0,900), lower-right (1024,-124), y decreases downward.
    Result: (x, 900 - y) for standard y-increases-downward display.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    out = pts.copy()
    out[:, 1] = 900.0 - pts[:, 1]
    return out


def makemeahanzi_to_box_px(pts, bbox_px):
    """
    Apply MakeMeAHanzi transform and map to bbox pixels.
    Uses makemeahanzi_to_display then linear 0-1024 -> bbox.
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
    """
    Get points for a single stroke (SVG path d string).
    Uses same arc-length sampling as draw_character.
    Returns list of (x, y) in MakeMeAHanzi coords.
    """
    path = parse_path(path_d)
    pts = []
    for segment in path:
        seg_pts = list(points_from_path(segment, density, 1, 0j))
        pts.extend(seg_pts)
    return pts


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

# Path relative to this file: experiments/draw_svg.py -> makemeahanzi/graphics.txt
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRAPHICS_PATH = os.path.join(_SCRIPT_DIR, "..", "makemeahanzi", "graphics.txt")
DATA = load_graphics(_GRAPHICS_PATH)

def character_to_stroke_svgs(char: str, obj: dict, viewbox: str = "0 0 1024 1024") -> str:

    strokes = obj.get("strokes", [])
    stroke_svgs = []
    for d in strokes:
        path = f'    <path d="{d}" fill="none" stroke="black" stroke-width="4"/>'
        stroke_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- Character: {char} -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">
  <g transform="scale(1, -1) translate(0, -900)">
{path}
  </g>
</svg>
'''
        stroke_svgs.append(stroke_svg)
    return stroke_svgs

def get_stroke_midline(char: str, obj: dict, stroke_idx: int):
    strokes = obj.get("medians", [])
    return np.array(strokes[stroke_idx])

DATA = load_graphics("makemeahanzi/graphics.txt")
cached_character_data = {}

class CharacterData:
    def __init__(self, char: str):
        self.char = char
        self.obj = DATA[char]
        self.stroke_svgs = character_to_stroke_svgs(char, self.obj)
        self.stroke_polygons = []
        self.num_strokes = len(self.stroke_svgs)
        for i, stroke_svg in enumerate(self.stroke_svgs):
            doc = minidom.parseString(stroke_svg)
            pts = np.array(points_from_doc(doc, density=0.5, scale=1, offset=(0, 0)))
            self.stroke_polygons.append(sg.Polygon(pts))
        self.union = unary_union(self.stroke_polygons)

        
    def get_stroke_midline(self, stroke_idx: int, bbox_px: tuple[int, int, int, int]):
        strokes = self.obj.get("medians", [])
        pts = np.array(strokes[stroke_idx])
        pts = makemeahanzi_to_box_px(pts, bbox_px)
        return pts

    def draw_stroke(self, frame, stroke_idx: int, bbox_px: tuple[int, int, int, int], filled = False):
        if stroke_idx >= self.num_strokes or stroke_idx < 0:
            return
        pts = self.stroke_polygons[stroke_idx].exterior.coords
        pts = makemeahanzi_to_box_px(pts, bbox_px).astype(np.int32)
        if filled:
            cv2.fillPoly(frame, [pts], (0, 0, 255))
        else:
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    
    def draw_stroke_midline(self, frame, stroke_idx: int, bbox_px: tuple[int, int, int, int]):
        if stroke_idx >= len(self.obj.get("medians", [])) or stroke_idx < 0:
            return
        pts = np.array(self.get_stroke_midline(stroke_idx, bbox_px)).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    
    def draw_union(self, frame, bbox_px: tuple[int, int, int, int]):
        if type(self.union) == sg.MultiPolygon:
            for polygon in self.union.geoms:
                pts = polygon.exterior.coords
                pts = makemeahanzi_to_box_px(pts, bbox_px).astype(np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            pts = self.union.exterior.coords
            pts = makemeahanzi_to_box_px(pts, bbox_px).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)    



def draw_character(frame, char: str, bbox_px: tuple[int, int, int, int], current_stroke_idx = -1):
    if char not in cached_character_data:
        cached_character_data[char] = CharacterData(char)
    character_data = cached_character_data[char]
    character_data.draw_stroke_midline(frame, current_stroke_idx, bbox_px)
    character_data.draw_union(frame, bbox_px)

    if current_stroke_idx > 0:
        for i in range(min(current_stroke_idx, character_data.num_strokes)):
            character_data.draw_stroke(frame, i, bbox_px, filled = True)

if __name__ == "__main__":
    character = "会"
    data = load_graphics("makemeahanzi/graphics.txt")
    CAMERA_INDEX = 0
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Camera failed to initialize.")
        print("Check System Settings → Privacy & Security → Camera")
        print("Make sure Terminal / IDE has camera access.")
        sys.exit(1)
    cur_stroke = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        draw_character(frame, character, (0, 0, w, h), current_stroke_idx = cur_stroke)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord("w"):
            cur_stroke += 1
        elif key == ord("s"):
            cur_stroke -= 1

    cap.release()
    cv2.destroyAllWindows()
