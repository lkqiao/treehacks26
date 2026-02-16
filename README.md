# Freestroke

###Inspiration 
Learning character-based languages like Chinese can feel overwhelming, especially online. Apps often reduce writing to tapping pre-made strokes or tracing on a screen, which doesn’t build true muscle memory. Inspired by the physicality of handwriting practice and the importance of stroke order in languages like Chinese, we wanted to recreate the embodied experience of writing, but without pen and paper. 

Freestroke was born from the idea that language learning should be active, immersive, and intuitive. We combine motion, haptics, and computer vision, turning the air around you into a canvas.

### What it does
Freestroke is an interactive language-learning app that lets users learn Chinese and other character-based languages by drawing characters in the air.

Using computer vision, the system detects and interprets the user’s hand movements as character strokes in real time. The haptic feedback helps enhance the sense of physical interaction, making the input feel grounded and intentional. The app also provides accuracy feedback and signals mistakes, helping users internalize stroke order and structure.

The app includes three core modes:
- Practice Mode: Guided character writing with stroke-by-stroke feedback.
- Test Mode: Independent writing with scoring based on accuracy and stroke order.
- Comprehension Mode: Reinforces recognition and meaning through reading and contextual exercises.

Together, these modes build muscle memory, accuracy, and understanding.

### How we built it
Freestroke is implemented as a real-time multimodal system primarily implemented in Python using OpenCV and MediaPipe. We used the MakeMeAHanzi dataset for the ground-truth Chinese strokes in our character stroke detection algorithm.

The system integrates:
- MediaPipe hand tracking 
- Pinch-based gesture control for air drawing
- Character stroke detection
- WiFi-based custom haptic feedback system
- Real-time UI overlay

The architecture is event-driven and frame-synchronous at ~30 FPS.

## Features

### Teaching Mode
- See outline of Chinese character, English definition, pinyin
- Real-time arrows showing stroke direction and feedback with character stroke accuracy 
- Complete one stroke at a time with visual feedback

### Pinyin Recognition Mode
- See the pinyin and recall the character
- Real-time feedback on stroke accuracy

### English Translation Mode
- See English word, write the corresponding character

### WiFi-Based Custom Haptic Feedback System 
When the "pen-down" signal is received, the computer sends a WiFi signal to the ESP8266 microcontroller that is connected to the same network. The microcontroller sets one of its GPIO pins high, turning on a NPN transistor that allows for the required current to be supplied from the power source to power a DC motor and LED. We attached an asymmetrical servo arm to the DC motor to generate the vibrations. The firmware also constantly checks the strength of the WiFi RSSI level (typically around -20dB to -30dB) and continuously monitors the connection.

### Real-Time UI Overlay 
The UI overlay in the Zoom camera feed displays buttons for each of the modes that can be selected using the same pinch gesture for writing. We also allow the user to rescale the size of a bounding box that defines the area in which the character is drawn. We also display the stroke feedback and accuracy metrics so that they are easily visible to the user.

---

## Quick Start

### Installation

```bash
cd /Users/lukeqiao/Documents/Projects/treehacks_2026
uv venv
uv sync
uv run launcher.py
```

### First Run
1. Press **1**, **2**, or **3** to choose a learning mode
2. Start writing characters with your finger in front of the camera
3. Press **SPACE** to submit your work or move to the next character

---

## Learning Modes

### Teaching Mode (Press 1)
Learn proper stroke technique with guided instructions.

```
Teaching Mode - 一 (one)
Pinyin: yi1

    ➜ ➜ ➜ ➜  (animated arrow)
    ─────────────────  (stroke 1)

Stroke 1 / 1
```

**How it works:**
- Animated arrow shows stroke direction
- Semi-transparent guide shows exact path
- Real-time validation after each stroke
- Move to next stroke automatically on success

### Pinyin Recognition Mode (Press 2)
Practice recalling characters from their sound.

```
Pinyin Mode
Write the character that sounds like: shui3 (water)

    [Your drawing here]

Score: 0150
```

**Scoring:**
- Correct character: +150 points
- Incorrect: 0 points (try again)
- Build your vocabulary systematically

### English Translation Mode (Press 3)
The ultimate memory challenge with gamification.

```
Translation Mode
Write the character for: WATER

    [Your drawing here]

Completed: 5        Score: 00850
✓ Correct! 水
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| **1** | Teaching Mode |
| **2** | Pinyin Recognition Mode |
| **3** | English Translation Mode |
| **SPACE** | Submit drawing / Next character |
| **C** | Clear current drawing |
| **M** | Return to mode selection |
| **Q** | Quit application |

---

## Zoom Integration

### Setup for Virtual Teaching

1. **Start Zoom meeting**
2. **Launch tutor**: `python main_app.py`
3. **Share screen**: Click "Share Screen" → Select tutor window
4. **Everyone sees**: Live character learning with feedback!

### Perfect For:
- Virtual Chinese classes
- Group tutoring sessions
- Hybrid learning (in-person + Zoom)
- Student demonstrations
- Interactive practice sessions

### Best Practices:
- Use **Teaching Mode** for demonstrations
- Use **Pinyin/English Modes** for interactive practice
- Ask students to draw in their own cameras while you evaluate
- Keep window at native resolution for clarity

---

## 🔧 Technical Architecture

### Core Components

**`main_app.py`** - Main application
- Mode management, state, scoring, event handling

**`stroke_engine.py`** - Recognition engine
- DTW stroke matching algorithm
- Character recognition & validation
- Stroke order verification

**`ui_renderer.py`** - Visual rendering
- Template stroke drawing
- Animated arrow generation
- UI panels & feedback

**`zoom_integration.py`** - Zoom optimization
- Screen share compatibility
- Setup instructions

### Stroke Matching Algorithm

We use **Dynamic Time Warping (DTW)** with angle scoring:

```
Score = DTW_Distance + (Angle_Penalty × 0.3)
Match = Score ≤ Threshold
```

**Why DTW?**
- ✓ Handles speed variations (fast/slow writing)
- ✓ Accounts for individual handwriting styles
- ✓ Sensitive to stroke direction mistakes
- ✓ Forgiving for minor deviations

---

## Installation & Troubleshooting

### Prerequisites
- Python 3.8+
- Webcam with 30+ FPS
- Camera permissions enabled

### Troubleshooting

**"Camera failed to initialize"**
- macOS: Settings → Privacy & Security → Camera → Grant access to Terminal/IDE
- Windows: Check camera in Device Manager
- Restart the application

**"Strokes not detecting"**
- Improve lighting in your room
- Move finger closer to camera (but keep full hand visible)
- Ensure minimum finger movement (MOVE_THRESHOLD = 5 pixels)

**"Strokes not matching"**
- Stroke must be drawn in the correct direction
- Try different handwriting styles - it learns from you
- Practice a few times - accuracy improves with familiarity

**"Low FPS / Lag"**
- Close unnecessary applications
- Update camera driver
- Lower resolution if needed (edit WINDOW_WIDTH/HEIGHT in main_app.py)

---

## File Structure

```
treehacks_2026/
├── main_app.py              # Main application (run this!)
├── stroke_engine.py         # Recognition & validation
├── ui_renderer.py           # Visual rendering
├── zoom_integration.py      # Zoom setup & helpers
├── characters.json          # Character database
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Tech Stack

- **MediaPipe**: Real-time hand detection
- **OpenCV**: Image processing & rendering
- **DTW Algorithm**: Dynamic Time Warping for stroke matching
- **HSK Standard**: Chinese learning framework
