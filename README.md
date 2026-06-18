<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:0f2027,100:1a1a2e&height=220&section=header&text=AI-Powered%20Dashcam&fontSize=48&fontColor=58a6ff&animation=fadeIn&fontAlignY=38&desc=Dual-Camera%20Road%20Safety%20System%20%7C%20Real-Time%20ADAS&descAlignY=58&descColor=8b949e" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-27338e?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![Roboflow](https://img.shields.io/badge/Roboflow-API-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **🚗 A real-time, dual-camera Advanced Driver Assistance System (ADAS) that detects road hazards, monitors vehicle proximity, recognizes speed limits, and delivers instant voice + audio alerts — all running locally on a standard webcam.**

</div>

---

## 🗺️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  AI-POWERED DASHCAM                     │
├────────────────────┬────────────────────────────────────┤
│   driverview.py    │         roadview.py                │
│  (Driver Camera)   │      (Road / Front Camera)        │
│  USB Webcam [1]    │       Laptop Camera [0]           │
├────────────────────┴────────────────────────────────────┤
│                   YOLOv8n Detection                     │
│         Vehicles │ Pedestrians │ Animals                │
├─────────────────────────────────────────────────────────┤
│              ADAS Same-Lane Filtering                   │
│         Trapezoidal ROI → Lane-Only Warnings           │
├───────────┬─────────────┬──────────────────────────────┤
│ Proximity │  Accident   │  Speed Limit   │ Environment │
│  Scoring  │  Detection  │  (Roboflow AI) │  Analysis   │
├───────────┴─────────────┴──────────────────────────────┤
│          Voice Alerts (pyttsx3) + Beep (pygame)        │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🎯 Real-Time Object Detection (YOLOv8n)
Detects and classifies objects into three categories simultaneously:

| Category | Objects Detected |
|----------|-----------------|
| 🚗 **Vehicles** | Car, Bus, Truck, Motorcycle, Bicycle |
| 🚶 **Pedestrians** | Person |
| 🐄 **Animals** | Cat, Dog, Horse, Cow, Elephant, Bear, Zebra, Giraffe + more |

---

### 🛣️ ADAS Same-Lane Proximity Filtering
The system only warns about objects **inside your driving lane** using a dynamic trapezoidal ROI — eliminating false alerts from vehicles on adjacent lanes.

```
          ┌──────┐          ← Lane Top (16% frame width)
         /        \
        /          \
       /            \
      └──────────────┘     ← Lane Bottom (70% frame width)
```

**Proximity Levels** (based on bounding box height ratio):

| Level | Threshold | Color | Action |
|-------|-----------|-------|--------|
| 🔴 **DANGER** | ≥ 55% frame height | Red | High-pitch beep + HUD alert |
| 🟠 **WARNING** | ≥ 35% frame height | Orange | Mid-pitch beep + HUD alert |
| 🟡 **CAUTION** | ≥ 18% frame height | Yellow | Low beep |
| 🟢 **SAFE** | < 18% frame height | Green | No alert |

---

### 🚨 Alert System

**Voice Alerts** (pyttsx3 — spoken audio):
- *"Animal close to the road"* — animal in lane proximity
- *"Pedestrian very close"* — pedestrian in danger/warning zone
- *"Camera view is blocked"* — 80%+ of frame is occluded
- *"Low visibility, drive carefully"* — night/low light detected
- *"Possible accident detected"* — accident scoring threshold exceeded

**Audio Beeps** (pygame — synthesized tones):
| Alert Level | Frequency | Duration |
|-------------|-----------|---------|
| Danger | 880 Hz | 180ms |
| Warning | 660 Hz | 140ms |
| Caution | 440 Hz | 100ms |

---

### 🏎️ Speed Limit Recognition
A **background thread** submits road frames to a **Roboflow-hosted Speed Limit Detection model** every 5 seconds — detecting speed limit signs from live video and displaying them on the HUD in real-time.

---

### 💥 Accident Detection Engine (`AccidentScorer`)
Multi-signal scoring system that raises an alert when any combination of these triggers is detected:

- **Frame Change Intensity**: High pixel difference (`cv2.absdiff`) between consecutive frames
- **Sustained Scene Change**: Scene change above threshold for 3+ consecutive frames
- **Object Count Drop**: 2+ objects suddenly disappearing from frame (e.g. vehicles passing out of view at impact)

---

### 🌙 Environmental Awareness

| Condition | Detection Method | Response |
|-----------|-----------------|----------|
| **Night** | Frame brightness < 50 | Voice alert |
| **Low Light** | Frame brightness 50–99 | Voice alert |
| **Camera Blocked** | ≥80% grid patches with std < 15 | Voice alert + red HUD bar |

---

### 📟 Live HUD Overlay
Real-time heads-up display rendered on the video feed:
- **FPS counter** — live frame rate
- **Lighting condition** — Day / Low Light / Night
- **Speed Limit sign** — when detected by Roboflow
- **Status bar** — COLLISION RISK / CLOSE PEDESTRIAN / CLOSE VEHICLE / CAMERA BLOCKED

---

## 📁 Project Structure

```
ai-powered-dashcam/
│
├── driverview.py     # Driver-facing camera system (USB Webcam / Index 1)
├── roadview.py       # Road-facing camera system (Laptop Camera / Index 0)
└── README.md
```

> **Note:** Both scripts use the same integrated ADAS engine. Run them simultaneously for the full dual-camera experience — `driverview.py` on the interior camera and `roadview.py` on the forward-facing road camera.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Webcam(s) connected

### 1. Clone the Repository
```bash
git clone https://github.com/ashwinsiva007/ai-powered-dashcam.git
cd ai-powered-dashcam
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python pyttsx3 pygame numpy roboflow
```

### 3. YOLOv8 Model
The `yolov8n.pt` model is **automatically downloaded** on first run via Ultralytics. No manual download needed.

---

## 🚀 Usage

### Run the Road Camera (Front View)
```bash
python roadview.py
```

### Run the Driver Camera (Interior View)
```bash
python driverview.py
```

### Use a Video File Instead of Webcam
```bash
python roadview.py --source path/to/video.mp4
```

### Dual-Camera Mode (Two Terminals)
```bash
# Terminal 1 — Road view
python roadview.py

# Terminal 2 — Driver view
python driverview.py
```

When prompted:
- Press **Enter** → uses connected USB webcam (Index 1) or falls back to laptop camera (Index 0)
- Enter a **video file path** → processes that video file

> Press **`Q`** to quit either window.

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 object detection |
| `opencv-python` | Video capture, frame processing, HUD rendering |
| `numpy` | Frame difference analysis, lane polygon math |
| `pyttsx3` | Cross-platform text-to-speech voice alerts |
| `pygame` | Synthesized audio beep alerts |
| `roboflow` | Speed limit sign detection via cloud API |
| `threading` | Non-blocking TTS and API calls |

---

## ⚙️ Configuration

All thresholds and settings are centralized in the `CONFIG` dictionary at the top of each script:

```python
CONFIG = {
    "danger_threshold":  0.55,   # bbox height ratio for danger
    "warning_threshold": 0.35,   # bbox height ratio for warning
    "caution_threshold": 0.18,   # bbox height ratio for caution
    "beep_cooldown":     1.5,    # seconds between audio beeps
    "confidence":        0.40,   # YOLO detection confidence
    "frame_skip":        2,      # process every N frames (performance)
    "lane_top_width":    0.16,   # lane ROI top width (% of frame)
    "lane_bottom_width": 0.70,   # lane ROI bottom width (% of frame)
}
```

---

## 🔮 Roadmap

- [ ] Driver drowsiness detection (eye aspect ratio / facial landmarks)
- [ ] SOS auto-alert integration (SMS / emergency call on accident)
- [ ] Lane departure warning
- [ ] Mobile app integration (Flutter client)
- [ ] Edge deployment (Raspberry Pi / Jetson Nano)
- [ ] Cloud dashboard for trip recording & replay

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:0f2027,100:0d1117&height=120&section=footer" />

**Built with ❤️ by [Ashwin Sivaram](https://github.com/ashwinsiva007)**

*⭐ Star this repo if you find it useful!*

</div>
