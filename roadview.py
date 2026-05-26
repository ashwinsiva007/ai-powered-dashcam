"""
DV 2.py - Integrated Roadview System
Combines: Pedestrian/Animal detection, Vehicle proximity, Accident detection, and Speed Limit recognition.
"""
import cv2
import numpy as np
import time
import threading
import argparse
import sys
import os
from collections import defaultdict, deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not found.")

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("[WARN] roboflow not found.")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "vehicle_classes": {
        1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
    },
    "two_wheeler_classes": {1: "bicycle", 3: "motorcycle"},
    "person_classes": {0: "person"},
    "animal_classes": {15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"},
    
    "danger_threshold":  0.55,
    "warning_threshold": 0.35,
    "caution_threshold": 0.18,
    "beep_cooldown": 1.5,
    "confidence": 0.40,
    "frame_skip": 2,
    "window_title": "DASHCAM ROADVIEW (Integrated)",
    "font": cv2.FONT_HERSHEY_DUPLEX,

    # Lane ROI boundaries (ADAS Same-Lane Filtering)
    "lane_top_width": 0.16,
    "lane_bottom_width": 0.70,
    "lane_top_y": 0.50,
    "lane_bottom_y": 1.0,
}

COLORS = {
    "danger":  (0,   30, 255),
    "warning": (0,  140, 255),
    "caution": (0,  220, 255),
    "safe":    (50, 205,  50),
    "two_wheeler": (255, 100,   0),
    "person":  (200, 100, 255),
    "animal":  (100, 200, 255),
    "white":   (255, 255, 255),
    "dim":     (160, 160, 160),
}

# Accident Scorer Config
SCENE_CHANGE_LOW = 10000
SCENE_CHANGE_HIGH = 40000
SCENE_SUSTAIN_FRAMES = 3
OBSTACLE_FRAMES = 15
OBSTACLE_TOLERANCE = 40
ACCIDENT_THRESHOLD = 5

# ══════════════════════════════════════════════════════════════════════════════
#  VOICE & BEEP ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class AlertEngine:
    def __init__(self):
        self.last_beep = 0.0
        self.last_voice = defaultdict(float)
        self.speaking = False
        self.lock = threading.Lock()
        self.cooldowns = {
            "accident": 10, "animal": 7, "person": 7, "vehicle": 8, "low_light": 12, "speed": 10, "blocked": 10
        }
        
        self.sounds = {}
        if PYGAME_AVAILABLE:
            self.sounds = {
                "danger":  self._make_tone(880, 0.18),
                "warning": self._make_tone(660, 0.14),
                "caution": self._make_tone(440, 0.10),
            }

    @staticmethod
    def _make_tone(freq, duration):
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        fade = int(n_samples * 0.10)
        wave[-fade:] = (wave[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
        return pygame.sndarray.make_sound(wave)

    def beep(self, level):
        now = time.time()
        if now - self.last_beep < CONFIG["beep_cooldown"]: return
        self.last_beep = now
        if PYGAME_AVAILABLE and level in self.sounds:
            self.sounds[level].play()
        elif level in ("danger", "warning"):
            self.speak("Vehicle near!")

    def speak(self, msg, key="default"):
        now = time.time()
        with self.lock:
            if self.speaking: return
            if key != "default" and now - self.last_voice[key] < self.cooldowns.get(key, 5): return
            self.speaking = True
            self.last_voice[key] = now
        threading.Thread(target=self._run_tts, args=(msg,), daemon=True).start()

    def _run_tts(self, msg):
        try:
            if TTS_AVAILABLE:
                engine = pyttsx3.init()
                engine.say(msg)
                engine.runAndWait()
        finally:
            with self.lock:
                self.speaking = False

# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND SPEED LIMIT THREAD
# ══════════════════════════════════════════════════════════════════════════════
class SpeedLimitDetector:
    def __init__(self):
        self.current_speed_limit = None
        self.last_check = 0
        self.lock = threading.Lock()
        self.model = None
        if ROBOFLOW_AVAILABLE:
            try:
                rf = Roboflow(api_key="pOQ0BGNHYaSNWyIKtqYY")
                project = rf.workspace("deepaks-workspace-85jkh").project("speed-limit-2")
                self.model = project.version(1).model
                print("[INFO] Roboflow Speed Limit model loaded.")
            except Exception as e:
                print(f"[WARN] Failed to load Roboflow: {e}")

    def update(self, frame):
        now = time.time()
        if self.model and (now - self.last_check > 5.0):
            self.last_check = now
            threading.Thread(target=self._infer, args=(frame.copy(),), daemon=True).start()

    def _infer(self, frame):
        try:
            temp_path = "temp_speed_check.jpg"
            cv2.imwrite(temp_path, frame)
            pred = self.model.predict(temp_path, confidence=40, overlap=30).json()
            if pred and "predictions" in pred and len(pred["predictions"]) > 0:
                highest_conf = max(pred["predictions"], key=lambda x: x["confidence"])
                with self.lock:
                    self.current_speed_limit = highest_conf["class"]
        except Exception as e:
            pass

    def get_limit(self):
        with self.lock:
            return self.current_speed_limit

# ══════════════════════════════════════════════════════════════════════════════
#  TRACKERS & SCORERS
# ══════════════════════════════════════════════════════════════════════════════
class AccidentScorer:
    def __init__(self):
        self.chg_hist = deque(maxlen=SCENE_SUSTAIN_FRAMES)
        self.prev_count = 0

    def score(self, chg, num_boxes):
        total = 0
        if chg > SCENE_CHANGE_HIGH: total += 2
        self.chg_hist.append(chg)
        if len(self.chg_hist) == SCENE_SUSTAIN_FRAMES and all(v > SCENE_CHANGE_LOW for v in self.chg_hist):
            total += 1
        if self.prev_count - num_boxes >= 2:
            total += 1
        self.prev_count = num_boxes
        return total

def get_lighting(frame):
    b = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return "Day" if b >= 100 else "Low Light" if b >= 50 else "Night"

def proximity_level(bbox, frame_h):
    _, y1, _, y2 = bbox
    ratio = (y2 - y1) / frame_h
    if ratio >= CONFIG["danger_threshold"]: return "danger"
    elif ratio >= CONFIG["warning_threshold"]: return "warning"
    elif ratio >= CONFIG["caution_threshold"]: return "caution"
    return "safe"

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
def run(source):
    if not YOLO_AVAILABLE:
        print("[ERROR] ultralytics is required.")
        return

    model = YOLO("yolov8n.pt")
    alert_engine = AlertEngine()
    speed_detector = SpeedLimitDetector()
    scorer = AccidentScorer()
    
    cap = cv2.VideoCapture(source)
    ret, prev_frame = cap.read()
    if not ret: return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    fps_timer = time.time()
    frame_count = 0
    fps = 0.0
    skip = CONFIG["frame_skip"]
    detections = []

    print("[INFO] Roadview integrated system running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret: break
            else:
                break

        frame_count += 1
        h, w = frame.shape[:2]
        
        # Calculate active driving lane polygon (trapezoid)
        cx_frame = w // 2
        y_top = int(h * CONFIG["lane_top_y"])
        y_bottom = int(h * CONFIG["lane_bottom_y"])
        w_top = int(w * CONFIG["lane_top_width"])
        w_bottom = int(w * CONFIG["lane_bottom_width"])
        
        lane_poly = np.array([
            [cx_frame - w_top // 2, y_top],
            [cx_frame + w_top // 2, y_top],
            [cx_frame + w_bottom // 2, y_bottom],
            [cx_frame - w_bottom // 2, y_bottom]
        ], dtype=np.int32)
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chg = int(np.sum(cv2.absdiff(prev_gray, curr_gray)))
        prev_gray = curr_gray
        lighting = get_lighting(frame)
        # Check if view is >= 80% blocked by analyzing grid patches
        grid_h, grid_w = 10, 10
        h_step, w_step = h // grid_h, w // grid_w
        blocked_patches = 0
        for r in range(grid_h):
            for c in range(grid_w):
                patch = curr_gray[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
                if np.std(patch) < 15:
                    blocked_patches += 1
                    
        view_blocked = (blocked_patches / (grid_h * grid_w)) >= 0.80

        # Trigger background speed limit check
        speed_detector.update(frame)
        
        close_persons = False
        close_animals = False

        if frame_count % skip == 0:
            results = model(frame, conf=CONFIG["confidence"], verbose=False)[0]
            detections = []
            
            if results.boxes:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    obj_type = None
                    label = ""
                    if cls_id in CONFIG["vehicle_classes"]:
                        obj_type = "vehicle"
                        label = CONFIG["vehicle_classes"][cls_id]
                    elif cls_id in CONFIG["person_classes"]:
                        obj_type = "person"
                        label = "person"
                    elif cls_id in CONFIG["animal_classes"]:
                        obj_type = "animal"
                        label = CONFIG["animal_classes"][cls_id]

                    if obj_type:
                        # Same-lane ADAS check using bottom-center contact point
                        cx = (x1 + x2) // 2
                        cy = y2
                        in_lane = cv2.pointPolygonTest(lane_poly, (float(cx), float(cy)), False) >= 0
                        
                        # Apply warning/caution levels only for objects inside our lane
                        lvl = proximity_level((x1, y1, x2, y2), h) if in_lane else "safe"
                        
                        det = {
                            "bbox": (x1, y1, x2, y2), "label": label, "conf": conf,
                            "type": obj_type, "is_two_wheeler": cls_id in CONFIG["two_wheeler_classes"],
                            "in_lane": in_lane, "level": lvl
                        }
                        detections.append(det)
                        
                        # Check if person or animal is close in our lane
                        if lvl in ("danger", "warning"):
                            if obj_type == "person":
                                close_persons = True
                            elif obj_type == "animal":
                                close_animals = True

            # Voice Alerts for close pedestrians/animals
            if close_animals: alert_engine.speak("Animal close to the road", "animal")
            elif close_persons: alert_engine.speak("Pedestrian very close", "person")
            
            # View Blocked or Low light warning
            if view_blocked:
                alert_engine.speak("Camera view is blocked", "blocked")
            elif lighting in ("Low Light", "Night"): 
                alert_engine.speak("Low visibility, drive carefully", "low_light")
            
            # Accident score
            score = scorer.score(chg, len(detections))
            if score >= ACCIDENT_THRESHOLD:
                alert_engine.speak("Possible accident detected", "accident")

            # Beeps for vehicle proximity (restricted to same lane)
            levels = [d["level"] for d in detections if d["type"] == "vehicle"]
            if "danger" in levels: alert_engine.beep("danger")
            elif "warning" in levels: alert_engine.beep("warning")
            elif "caution" in levels: alert_engine.beep("caution")

        # FPS
        now = time.time()
        elapsed = now - fps_timer
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = now

        # HUD Drawing
        cv2.rectangle(frame, (0, 0), (w, 45), (10, 10, 10), -1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Light: {lighting}", (12, 28), CONFIG["font"], 0.55, COLORS["dim"], 1)
        
        limit = speed_detector.get_limit()
        if limit:
            cv2.putText(frame, f"Speed Limit: {limit}", (w - 300, 30), CONFIG["font"], 0.75, (0, 255, 255), 2)

        any_danger = False
        any_warning = False
        pedestrian_danger = False

        for det in detections:
            lvl = det.get("level", "safe")
            if det["type"] == "vehicle":
                if lvl == "danger": any_danger = True
                elif lvl == "warning": any_warning = True
            elif det["type"] == "person":
                if lvl in ("danger", "warning"):
                    pedestrian_danger = True


        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            lvl = det.get("level", "safe")
            
            if det["type"] == "vehicle":
                col = COLORS[lvl] if lvl != "safe" else (COLORS["two_wheeler"] if det["is_two_wheeler"] else COLORS["safe"])
                thick = {"danger": 4, "warning": 3, "caution": 2, "safe": 1}[lvl]
            elif det["type"] == "person":
                col = COLORS["person"]
                thick = 3 if lvl in ("danger", "warning") else 2
            elif det["type"] == "animal":
                col = COLORS["animal"]
                thick = 3 if lvl in ("danger", "warning") else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
            label_txt = f"{det['label']} {det['conf']:.0%}"
            cv2.putText(frame, label_txt, (x1, y1 - 5), CONFIG["font"], 0.5, col, 1)

        if view_blocked:
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 255), -1)
            cv2.putText(frame, "⚠ CAMERA BLOCKED!", (20, h - 15), CONFIG["font"], 0.8, COLORS["white"], 2)
        elif any_danger:
            cv2.rectangle(frame, (0, h - 50), (w, h), COLORS["danger"], -1)
            cv2.putText(frame, "⚠ COLLISION RISK!", (20, h - 15), CONFIG["font"], 0.8, COLORS["white"], 2)
        elif pedestrian_danger:
            cv2.rectangle(frame, (0, h - 50), (w, h), COLORS["person"], -1)
            cv2.putText(frame, "⚠ CLOSE PEDESTRIAN!", (20, h - 15), CONFIG["font"], 0.8, COLORS["white"], 2)
        elif any_warning:
            cv2.rectangle(frame, (0, h - 50), (w, h), COLORS["warning"], -1)
            cv2.putText(frame, "CLOSE VEHICLE", (20, h - 15), CONFIG["font"], 0.7, (10, 10, 10), 2)

        cv2.imshow(CONFIG["window_title"], frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    args = parser.parse_args()
    
    src = args.source
    if src is None:
        print("=========================================================")
        print("               DASHCAM ROADVIEW SYSTEM                   ")
        print("=========================================================")
        video_path = input("Enter video file path (or press [Enter] to use camera): ").strip()
        
        # Strip quotes if copied from file explorer
        if (video_path.startswith('"') and video_path.endswith('"')) or (video_path.startswith("'") and video_path.endswith("'")):
            video_path = video_path[1:-1].strip()
            
        if video_path == "":
            print("[INFO] Attempting to connect to USB webcam (Index 1)...")
            cap = cv2.VideoCapture(1)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    src = 1
                    print("[INFO] Successfully connected to USB webcam (Index 1).")
                else:
                    src = None
                cap.release()
            else:
                src = None
            
            if src is None:
                print("[INFO] USB webcam not found/active. Falling back to laptop camera (Index 0)...")
                src = 0
        else:
            src = video_path
    else:
        src = int(src) if str(src).isdigit() else src

    run(src)
