"""
ArthoSense NER - High-Accuracy Computer Vision Kinematics Module
Integrates MediaPipe Pose Landmarker, CLAHE Low-Light Image Enhancement, 
Left & Right Knee Flexion tracking, Trunk Sway calculation, Gait Asymmetry Index,
and Live Rolling Angle Graph Rendering.
"""

import cv2
import numpy as np
import math
import time
import os
import urllib.request
from collections import deque
from typing import Tuple, Optional, Dict, Any

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

# Download model asset if missing
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pose_landmarker_lite.task')
def ensure_model_download():
    if not os.path.exists(MODEL_PATH):
        try:
            print("Downloading MediaPipe AI pose model (pose_landmarker_lite.task)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("Download complete!")
        except Exception as e:
            print(f"Warning: Could not download MediaPipe task model: {e}")

def calculate_joint_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calculates 3D/2D joint angle at vertex 'b' formed by points a-b-c in degrees.
    """
    a_arr, b_arr, c_arr = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c_arr[1]-b_arr[1], c_arr[0]-b_arr[0]) - np.arctan2(a_arr[1]-b_arr[1], a_arr[0]-b_arr[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)

def calculate_gait_asymmetry(l_rom: float, r_rom: float) -> Tuple[float, str]:
    """
    Asymmetry Index Formula: abs(L_ROM - R_ROM) / max(L_ROM, R_ROM) * 100
    Returns asymmetry percentage and diagnostic classification string.
    """
    max_rom = max(l_rom, r_rom)
    if max_rom < 1.0:
        return 0.0, "NORMAL (Symmetrical Gait)"
    
    asymmetry = (abs(l_rom - r_rom) / max_rom) * 100.0
    if asymmetry < 5.0:
        diag = "NORMAL (Symmetrical Gait)"
    elif asymmetry < 15.0:
        diag = "MILD/MODERATE OA RISK (Compensatory Gait)"
    else:
        diag = "SEVERE OA RISK (Antalgic / Highly Asymmetrical Gait)"
        
    return round(asymmetry, 1), diag

def apply_clahe_enhancement(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for indoor low-light enhancement.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

class DualLegKinematicsTracker:
    """
    Tracks Left & Right Knee Flexion, Trunk Sway, Gait Asymmetry, and renders live HUD + rolling graphs.
    """
    def __init__(self, graph_length: int = 100, enable_clahe: bool = True):
        self.enable_clahe = enable_clahe
        self.graph_length = graph_length
        self.left_history = deque(maxlen=graph_length)
        self.right_history = deque(maxlen=graph_length)
        
        self.left_angle = 180.0
        self.right_angle = 180.0
        self.trunk_sway = 0.0
        
        self.left_min, self.left_max = 180.0, 0.0
        self.right_min, self.right_max = 180.0, 0.0
        
        self.landmarker = None
        if MEDIAPIPE_AVAILABLE:
            ensure_model_download()
            if os.path.exists(MODEL_PATH):
                try:
                    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.VIDEO,
                        min_pose_detection_confidence=0.65,
                        min_pose_presence_confidence=0.65,
                        min_tracking_confidence=0.65
                    )
                    self.landmarker = vision.PoseLandmarker.create_from_options(options)
                except Exception as err:
                    print(f"Error initializing MediaPipe PoseLandmarker task: {err}")

    def draw_live_graph_overlay(self, frame: np.ndarray):
        g_w, g_h = 250, 110
        x_offset, y_offset = 20, 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_offset), (x_offset+g_w, y_offset+g_h), (25, 28, 32), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, "Knee Flexion Plot (Deg)", (x_offset + 5, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "180", (x_offset - 22, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(frame, "0", (x_offset - 15, y_offset + g_h), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        if len(self.left_history) > 1:
            l_pts = []
            for i, val in enumerate(self.left_history):
                x = x_offset + int((i / self.graph_length) * g_w)
                y = y_offset + g_h - int((val / 180.0) * g_h)
                l_pts.append((x, y))
            cv2.polylines(frame, [np.array(l_pts)], False, (0, 255, 0), 2)

        if len(self.right_history) > 1:
            r_pts = []
            for i, val in enumerate(self.right_history):
                x = x_offset + int((i / self.graph_length) * g_w)
                y = y_offset + g_h - int((val / 180.0) * g_h)
                r_pts.append((x, y))
            cv2.polylines(frame, [np.array(r_pts)], False, (255, 120, 0), 2)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.enable_clahe:
            processed_frame = apply_clahe_enhancement(frame_bgr)
        else:
            processed_frame = frame_bgr.copy()

        h, w, _ = processed_frame.shape
        metrics = {
            "left_angle": self.left_angle,
            "right_angle": self.right_angle,
            "left_rom": max(0.0, self.left_max - self.left_min),
            "right_rom": max(0.0, self.right_max - self.right_min),
            "trunk_sway": self.trunk_sway,
            "asymmetry_index": 0.0,
            "gait_diagnosis": "Standby",
            "landmarks_detected": False
        }

        if not MEDIAPIPE_AVAILABLE or self.landmarker is None:
            cv2.putText(processed_frame, "MediaPipe Task Model Standby", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return processed_frame, metrics

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.perf_counter_ns() // 1_000_000)
        
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            metrics["landmarks_detected"] = True

            def get_px(lm): return (int(lm.x * w), int(lm.y * h))

            try:
                l_hip, l_knee, l_ankle = [landmarks[23].x, landmarks[23].y], [landmarks[25].x, landmarks[25].y], [landmarks[27].x, landmarks[27].y]
                r_hip, r_knee, r_ankle = [landmarks[24].x, landmarks[24].y], [landmarks[26].x, landmarks[26].y], [landmarks[28].x, landmarks[28].y]

                mid_sh_x = (landmarks[11].x + landmarks[12].x) / 2
                mid_sh_y = (landmarks[11].y + landmarks[12].y) / 2
                mid_hp_x = (landmarks[23].x + landmarks[24].x) / 2
                mid_hp_y = (landmarks[23].y + landmarks[24].y) / 2

                dx = (mid_sh_x - mid_hp_x) * w
                dy = (mid_hp_y - mid_sh_y) * h
                self.trunk_sway = round(np.abs(np.arctan2(dx, dy) * 180.0 / np.pi), 1)

                self.left_angle = round(calculate_joint_angle(l_hip, l_knee, l_ankle), 1)
                self.right_angle = round(calculate_joint_angle(r_hip, r_knee, r_ankle), 1)

                self.left_min = min(self.left_min, self.left_angle)
                self.left_max = max(self.left_max, self.left_angle)
                self.right_min = min(self.right_min, self.right_angle)
                self.right_max = max(self.right_max, self.right_angle)

                self.left_history.append(self.left_angle)
                self.right_history.append(self.right_angle)

                l_hip_px, l_knee_px, l_ankle_px = get_px(landmarks[23]), get_px(landmarks[25]), get_px(landmarks[27])
                r_hip_px, r_knee_px, r_ankle_px = get_px(landmarks[24]), get_px(landmarks[26]), get_px(landmarks[28])
                mid_sh_px = (int(mid_sh_x * w), int(mid_sh_y * h))
                mid_hp_px = (int(mid_hp_x * w), int(mid_hp_y * h))

                # Draw Left Leg (Green)
                cv2.line(processed_frame, l_hip_px, l_knee_px, (0, 255, 0), 3)
                cv2.line(processed_frame, l_knee_px, l_ankle_px, (0, 255, 0), 3)
                cv2.circle(processed_frame, l_knee_px, 6, (0, 0, 255), -1)
                cv2.putText(processed_frame, f"L: {int(self.left_angle)}deg", (l_knee_px[0] + 10, l_knee_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw Right Leg (Blue)
                cv2.line(processed_frame, r_hip_px, r_knee_px, (255, 120, 0), 3)
                cv2.line(processed_frame, r_knee_px, r_ankle_px, (255, 120, 0), 3)
                cv2.circle(processed_frame, r_knee_px, 6, (0, 0, 255), -1)
                cv2.putText(processed_frame, f"R: {int(self.right_angle)}deg", (r_knee_px[0] + 10, r_knee_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)

                # Draw Trunk Sway (Purple)
                cv2.line(processed_frame, mid_sh_px, mid_hp_px, (255, 0, 255), 3)
                cv2.putText(processed_frame, f"Sway: {self.trunk_sway}deg", (mid_sh_px[0] + 10, mid_sh_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

                l_rom = self.left_max - self.left_min
                r_rom = self.right_max - self.right_min
                asym_val, diag = calculate_gait_asymmetry(l_rom, r_rom)

                metrics.update({
                    "left_angle": self.left_angle,
                    "right_angle": self.right_angle,
                    "left_rom": round(l_rom, 1),
                    "right_rom": round(r_rom, 1),
                    "trunk_sway": self.trunk_sway,
                    "asymmetry_index": asym_val,
                    "gait_diagnosis": diag
                })
            except IndexError:
                pass

        self.draw_live_graph_overlay(processed_frame)
        return processed_frame, metrics

def generate_simulated_kinematic_frame(frame_num: int, target_max_rom: float = 125.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (28, 32, 38)

    t = frame_num * 0.08
    left_norm = (math.sin(t) + 1.0) / 2.0
    right_norm = (math.sin(t + 0.5) + 1.0) / 2.0

    left_angle = round(170.0 - left_norm * (170.0 - target_max_rom), 1)
    right_angle = round(170.0 - right_norm * (170.0 - (target_max_rom + 8.0)), 1)
    trunk_sway = round(abs(math.sin(t * 0.5)) * 6.5, 1)

    hip_l = (280, 140)
    hip_r = (360, 140)

    knee_l = (int(hip_l[0] + 120 * math.sin(math.radians(20))), int(hip_l[1] + 120 * math.cos(math.radians(20))))
    knee_r = (int(hip_r[0] + 120 * math.sin(math.radians(20))), int(hip_r[1] + 120 * math.cos(math.radians(20))))

    ankle_l = (int(knee_l[0] + 120 * math.sin(math.radians(20 - (180 - left_angle)))), int(knee_l[1] + 120 * math.cos(math.radians(20 - (180 - left_angle)))))
    ankle_r = (int(knee_r[0] + 120 * math.sin(math.radians(20 - (180 - right_angle)))), int(knee_r[1] + 120 * math.cos(math.radians(20 - (180 - right_angle)))))

    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (38, 44, 52), 1)
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (38, 44, 52), 1)

    cv2.line(frame, hip_l, knee_l, (0, 255, 0), 3)
    cv2.line(frame, knee_l, ankle_l, (0, 255, 0), 3)
    cv2.circle(frame, knee_l, 8, (0, 0, 255), -1)

    cv2.line(frame, hip_r, knee_r, (255, 120, 0), 3)
    cv2.line(frame, knee_r, ankle_r, (255, 120, 0), 3)
    cv2.circle(frame, knee_r, 8, (0, 0, 255), -1)

    cv2.line(frame, (320, 60), (320, 140), (255, 0, 255), 3)

    cv2.rectangle(frame, (20, 20), (380, 145), (18, 20, 24), -1)
    cv2.rectangle(frame, (20, 20), (380, 145), (0, 220, 100), 2)
    cv2.putText(frame, f"L KNEE: {left_angle:.1f}deg | R KNEE: {right_angle:.1f}deg", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"TRUNK SWAY: {trunk_sway:.1f}deg", (30, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    cv2.putText(frame, "Simulated Kinematics Engine (MediaPipe Mode)", (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    l_rom = 170.0 - target_max_rom
    r_rom = 170.0 - (target_max_rom + 8.0)
    asym_val, diag = calculate_gait_asymmetry(l_rom, r_rom)

    metrics = {
        "left_angle": left_angle,
        "right_angle": right_angle,
        "left_rom": round(l_rom, 1),
        "right_rom": round(r_rom, 1),
        "trunk_sway": trunk_sway,
        "asymmetry_index": asym_val,
        "gait_diagnosis": diag,
        "landmarks_detected": True
    }
    return frame, metrics
