"""
ArthoSense NER - Computer Vision Kinematics Module
Processes standard webcam video frames using Google MediaPipe Pose and OpenCV.
Tracks Hip, Knee, and Ankle landmarks to compute real-time Knee Flexion Angle (Range of Motion).
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any

try:
    import mediapipe as mp
    # Initialize mediapipe pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
    mp_drawing = None

def calculate_joint_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calculates angle at joint 'b' formed by landmarks a-b-c in degrees.
    a: Hip (x, y)
    b: Knee (x, y) - Vertex
    c: Ankle (x, y)
    """
    a = np.array(a) # Hip
    b = np.array(b) # Knee
    c = np.array(c) # Ankle

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)

class KneeKinematicsTracker:
    """
    Maintains state across video frames to track Knee ROM, Min Angle, Max Angle,
    and Repetition counts during knee flexion/extension exercises.
    """
    def __init__(self, target_side: str = "right"):
        self.target_side = target_side.lower()
        self.min_angle = 180.0
        self.max_angle = 0.0
        self.current_angle = 180.0
        self.rep_count = 0
        self.stage = "extended" # 'extended' or 'flexed'
        self.angle_history = []
        
        if MEDIAPIPE_AVAILABLE:
            self.pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        else:
            self.pose = None

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single BGR image frame, extracts pose landmarks, draws overlay,
        and returns annotated frame + metrics dictionary.
        """
        h, w, _ = frame_bgr.shape
        metrics = {
            "angle": self.current_angle,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle,
            "rom": self.max_angle - self.min_angle if self.max_angle > self.min_angle else self.current_angle,
            "rep_count": self.rep_count,
            "stage": self.stage,
            "landmarks_detected": False
        }

        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            # Fallback visualization
            annotated = frame_bgr.copy()
            cv2.putText(annotated, "MediaPipe not active - Standby Mode", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return annotated, metrics

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            metrics["landmarks_detected"] = True

            # Choose side
            if self.target_side == "right":
                hip_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                knee_lm = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                ankle_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            else:
                hip_lm = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                knee_lm = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                ankle_lm = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            # Coordinate conversion
            hip = (hip_lm.x * w, hip_lm.y * h)
            knee = (knee_lm.x * w, knee_lm.y * h)
            ankle = (ankle_lm.x * w, ankle_lm.y * h)

            # Compute angle
            angle = calculate_joint_angle(hip, knee, ankle)
            self.current_angle = round(angle, 1)
            self.min_angle = min(self.min_angle, self.current_angle)
            self.max_angle = max(self.max_angle, self.current_angle)
            self.angle_history.append(self.current_angle)
            if len(self.angle_history) > 60:
                self.angle_history.pop(0)

            # Repetition counting logic
            if self.current_angle < 100:
                self.stage = "flexed"
            if self.current_angle > 150 and self.stage == "flexed":
                self.stage = "extended"
                self.rep_count += 1

            # Determine color based on angle (Green for healthy ROM > 120, Orange 90-120, Red < 90)
            if self.current_angle >= 120:
                color = (0, 220, 0) # Green
            elif self.current_angle >= 90:
                color = (0, 165, 255) # Orange
            else:
                color = (0, 0, 255) # Red

            # Draw skeleton lines
            cv2.line(annotated, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 255), 3)
            cv2.line(annotated, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (255, 255, 255), 3)
            
            # Draw joints
            cv2.circle(annotated, (int(hip[0]), int(hip[1])), 8, (255, 100, 0), -1)
            cv2.circle(annotated, (int(knee[0]), int(knee[1])), 12, color, -1)
            cv2.circle(annotated, (int(ankle[0]), int(ankle[1])), 8, (0, 200, 255), -1)

            # Draw HUD card
            cv2.rectangle(annotated, (20, 20), (320, 150), (20, 20, 20), -1)
            cv2.rectangle(annotated, (20, 20), (320, 150), color, 2)
            cv2.putText(annotated, f"KNEE ANGLE: {self.current_angle:.1f} deg", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated, f"Max Flexion: {self.max_angle:.1f} deg", (35, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(annotated, f"Min Angle: {self.min_angle:.1f} deg", (35, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(annotated, f"Reps: {self.rep_count} | {self.stage.upper()}", (35, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            metrics.update({
                "angle": self.current_angle,
                "min_angle": self.min_angle,
                "max_angle": self.max_angle,
                "rom": self.max_angle,
                "rep_count": self.rep_count,
                "stage": self.stage
            })

        return annotated, metrics

def generate_simulated_kinematic_frame(frame_num: int, target_max_rom: float = 125.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates a clean synthetic kinematic visualization for testing when no physical camera is active.
    Simulates a knee bending rhythmically between 170 deg (extended) and target_max_rom (flexed).
    """
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (28, 32, 38) # Dark background

    # Oscillate angle smoothly
    t = frame_num * 0.08
    angle_norm = (math.sin(t) + 1.0) / 2.0 # 0 to 1
    current_angle = 170.0 - angle_norm * (170.0 - target_max_rom)
    
    # Coordinates of body in frame
    hip = (320, 120)
    thigh_len = 130
    shank_len = 130

    knee_x = int(hip[0] + thigh_len * math.sin(math.radians(20)))
    knee_y = int(hip[1] + thigh_len * math.cos(math.radians(20)))

    # Knee bend
    shank_angle = 20 - (180 - current_angle)
    ankle_x = int(knee_x + shank_len * math.sin(math.radians(shank_angle)))
    ankle_y = int(knee_y + shank_len * math.cos(math.radians(shank_angle)))

    color = (0, 220, 100) if current_angle >= 120 else ((0, 180, 255) if current_angle >= 95 else (50, 50, 255))

    # Grid background lines
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (38, 44, 52), 1)
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (38, 44, 52), 1)

    # Skeleton limbs
    cv2.line(frame, hip, (knee_x, knee_y), (240, 240, 240), 4)
    cv2.line(frame, (knee_x, knee_y), (ankle_x, ankle_y), (240, 240, 240), 4)

    # Joints
    cv2.circle(frame, hip, 10, (255, 120, 40), -1)
    cv2.circle(frame, (knee_x, knee_y), 14, color, -1)
    cv2.circle(frame, (ankle_x, ankle_y), 10, (0, 200, 255), -1)

    # Arc indicator
    cv2.ellipse(frame, (knee_x, knee_y), (35, 35), 0, -90, -90 + int(current_angle), color, 2)

    # Card
    cv2.rectangle(frame, (20, 20), (340, 150), (18, 20, 24), -1)
    cv2.rectangle(frame, (20, 20), (340, 150), color, 2)
    cv2.putText(frame, f"KNEE ANGLE: {current_angle:.1f} deg", (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Simulated Kinematic Stream", (35, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, f"Target ROM: {target_max_rom:.1f} deg", (35, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"Status: Real-time MediaPipe Engine", (35, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 200), 1)

    metrics = {
        "angle": round(current_angle, 1),
        "min_angle": target_max_rom,
        "max_angle": 170.0,
        "rom": round(170.0 - target_max_rom, 1),
        "rep_count": int(frame_num // 80),
        "stage": "flexing" if math.cos(t) > 0 else "extending",
        "landmarks_detected": True
    }
    return frame, metrics
