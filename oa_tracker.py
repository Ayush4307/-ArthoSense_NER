import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import csv
import os
import sys
import urllib.request
from collections import deque

# 1. Download Model if Missing
model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_lite.task')
if not os.path.exists(model_path):
    print("Downloading MediaPipe AI pose model (pose_landmarker_lite.task)...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Angle calculation math
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# 3. Live Graph Renderer
GRAPH_LENGTH = 100
left_history = deque(maxlen=GRAPH_LENGTH)
right_history = deque(maxlen=GRAPH_LENGTH)

def draw_live_graph(frame, left_data, right_data):
    g_w, g_h = 250, 120
    x_offset, y_offset = 20, 20
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, y_offset), (x_offset+g_w, y_offset+g_h), (30,30,30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "Knee Flexion (Deg)", (x_offset + 5, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(frame, "180", (x_offset - 20, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)
    cv2.putText(frame, "0", (x_offset - 15, y_offset + g_h), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)

    if len(left_data) > 1:
        l_pts = []
        for i, val in enumerate(left_data):
            x = x_offset + int((i / GRAPH_LENGTH) * g_w)
            y = y_offset + g_h - int((val / 180.0) * g_h)
            l_pts.append((x, y))
        cv2.polylines(frame, [np.array(l_pts)], False, (0, 255, 0), 2)

    if len(right_data) > 1:
        r_pts = []
        for i, val in enumerate(right_data):
            x = x_offset + int((i / GRAPH_LENGTH) * g_w)
            y = y_offset + g_h - int((val / 180.0) * g_h)
            r_pts.append((x, y))
        cv2.polylines(frame, [np.array(r_pts)], False, (255, 100, 0), 2)

def run_oa_tracker(camera_index=0, enable_clahe=True, log_csv_path='knee_angles_log.csv'):
    # Setup MediaPipe Tasks API
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened() and camera_index != 0:
        print(f"Camera index {camera_index} failed, falling back to camera index 0...")
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting Camera... Step back to show full body. Press 'q' to quit.")

    with open(log_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Left_Knee_Angle', 'Right_Knee_Angle', 'Trunk_Sway'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            if enable_clahe:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(time.perf_counter_ns() // 1_000_000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                h, w, _ = frame.shape
                def get_px(lm): return (int(lm.x * w), int(lm.y * h))
                
                try:
                    l_hip, l_knee, l_ankle = [landmarks[23].x, landmarks[23].y], [landmarks[25].x, landmarks[25].y], [landmarks[27].x, landmarks[27].y]
                    r_hip, r_knee, r_ankle = [landmarks[24].x, landmarks[24].y], [landmarks[26].x, landmarks[26].y], [landmarks[28].x, landmarks[28].y]
                    
                    mid_shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
                    mid_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
                    mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
                    mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2

                    dx = (mid_shoulder_x - mid_hip_x) * w
                    dy = (mid_hip_y - mid_shoulder_y) * h
                    trunk_sway = np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)

                    left_angle = calculate_angle(l_hip, l_knee, l_ankle)
                    right_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    
                    left_history.append(left_angle)
                    right_history.append(right_angle)
                    
                    l_hip_px, l_knee_px, l_ankle_px = get_px(landmarks[23]), get_px(landmarks[25]), get_px(landmarks[27])
                    r_hip_px, r_knee_px, r_ankle_px = get_px(landmarks[24]), get_px(landmarks[26]), get_px(landmarks[28])
                    mid_sh_px = (int(mid_shoulder_x * w), int(mid_shoulder_y * h))
                    mid_hip_px = (int(mid_hip_x * w), int(mid_hip_y * h))
                    
                    # Draw Left Leg (Green)
                    cv2.line(frame, l_hip_px, l_knee_px, (0, 255, 0), 3)
                    cv2.line(frame, l_knee_px, l_ankle_px, (0, 255, 0), 3)
                    cv2.circle(frame, l_knee_px, 6, (0, 0, 255), -1)
                    cv2.putText(frame, f"L: {int(left_angle)}", (l_knee_px[0] + 15, l_knee_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Draw Right Leg (Blue)
                    cv2.line(frame, r_hip_px, r_knee_px, (255, 100, 0), 3)
                    cv2.line(frame, r_knee_px, r_ankle_px, (255, 100, 0), 3)
                    cv2.circle(frame, r_knee_px, 6, (0, 0, 255), -1)
                    cv2.putText(frame, f"R: {int(right_angle)}", (r_knee_px[0] + 15, r_knee_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

                    # Draw Trunk Sway Line (Purple)
                    cv2.line(frame, mid_sh_px, mid_hip_px, (255, 0, 255), 4)
                    cv2.putText(frame, f"Sway: {int(trunk_sway)} Deg", (mid_sh_px[0] + 15, mid_sh_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    writer.writerow([time.time(), left_angle, right_angle, trunk_sway])
                except IndexError:
                    pass
            
            draw_live_graph(frame, left_history, right_history)
            
            cv2.imshow('ArthoSense NER - Computer Vision Kinematics', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_idx = 0
    if len(sys.argv) > 1:
        cam_idx = int(sys.argv[1])
    run_oa_tracker(camera_index=cam_idx)
