"""
ArthoSense NER - Wearable Sensor Recorder (GUI Launcher)
2-Phase Standby & Recording Mode for MPU6050 IMU + Piezo Acoustic Contact Stethoscope.
Handles both physical USB Serial hardware (COM ports) and Calibrated Field Simulator.
"""

import cv2
import numpy as np
import time
import csv
import os
import sys
import math
from sensor_stream import SensorStreamManager, list_available_com_ports

def run_sensor_recorder():
    joint_condition = sys.argv[1] if len(sys.argv) > 1 else "moderate"
    hardware_mode = sys.argv[2] if len(sys.argv) > 2 else "simulator"
    requested_port = sys.argv[3] if len(sys.argv) > 3 else "AUTO"

    stream_mgr = None
    data_source_label = "Calibrated Field Simulator"
    is_physical = False

    if hardware_mode == "physical":
        ports = list_available_com_ports()
        real_ports = [p for p in ports if "SIMULATED" not in p and "No physical" not in p]
        target_port = requested_port if requested_port in real_ports else (real_ports[0] if real_ports else None)
        
        if target_port:
            stream_mgr = SensorStreamManager(mode="physical", port=target_port)
            if stream_mgr.connect_serial():
                is_physical = True
                data_source_label = f"Physical Hardware ({target_port} @ 115200)"
            else:
                data_source_label = "Simulator (COM Port Unreachable)"
        else:
            data_source_label = "Simulator (No USB COM Hardware Found)"

    if not stream_mgr:
        stream_mgr = SensorStreamManager(mode="simulator")
        if hardware_mode != "physical":
            data_source_label = "Calibrated Field Simulator"

    # Canvas setup
    win_w, win_h = 750, 480
    bg_color = (15, 23, 42) # Slate dark #0f172a
    
    csv_file = "mock_imu_data.csv"
    
    recording = False
    start_time = None
    recorded_rows = []
    
    vibration_buffer = []
    accel_buffer = []
    gyro_buffer = []
    piezo_buffer = []
    max_buf = 100

    print("==================================================")
    print(" ARTHOSENSE WEARABLE SENSOR RECORDER LAUNCHED")
    print("==================================================")
    print(f"Data Source       : {data_source_label}")
    print(f"Condition Profile : {joint_condition}")
    print("Instructions      : Standby Mode active.")
    print("                  : Press 'R' key to start 15-second sensor recording.")
    print("                  : Window auto-closes upon completion.")
    print("==================================================")

    while True:
        canvas = np.full((win_h, win_w, 3), bg_color, dtype=np.uint8)
        sample = stream_mgr.read_sample(joint_condition=joint_condition)
        
        rms = sample["vibration_rms"]
        piezo_raw = sample["piezo_raw"]
        accel_z = abs(sample["accel"]["z"])
        gyro_mag = abs(sample["gyro"]["x"]) + abs(sample["gyro"]["y"])
        
        vibration_buffer.append(rms)
        piezo_buffer.append(piezo_raw)
        accel_buffer.append(accel_z)
        gyro_buffer.append(gyro_mag)
        
        if len(vibration_buffer) > max_buf:
            vibration_buffer.pop(0)
            piezo_buffer.pop(0)
            accel_buffer.pop(0)
            gyro_buffer.pop(0)
            
        current_time = time.time()
        
        if recording:
            elapsed = current_time - start_time
            remaining = max(0.0, duration if 'duration' in locals() else 15.0 - elapsed)
            
            recorded_rows.append({
                "Timestamp": current_time,
                "Vibration_RMS": rms,
                "Accel_Impact": round(accel_z, 3),
                "Gyro_Speed": round(gyro_mag, 1),
                "Acoustic_Signal": round(piezo_raw, 3)
            })
            
            if elapsed >= 15.0:
                try:
                    with open(csv_file, mode="w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Vibration_RMS", "Accel_Impact", "Gyro_Speed", "Acoustic_Signal"])
                        writer.writeheader()
                        writer.writerows(recorded_rows)
                    print(f"✅ Recording Complete! Saved {len(recorded_rows)} sensor frames to {csv_file}")
                except Exception as err:
                    print(f"Error writing sensor CSV: {err}")
                break

        # Render Header
        cv2.rectangle(canvas, (0, 0), (win_w, 60), (30, 41, 59), -1)
        cv2.putText(canvas, "ArthoSense NER - Wearable Sensor Telemetry", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Source Sub-label
        src_color = (34, 197, 94) if is_physical else (234, 179, 8) # Green if physical, Yellow if simulator
        cv2.putText(canvas, f"Source: {data_source_label}", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, src_color, 1)

        # Render Status Badge
        if not recording:
            cv2.rectangle(canvas, (win_w - 280, 15), (win_w - 20, 45), (16, 185, 129), -1) # Green
            cv2.putText(canvas, "STANDBY: PRESS 'R' TO RECORD", (win_w - 272, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            elapsed = current_time - start_time
            remaining = max(0.0, 15.0 - elapsed)
            cv2.rectangle(canvas, (win_w - 280, 15), (win_w - 20, 45), (239, 68, 68), -1) # Red
            cv2.putText(canvas, f"RECORDING: {remaining:.1f}s REMAINING", (win_w - 272, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

        # Render Telemetry Metrics (3 boxes)
        col_w = 220
        # Metric 1: Vibration RMS
        cv2.rectangle(canvas, (20, 80), (20 + col_w, 150), (30, 41, 59), -1)
        cv2.putText(canvas, "Vibration RMS (Piezo)", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
        cv2.putText(canvas, f"{rms:.3f} g", (30, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (34, 197, 94) if rms < 0.40 else (239, 68, 68), 2)
        
        # Metric 2: Peak Impact Accel
        cv2.rectangle(canvas, (260, 80), (260 + col_w, 150), (30, 41, 59), -1)
        cv2.putText(canvas, "Accel Impact (IMU Z)", (270, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
        cv2.putText(canvas, f"{accel_z:.2f} g", (270, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (59, 130, 246), 2)

        # Metric 3: Gyro Speed
        cv2.rectangle(canvas, (500, 80), (500 + col_w, 150), (30, 41, 59), -1)
        cv2.putText(canvas, "Angular Velocity (Gyro)", (510, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
        cv2.putText(canvas, f"{gyro_mag:.1f} deg/s", (510, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (168, 85, 247), 2)

        # Render Live Telemetry Graph Box
        graph_x, graph_y, graph_w, graph_h = 20, 180, 710, 240
        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (30, 41, 59), -1)
        cv2.putText(canvas, "Live Real-Time Acoustic & Inertial Signal Stream", (graph_x + 15, graph_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw legend
        cv2.rectangle(canvas, (graph_x + 380, graph_y + 10), (graph_x + 480, graph_y + 28), (249, 115, 22), -1)
        cv2.putText(canvas, "Acoustic (Piezo)", (graph_x + 390, graph_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        cv2.rectangle(canvas, (graph_x + 490, graph_y + 10), (graph_x + 580, graph_y + 28), (59, 130, 246), -1)
        cv2.putText(canvas, "IMU Accel", (graph_x + 500, graph_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.rectangle(canvas, (graph_x + 590, graph_y + 10), (graph_x + 690, graph_y + 28), (168, 85, 247), -1)
        cv2.putText(canvas, "Gyro Speed", (graph_x + 600, graph_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Plot waveform lines
        if len(piezo_buffer) > 1:
            p_pts, a_pts, g_pts = [], [], []
            for i in range(len(piezo_buffer)):
                px = graph_x + 15 + int((i / max_buf) * (graph_w - 30))
                
                py = graph_y + graph_h - 20 - int(min(1.0, piezo_buffer[i]) * (graph_h - 50))
                p_pts.append((px, py))
                
                ay = graph_y + graph_h - 20 - int(min(2.0, accel_buffer[i]) / 2.0 * (graph_h - 50))
                a_pts.append((px, ay))

                gy = graph_y + graph_h - 20 - int(min(120.0, gyro_buffer[i]) / 120.0 * (graph_h - 50))
                g_pts.append((px, gy))

            cv2.polylines(canvas, [np.array(p_pts)], False, (22, 115, 249), 2) # Orange
            cv2.polylines(canvas, [np.array(a_pts)], False, (246, 130, 59), 2) # Blue
            cv2.polylines(canvas, [np.array(g_pts)], False, (247, 85, 168), 2) # Purple

        # Footer Instruction Bar
        cv2.rectangle(canvas, (0, win_h - 35), (win_w, win_h), (30, 41, 59), -1)
        if not recording:
            cv2.putText(canvas, "Press 'R' to Start 15s Recording | Press 'Q' or ESC to Exit", (20, win_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (251, 191, 36), 1)
        else:
            cv2.putText(canvas, "RECORDING IN PROGRESS... Keep knee movement fluid. Window auto-closes when done.", (20, win_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 2)

        cv2.imshow("ArthoSense Wearable Sensor Recorder", canvas)
        
        key = cv2.waitKey(80) & 0xFF
        if key == ord('r') or key == ord('R'):
            if not recording:
                recording = True
                start_time = time.time()
                recorded_rows = []
                print("🔴 Recording Started! Collecting 15s sensor stream...")
        elif key == ord('q') or key == ord('Q') or key == 27: # ESC
            print("User exited sensor recorder.")
            break

    if stream_mgr:
        stream_mgr.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_sensor_recorder()
