"""
ArthoSense NER - High-Speed Zero-Latency Piezo Acoustic Telemetry Recorder
Dedicated standalone Piezo disc plotter (No MPU6050).
Features real-time zero-lag tapping response and rock-solid stable OpenCV loop execution.
"""

import cv2
import numpy as np
import time
import csv
import os
import sys
import math
from sensor_stream import SensorStreamManager, list_available_com_ports, list_physical_com_ports

def run_sensor_recorder():
    joint_condition = sys.argv[1] if len(sys.argv) > 1 else "moderate"
    hardware_mode = sys.argv[2] if len(sys.argv) > 2 else "simulator"
    requested_port = sys.argv[3] if len(sys.argv) > 3 else "AUTO"

    stream_mgr = None
    data_source_label = ""
    is_physical = False
    hw_disconnected = False

    if hardware_mode == "physical":
        real_ports = list_physical_com_ports()
        target_port = requested_port if requested_port in real_ports else (real_ports[0] if real_ports else None)
        
        if target_port:
            stream_mgr = SensorStreamManager(mode="physical", port=target_port)
            if stream_mgr.connect_serial():
                is_physical = True
                data_source_label = f"Physical Piezo Hardware ({target_port} @ 115200)"
            else:
                hw_disconnected = True
                data_source_label = "PHYSICAL HARDWARE DISCONNECTED (Port Unreachable)"
        else:
            hw_disconnected = True
            data_source_label = "PHYSICAL HARDWARE DISCONNECTED (No USB Serial Band Found)"
    else:
        stream_mgr = SensorStreamManager(mode="simulator")
        data_source_label = "Calibrated Piezo Field Simulator"

    # Canvas setup
    win_w, win_h = 780, 500
    bg_color = (15, 23, 42) # Slate dark #0f172a
    
    csv_file = "recorded_sensor_data.csv"
    
    recording = False
    start_time = None
    recorded_rows = []
    
    piezo_buffer = []
    rms_buffer = []
    max_buf = 140

    frame_counter = 0

    print("==================================================")
    print(" ARTHOSENSE ZERO-LATENCY PIEZO RECORDER LAUNCHED")
    print("==================================================")
    print(f"Hardware Mode     : {hardware_mode.upper()}")
    print(f"Data Source       : {data_source_label}")
    print("Performance       : High-Speed 60 FPS Real-Time Plotting (Zero Buffer Lag)")
    print("==================================================")

    while True:
        frame_counter += 1
        canvas = np.full((win_h, win_w, 3), bg_color, dtype=np.uint8)
        
        # Periodic USB Port Check (Every 20 frames = ~1s interval, to prevent CPU lag)
        if hardware_mode == "physical" and frame_counter % 20 == 0:
            current_ports = list_physical_com_ports()
            if not current_ports:
                if not hw_disconnected:
                    print("🔴 USB HARDWARE UNPLUGGED: Switching to Disconnected Flatline state instantly!")
                hw_disconnected = True
                is_physical = False
                data_source_label = "PHYSICAL HARDWARE DISCONNECTED (USB Cable Unplugged)"
                if stream_mgr:
                    stream_mgr.close()
                    stream_mgr = None
            else:
                active_port = current_ports[0]
                if hw_disconnected or not stream_mgr:
                    print(f"🟢 USB HARDWARE DETECTED: Reconnecting to {active_port}...")
                    stream_mgr = SensorStreamManager(mode="physical", port=active_port)
                    if stream_mgr.connect_serial():
                        hw_disconnected = False
                        is_physical = True
                        data_source_label = f"Physical Piezo Hardware ({active_port} @ 115200)"
                    else:
                        hw_disconnected = True

        if hw_disconnected or not stream_mgr:
            rms = 0.0
            piezo_raw = 2.350
        else:
            try:
                sample = stream_mgr.read_sample(joint_condition=joint_condition)
                if sample is None:
                    rms = 0.0
                    piezo_raw = 2.350
                else:
                    rms = sample["vibration_rms"]
                    piezo_raw = sample["piezo_raw"]
            except Exception:
                rms = 0.0
                piezo_raw = 2.350
        
        piezo_buffer.append(piezo_raw)
        rms_buffer.append(rms)
        
        if len(piezo_buffer) > max_buf:
            piezo_buffer.pop(0)
            rms_buffer.pop(0)
            
        current_time = time.time()
        
        if recording and not hw_disconnected:
            elapsed = current_time - start_time
            remaining = max(0.0, 15.0 - elapsed)
            
            recorded_rows.append({
                "Timestamp": current_time,
                "Vibration_RMS": rms,
                "Acoustic_Signal": round(piezo_raw, 3)
            })
            
            if elapsed >= 15.0:
                try:
                    with open(csv_file, mode="w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Vibration_RMS", "Acoustic_Signal"])
                        writer.writeheader()
                        writer.writerows(recorded_rows)
                    print(f"✅ Recording Complete! Saved {len(recorded_rows)} Piezo frames to {csv_file}")
                except Exception as err:
                    print(f"Error writing sensor CSV: {err}")
                break

        # Render Header Bar
        cv2.rectangle(canvas, (0, 0), (win_w, 60), (30, 41, 59), -1)
        cv2.putText(canvas, "ArthoSense NER - Piezo Acoustic Stethoscope Telemetry", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Source Sub-label
        src_color = (34, 197, 94) if is_physical else ((239, 68, 68) if hw_disconnected else (234, 179, 8))
        cv2.putText(canvas, f"Source: {data_source_label}", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, src_color, 1)

        # Render Status Badge
        if hw_disconnected:
            cv2.rectangle(canvas, (win_w - 320, 15), (win_w - 20, 45), (239, 68, 68), -1) # Red
            cv2.putText(canvas, "HARDWARE DISCONNECTED (NO SIGNAL)", (win_w - 312, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        elif not recording:
            cv2.rectangle(canvas, (win_w - 280, 15), (win_w - 20, 45), (16, 185, 129), -1) # Green
            cv2.putText(canvas, "STANDBY: PRESS 'R' TO RECORD", (win_w - 272, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            elapsed = current_time - start_time
            remaining = max(0.0, 15.0 - elapsed)
            cv2.rectangle(canvas, (win_w - 280, 15), (win_w - 20, 45), (239, 68, 68), -1) # Red
            cv2.putText(canvas, f"RECORDING: {remaining:.1f}s REMAINING", (win_w - 272, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

        # Render Telemetry Cards (2 columns)
        card_w = 345
        # Card 1: Piezo Vibration RMS
        cv2.rectangle(canvas, (20, 75), (20 + card_w, 145), (30, 41, 59), -1)
        cv2.putText(canvas, "Acoustic Vibration RMS", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
        val_str1 = f"{rms:.3f} g" if not hw_disconnected else "0.000 g (NO SIGNAL)"
        cv2.putText(canvas, val_str1, (35, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (239, 68, 68) if hw_disconnected or rms >= 0.40 else (34, 197, 94), 2)
        
        # Card 2: Instantaneous Acoustic Signal (V)
        cv2.rectangle(canvas, (415, 75), (415 + card_w, 145), (30, 41, 59), -1)
        cv2.putText(canvas, "Live Piezo Signal (A0 Voltage)", (430, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
        val_str2 = f"{piezo_raw:.3f} V" if not hw_disconnected else "0.000 V (NO SIGNAL)"
        cv2.putText(canvas, val_str2, (430, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (239, 68, 68) if hw_disconnected else (56, 189, 248), 2)

        # Render Live Telemetry Graph Box
        graph_x, graph_y, graph_w, graph_h = 20, 165, 740, 280
        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (30, 41, 59), -1)
        cv2.putText(canvas, "Piezo Disc Live Acoustic Waveform (A0 Volts)", (graph_x + 15, graph_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not hw_disconnected:
            cv2.rectangle(canvas, (graph_x + graph_w - 180, graph_y + 10), (graph_x + graph_w - 15, graph_y + 32), (56, 189, 248), -1)
            cv2.putText(canvas, f"Signal: {piezo_raw:.3f} V", (graph_x + graph_w - 170, graph_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (15, 23, 42), 2)

        if hw_disconnected:
            # FLAT LINE + WARNING OVERLAY
            zero_y = graph_y + graph_h - 30
            cv2.line(canvas, (graph_x + 15, zero_y), (graph_x + graph_w - 15, zero_y), (100, 116, 139), 2)
            
            cv2.rectangle(canvas, (graph_x + 90, graph_y + 90), (graph_x + graph_w - 90, graph_y + 180), (127, 29, 29), -1)
            cv2.rectangle(canvas, (graph_x + 90, graph_y + 90), (graph_x + graph_w - 90, graph_y + 180), (239, 68, 68), 2)
            cv2.putText(canvas, "NO USB HARDWARE DETECTED!", (graph_x + 170, graph_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(canvas, "Connect Joint Band or switch to SIMULATED mode", (graph_x + 140, graph_y + 158), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (254, 202, 202), 1)
        else:
            # Plot high-speed dynamic waveform line
            if len(piezo_buffer) > 1:
                p_pts = []
                buf_min = min(piezo_buffer)
                buf_max = max(piezo_buffer)
                
                # Dynamic scope range centered around signal swing
                center_v = (buf_min + buf_max) / 2.0
                span = max(0.50, (buf_max - buf_min) * 1.4)
                
                plot_min = max(0.0, center_v - span / 2.0)
                plot_max = min(5.0, center_v + span / 2.0)
                range_v = max(0.20, plot_max - plot_min)

                for i in range(len(piezo_buffer)):
                    px = graph_x + 20 + int((i / max_buf) * (graph_w - 40))
                    norm_y = (piezo_buffer[i] - plot_min) / range_v
                    py = graph_y + graph_h - 25 - int(norm_y * (graph_h - 60))
                    py = max(graph_y + 35, min(graph_y + graph_h - 15, py))
                    p_pts.append((px, py))

                cv2.polylines(canvas, [np.array(p_pts)], False, (248, 189, 56), 2, cv2.LINE_AA)

        # Footer Instruction Bar
        cv2.rectangle(canvas, (0, win_h - 35), (win_w, win_h), (30, 41, 59), -1)
        if hw_disconnected:
            cv2.putText(canvas, "⚠️ HARDWARE DISCONNECTED: Plug in USB Joint Band or press 'Q' / ESC to exit", (20, win_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 2)
        elif not recording:
            cv2.putText(canvas, "Press 'R' to Start 15s Recording | Press 'Q' or ESC to Exit", (20, win_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (251, 191, 36), 1)
        else:
            cv2.putText(canvas, "RECORDING IN PROGRESS... Keep knee movement fluid. Window auto-closes when done.", (20, win_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 2)

        cv2.imshow("ArthoSense Wearable Sensor Recorder", canvas)
        
        key = cv2.waitKey(15) & 0xFF  # Fast 60 FPS loop
        if key == ord('r') or key == ord('R'):
            if hw_disconnected:
                print("⚠️ Cannot record: Physical Wearable Sensor Hardware is disconnected!")
            elif not recording:
                recording = True
                start_time = time.time()
                recorded_rows = []
                print("🔴 Recording Started! Collecting 15s Piezo acoustic stream...")
        elif key == ord('q') or key == ord('Q') or key == 27: # ESC
            print("User exited sensor recorder.")
            break

    if stream_mgr:
        stream_mgr.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_sensor_recorder()
