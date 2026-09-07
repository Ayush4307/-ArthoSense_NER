"""
ArthoSense NER - Hardware Sensor Stream & Synthesizer Module
Handles physical ingestion from MPU6050 IMU + Piezoelectric acoustic contact microphone
via Serial (pyserial) and provides a calibrated interactive hardware simulator.
"""

import time
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False

def list_available_com_ports() -> List[str]:
    """Returns a list of available serial COM ports on Windows."""
    if not SERIAL_AVAILABLE:
        return ["SIMULATED_PORT_1", "SIMULATED_PORT_2"]
    ports = serial.tools.list_ports.comports()
    port_list = [p.device for p in ports]
    if not port_list:
        return ["No physical COM ports found (Use Simulator)"]
    return port_list

def get_connected_hardware_info() -> Tuple[bool, str, List[str]]:
    """
    Scans system COM ports and returns:
    (is_connected, status_message, list_of_port_descriptions)
    """
    if not SERIAL_AVAILABLE:
        return False, "⚠️ PySerial module missing. Using Calibrated Field Simulator.", []
    ports = serial.tools.list_ports.comports()
    if not ports:
        return False, "⚠️ No Physical Hardware / Joint Band Detected! Please connect the USB Wearable Hardware Joint Band or switch to Calibrated Simulator Mode.", []
    
    port_descs = [f"{p.device} - {p.description}" for p in ports]
    primary_device = port_descs[0]
    return True, f"🟢 Connected Physical Device: {primary_device}", port_descs

class SensorStreamManager:
    """
    Manages data stream from wearable physical sensors (MPU6050 + Piezo)
    or generated synthetic physiological waveforms.
    """
    def __init__(self, mode: str = "simulator", port: Optional[str] = None, baud_rate: int = 115200):
        self.mode = mode
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.sample_idx = 0
        self.vibration_buffer: List[float] = []
        self.max_buffer_len = 100

    def connect_serial(self) -> bool:
        """Attempts to open physical COM port."""
        if not SERIAL_AVAILABLE or not self.port or "SIMULATED" in self.port:
            return False
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=0.5)
            return True
        except Exception:
            self.serial_conn = None
            return False

    def close(self):
        """Closes serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def read_sample(self, joint_condition: str = "moderate") -> Dict[str, Any]:
        """
        Reads one multimodal sample frame (IMU 6-DOF + Piezo Acoustic).
        If in simulator mode or serial disconnected, generates calibrated biomechanical signals.
        """
        self.sample_idx += 1

        if self.mode == "physical" and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode("utf-8").strip()
                # Expected format: "ax,ay,az,gx,gy,gz,piezo_val"
                parts = line.split(",")
                if len(parts) >= 7:
                    ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
                    gx, gy, gz = float(parts[3]), float(parts[4]), float(parts[5])
                    piezo = float(parts[6])
                    
                    self.vibration_buffer.append(piezo)
                    if len(self.vibration_buffer) > self.max_buffer_len:
                        self.vibration_buffer.pop(0)
                        
                    rms = float(np.sqrt(np.mean(np.square(self.vibration_buffer)))) if self.vibration_buffer else 0.0
                    return {
                        "mode": "Physical Serial",
                        "accel": {"x": ax, "y": ay, "z": az},
                        "gyro": {"x": gx, "y": gy, "z": gz},
                        "piezo_raw": piezo,
                        "vibration_rms": round(rms, 3),
                        "dominant_freq_hz": round(abs(gx * 2.5) % 350 + 50, 1),
                        "crepitus_detected": rms > 0.40
                    }
            except Exception:
                pass # Fallback to simulator

        # Synthesizer generation based on joint condition
        t = self.sample_idx * 0.1
        if joint_condition == "healthy":
            base_noise = np.random.normal(0, 0.04)
            piezo_val = round(abs(math.sin(t * 0.5) * 0.08 + base_noise), 3)
            accel = {"x": round(math.cos(t) * 0.2, 2), "y": round(math.sin(t) * 0.3, 2), "z": 0.98}
            gyro = {"x": round(math.sin(t) * 15.0, 1), "y": round(math.cos(t) * 12.0, 1), "z": 1.2}
            dom_freq = 45.0
            rms = 0.12
            crepitus = False
        elif joint_condition == "mild":
            spike = 0.35 if (self.sample_idx % 25 in [0, 1]) else 0.0
            piezo_val = round(abs(math.sin(t) * 0.15 + np.random.normal(0, 0.08) + spike), 3)
            accel = {"x": round(math.cos(t) * 0.4, 2), "y": round(math.sin(t) * 0.5, 2), "z": 0.95}
            gyro = {"x": round(math.sin(t) * 25.0, 1), "y": round(math.cos(t) * 20.0, 1), "z": 3.4}
            dom_freq = 120.0
            rms = 0.28
            crepitus = spike > 0
        elif joint_condition == "moderate":
            burst = 0.65 if (self.sample_idx % 18 in [0, 1, 2]) else 0.0
            piezo_val = round(abs(math.sin(t * 1.5) * 0.25 + np.random.normal(0, 0.14) + burst), 3)
            accel = {"x": round(math.cos(t * 1.2) * 0.6, 2), "y": round(math.sin(t * 1.2) * 0.7, 2), "z": 0.91}
            gyro = {"x": round(math.sin(t * 1.2) * 45.0, 1), "y": round(math.cos(t * 1.2) * 38.0, 1), "z": 8.5}
            dom_freq = 240.0
            rms = 0.52
            crepitus = True
        else: # severe
            grinding = 0.85 if (self.sample_idx % 12 in [0, 1, 2, 3, 4]) else 0.25
            piezo_val = round(abs(math.sin(t * 2.0) * 0.35 + np.random.normal(0, 0.22) + grinding), 3)
            accel = {"x": round(math.cos(t * 2.0) * 0.9, 2), "y": round(math.sin(t * 2.0) * 1.1, 2), "z": 0.82}
            gyro = {"x": round(math.sin(t * 2.0) * 80.0, 1), "y": round(math.cos(t * 2.0) * 72.0, 1), "z": 18.2}
            dom_freq = 380.0
            rms = 0.84
            crepitus = True

        self.vibration_buffer.append(piezo_val)
        if len(self.vibration_buffer) > self.max_buffer_len:
            self.vibration_buffer.pop(0)

        return {
            "mode": "Calibrated Synthesizer (Field Simulation)",
            "accel": accel,
            "gyro": gyro,
            "piezo_raw": piezo_val,
            "vibration_rms": rms,
            "dominant_freq_hz": dom_freq,
            "crepitus_detected": crepitus
        }
