"""
STEP 5 — Camera-Free 3D Digital Twin Inference (LIVE)
=====================================================
Reads live ESP32 COM ports, Auto-Calibrates the Kinetic Gate 
to the room's environment, and plots a 3D skeleton in real-time.
"""

import serial
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import time
import threading

# --- CONFIGURATION ---
COM1 = "COM3"
COM2 = "COM4"
BAUD_RATE = 115200
WINDOW_SIZE = 100
N_FEATURES = 256
OUTPUT_DIM = 51
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BONES = [
    (0, 1), (1, 2), (2, 3),       
    (0, 4), (4, 5), (5, 6),       
    (0, 7), (7, 8), (8, 9),       
    (8, 10), (10, 11), (11, 12),  
    (8, 13), (13, 14), (14, 15)   
]

class TEDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(N_FEATURES, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, WINDOW_SIZE, 128) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.regressor = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, OUTPUT_DIM))

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1) + self.pos_embed
        x = self.transformer(x).mean(dim=1)
        return self.regressor(x)

def parse_csi(line):
    match = re.search(r'\[(.*?)\]', line)
    if match:
        numbers = re.findall(r'-?\d+\.?\d*', match.group(1))
        if len(numbers) >= 128:
            return np.array([float(n) for n in numbers[:128]], dtype=np.float32)
    return None

def classify_state(kps_3d):
    head_y = kps_3d[9, 1]
    ankle_y = (kps_3d[3, 1] + kps_3d[6, 1]) / 2.0
    skeleton_height = ankle_y - head_y
    if skeleton_height < 0.85:
        return "Person Detected: SITTING", "orange"
    else:
        return "Person Detected: STANDING", "green"

def serial_reader(ser, buf):
    """Background thread to continuously read serial data and keep buffers fresh."""
    while True:
        try:
            # Prevent Windows OS Buffer overflow which causes data corruption
            if ser.in_waiting > 4096:
                print(f"⚠️ Cleared backed up buffer on {ser.port}")
                ser.reset_input_buffer()
                
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors="ignore").strip()
                if line:
                    c = parse_csi(line)
                    if c is not None:
                        buf.append(c)
            else:
                time.sleep(0.005)
        except (serial.SerialException, OSError):
            time.sleep(1)
        except Exception:
            time.sleep(0.05)

def run_live():
    print("🧠 Loading Trained TEDNet Model...")
    model = TEDNet().to(DEVICE)
    # Added weights_only=True to remove the PyTorch security warning!
    model.load_state_dict(torch.load("models/tednet_3d_best.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    try:
        ser1 = serial.Serial(COM1, BAUD_RATE, timeout=0.1)
        print(f"[RX1] Connected: {COM1}")
        ser2 = serial.Serial(COM2, BAUD_RATE, timeout=0.1)
        print(f"[RX2] Connected: {COM2}")
    except serial.SerialException as e:
        print(f"⚠️ Serial Error: Could not open COM ports. \n{e}")
        return

    buf1, buf2 = deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE)
    
    # Start background threads to keep serial buffers instantly fresh
    t1 = threading.Thread(target=serial_reader, args=(ser1, buf1), daemon=True)
    t2 = threading.Thread(target=serial_reader, args=(ser2, buf2), daemon=True)
    t1.start()
    t2.start()

    print("Waiting for CSI...")
    print("📡 Listening... (Press Ctrl+C to stop)")

    # --- AUTO-CALIBRATION VARIABLES ---
    is_calibrated = False
    calibration_mads = []
    dynamic_threshold = 0.0
    CALIBRATION_FRAMES = 50 

    try:
        while True:
            # Snapshot the latest window size of data
            b1 = list(buf1)
            b2 = list(buf2)
            
            if len(b1) < WINDOW_SIZE or len(b2) < WINDOW_SIZE: 
                time.sleep(0.01)
                continue

            x_raw = np.concatenate([np.array(b1), np.array(b2)], axis=1)
            activity_level = np.mean(np.abs(np.diff(x_raw, axis=0)))
            
            # ==========================================
            # STAGE 1: THE AUTO-CALIBRATION SEQUENCE
            # ==========================================
            if not is_calibrated:
                calibration_mads.append(activity_level)
                
                ax.cla()
                ax.set_title(f"⚙️ AUTO-CALIBRATING KINETIC GATE ⚙️\nPlease stand still or leave the room.\nProgress: {len(calibration_mads)}/{CALIBRATION_FRAMES}", 
                             fontsize=16, fontweight='bold', color='orange')
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
                ax.set_axis_off() # Hide grid during calibration
                plt.pause(0.03)
                
                if len(calibration_mads) >= CALIBRATION_FRAMES:
                    is_calibrated = True
                    # IEEE 3-Sigma Math to set the threshold
                    dynamic_threshold = np.mean(calibration_mads) + (3 * np.std(calibration_mads))
                    print(f"\n✅ Calibration Complete! Room Baseline: {np.mean(calibration_mads):.1f}")
                    print(f"✅ Kinetic Gate Threshold set to: {dynamic_threshold:.1f}\n")
                    print("Buffers full. Starting inference...")
                continue

            # ==========================================
            # STAGE 2: LIVE INFERENCE
            # ==========================================
            ax.cla()
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_xlabel('X'); ax.set_ylabel('Depth (Z)'); ax.set_zlabel('Y (Height)')
            ax.set_axis_on()
            
            if activity_level < dynamic_threshold:
                # Gate Closed (Empty Room)
                status_text = f"Empty Room (Activity: {activity_level:.1f} / Gate: {dynamic_threshold:.1f})"
                status_color = "red"
            else:
                # Gate Open (Person Detected - Run AI)
                x_np = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-8)
                x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred = model(x_tensor).cpu().numpy()[0] 
                
                kps_3d = pred.reshape(17, 3)
                status_text, status_color = classify_state(kps_3d)
                status_text += f" (Activity: {activity_level:.1f})"

                # Draw Skeleton
                ax.scatter(kps_3d[:, 0], kps_3d[:, 2], -kps_3d[:, 1], c='cyan', s=50, edgecolors='k')
                for bone in BONES:
                    pt1, pt2 = kps_3d[bone[0]], kps_3d[bone[1]]
                    ax.plot([pt1[0], pt2[0]], [pt1[2], pt2[2]], [-pt1[1], -pt2[1]], c='blue', linewidth=2.5)

            ax.set_title(f"LIVE 3D Digital Twin (Vision-Free)\nStatus: {status_text}", 
                         fontsize=14, fontweight='bold', color=status_color)
            plt.pause(0.03)

    except KeyboardInterrupt:
        print("\nDemo stopped by user. Safely closing serial ports...")
    
    finally:
        ser1.close()
        ser2.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    run_live()