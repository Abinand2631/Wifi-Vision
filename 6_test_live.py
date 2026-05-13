"""
STEP 5 — Camera-Free 3D Digital Twin Inference (LIVE)
=====================================================
Reads live ESP32 COM ports, Auto-Calibrates the Kinetic Gate 
to the room's environment, and plots a 3D skeleton in real-time
with advanced biomechanical posture classification.
"""

import serial
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import sys
import threading
import time

# --- CONFIGURATION ---
# Check if the GUI sent us COM ports, otherwise use defaults
if len(sys.argv) >= 3:
    COM1 = sys.argv[1]
    COM2 = sys.argv[2]
else:
    COM1 = "COM4"
    COM2 = "COM9"

BAUD_RATE = 115200
# ... rest of your configuration stays exactly the same
WINDOW_SIZE = 30
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

# --- NEURAL NETWORK ARCHITECTURE ---
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

# --- UTILITY FUNCTIONS ---
def parse_csi(line):
    match = re.search(r'\[(.*?)\]', line)
    if match:
        numbers = re.findall(r'-?\d+\.?\d*', match.group(1))
        if len(numbers) >= 128:
            return np.array([float(n) for n in numbers[:128]], dtype=np.float32)
    return None

def classify_state(kps_3d, activity=0.0, gate=0.0):
    """
    Advanced Biomechanical Posture Classifier.
    Uses relative joint proportions and aspect ratios.
    """
    min_y, max_y = np.min(kps_3d[:, 1]), np.max(kps_3d[:, 1])
    min_x, max_x = np.min(kps_3d[:, 0]), np.max(kps_3d[:, 0])
    min_z, max_z = np.min(kps_3d[:, 2]), np.max(kps_3d[:, 2])

    height = max_y - min_y
    spread = max((max_x - min_x), (max_z - min_z))

    neck_y = kps_3d[8, 1]
    pelvis_y = kps_3d[0, 1]
    avg_knee_y = (kps_3d[2, 1] + kps_3d[5, 1]) / 2.0
    l_wrist_y = kps_3d[12, 1]
    r_wrist_y = kps_3d[15, 1]
    
    # Calculate distance between left and right ankles (Indices 3 and 6)
    ankle_dist = np.linalg.norm(kps_3d[3] - kps_3d[6])

    if spread > (height * 1.1): 
        return "Person Detected: LYING DOWN", "red"
    elif l_wrist_y < neck_y or r_wrist_y < neck_y:
        return "Person Detected: HANDS RAISED", "magenta"
    else:
        torso_y_length = pelvis_y - neck_y
        thigh_y_length = avg_knee_y - pelvis_y
        
        if thigh_y_length < (torso_y_length * 0.6):
            return "Person Detected: SITTING", "orange"
        elif ankle_dist > (height * 0.25) or activity > (gate * 1.5):
            return "Person Detected: WALKING", "cyan"
        else:
            return "Person Detected: STANDING", "green"

def serial_reader(port_name, baud_rate, buf):
    """Background thread to continuously read serial data, parse carefully, and auto-reconnect."""
    while True:
        ser = None
        try:
            ser = serial.Serial(port_name, baud_rate, timeout=0.1)
            print(f"🔌 Live Connection Established: {port_name}")
            while True:
                try:
                    # Prevent OS buffer overrun lag
                    if ser.in_waiting > 4096:
                        ser.reset_input_buffer()
                        
                    if ser.in_waiting:
                        line = ser.readline().decode('utf-8', errors="ignore").strip()
                        if line:
                            match = re.search(r'\[(.*?)\]', line)
                            if match:
                                numbers = re.findall(r'-?\d+\.?\d*', match.group(1))
                                if len(numbers) >= 128:
                                    c = np.array([float(n) for n in numbers[:128]], dtype=np.float32)
                                    buf.append(c)
                    else:
                        time.sleep(0.005)
                # If USB unplugs or ESP resets, break inner loop to resurrect
                except (serial.SerialException, OSError):
                    print(f"⚠️ USB Interrupt on {port_name}. Auto-reconnecting...")
                    break 
                except Exception:
                    time.sleep(0.05)
        except (serial.SerialException, OSError):
            time.sleep(1.0)
        finally:
            if ser is not None and ser.is_open:
                ser.close()

# --- MAIN INFERENCE PIPELINE ---
def run_live():
    print("🧠 Loading Trained TEDNet Model...")
    model = TEDNet().to(DEVICE)
    # The weights_only=True fix is here!
    model.load_state_dict(torch.load("models/tednet_3d_best.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    buf1, buf2 = deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE)
    
    # Start background threads using auto-reconnect logic so resets don't kill the code
    t1 = threading.Thread(target=serial_reader, args=(COM1, BAUD_RATE, buf1), daemon=True)
    t2 = threading.Thread(target=serial_reader, args=(COM2, BAUD_RATE, buf2), daemon=True)
    t1.start()
    t2.start()

    print("Waiting for CSI...")
    print("📡 Listening... (Press Ctrl+C to stop)")

    is_calibrated = False
    calibration_mads = []
    dynamic_threshold = 0.0
    # Increased to 200 frames so you have ~2 seconds to see the Orange Calibration Screen
    CALIBRATION_FRAMES = 200 

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
            
            # STAGE 1: AUTO-CALIBRATION
            if not is_calibrated:
                calibration_mads.append(activity_level)
                
                ax.cla()
                ax.set_title(f"⚙️ AUTO-CALIBRATING KINETIC GATE ⚙️\nPlease stand still or leave the room.\nProgress: {len(calibration_mads)}/{CALIBRATION_FRAMES}", 
                             fontsize=16, fontweight='bold', color='orange')
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
                ax.set_axis_off() 
                plt.pause(0.03)
                
                if len(calibration_mads) >= CALIBRATION_FRAMES:
                    is_calibrated = True
                    # Relaxed the gate for flawless sensitivity
                    dynamic_threshold = np.mean(calibration_mads) + (0.3 * np.std(calibration_mads))
                    print(f"\n✅ Calibration Complete! Room Baseline: {np.mean(calibration_mads):.1f}")
                    print(f"✅ Kinetic Gate Threshold set to: {dynamic_threshold:.1f}\n")
                    print("Buffers full. Starting inference...")
                continue

            # STAGE 2: LIVE INFERENCE
            ax.cla()
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_xlabel('X'); ax.set_ylabel('Depth (Z)'); ax.set_zlabel('Y (Height)')
            ax.set_axis_on()
            
            if activity_level < dynamic_threshold:
                status_text = f"Empty Room (Activity: {activity_level:.1f} / Gate: {dynamic_threshold:.1f})"
                status_color = "red"
            else:
                # FIX: Clamp the standard deviation to prevent static RF noise from being 
                # artificially amplified (which mathematically collapses the 3D skeleton into a ball).
                std_val = max(x_raw.std(), 10.0)
                x_np = (x_raw - x_raw.mean()) / std_val
                x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred = model(x_tensor).cpu().numpy()[0] 
                
                kps_3d = pred.reshape(17, 3)
                status_text, status_color = classify_state(kps_3d, activity_level, dynamic_threshold)
                status_text += f" (Activity: {activity_level:.1f})"

                ax.scatter(kps_3d[:, 0], kps_3d[:, 2], -kps_3d[:, 1], c='cyan', s=50, edgecolors='k')
                for bone in BONES:
                    pt1, pt2 = kps_3d[bone[0]], kps_3d[bone[1]]
                    ax.plot([pt1[0], pt2[0]], [pt1[2], pt2[2]], [-pt1[1], -pt2[1]], c='blue', linewidth=2.5)

            ax.set_title(f"LIVE 3D Digital Twin (Vision-Free)\nStatus: {status_text}", 
                         fontsize=14, fontweight='bold', color=status_color)
            plt.pause(0.03)

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    run_live()