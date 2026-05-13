import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import serial
import serial.tools.list_ports
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import re

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Wi-Fi Vision Dashboard", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for a professional, sleek tech-dashboard look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .metric-container {
        background-color: #1E2127;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid #333;
    }
    .metric-title { font-size: 14px; color: #A0AEC0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #00F2FF; }
    </style>
""", unsafe_allow_html=True)

BONES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(8,10),(10,11),(11,12),(8,13),(13,14),(14,15)]

# --- 2. MODEL ARCHITECTURE ---
class TEDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 30, 128) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.regressor = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 51))

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1) + self.pos_embed
        x = self.transformer(x).mean(dim=1)
        return self.regressor(x)

# --- 3. ADVANCED CLASSIFIER ---
def classify_state(kps_3d, activity=0.0, gate=0.0, walking_ratio=0.0):
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
    
    ankle_dist = np.linalg.norm(kps_3d[3] - kps_3d[6])

    if spread > (height * 1.1): 
        return "LYING DOWN", "#FF3333"
    elif l_wrist_y < neck_y or r_wrist_y < neck_y:
        return "HANDS RAISED", "#FF00FF"
    else:
        torso_y_length = pelvis_y - neck_y
        thigh_y_length = avg_knee_y - pelvis_y
        
        if thigh_y_length < (torso_y_length * 0.6):
            return "SITTING", "#FFA500"
        elif ankle_dist > (height * 0.3) or walking_ratio > 6.0:
            return "WALKING", "#00F2FF"
        else:
            return "STANDING", "#00FF00"

# --- 4. HARDWARE & PLOTTING HELPERS ---
def get_active_ports():
    ports = serial.tools.list_ports.comports()
    esp_ports = [p.device for p in ports if any(x in p.description for x in ["USB", "CH340", "CP210x", "Serial"])]
    if len(esp_ports) < 2:
        esp_ports = [p.device for p in ports]
    return esp_ports

def render_plot(kps_3d=None, message=None, msg_color='white'):
    """Always renders an 8x8 plot to prevent UI jumping. Displays text if no skeleton."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor('#0E1117')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0E1117')
    ax.set_axis_off()
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

    if message:
        # Draw floating text in the void instead of resizing the UI
        ax.text2D(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center', 
                  color=msg_color, fontsize=18, weight='bold', linespacing=1.5)
    elif kps_3d is not None:
        # Draw the skeleton
        ax.view_init(elev=20, azim=45)
        ax.scatter(kps_3d[:, 0], kps_3d[:, 2], -kps_3d[:, 1], c='#00f2ff', s=50, edgecolors='w', linewidth=0.5)
        for b in BONES:
            p1, p2 = kps_3d[b[0]], kps_3d[b[1]]
            ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], c='#1f77b4', linewidth=3, alpha=0.8)
            
    return fig

def update_metric_html(title, value, color="#00F2FF"):
    """Helper to draw custom metric cards"""
    return f"""
    <div class="metric-container">
        <div class="metric-title">{title}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """

# --- 5. UI LAYOUT ---
st.title("🛰️ Wi-Fi Vision: 3D Digital Twin")
st.markdown("<p style='color: #A0AEC0; font-size: 16px;'>Vision-Free Real-Time Human Posture Inference Dashboard</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔌 Hardware Interface")
all_available = get_active_ports()
if len(all_available) < 2:
    st.sidebar.error(f"Only {len(all_available)} USB device(s) found! Plug in both ESP32s.")

com1 = st.sidebar.selectbox("RX1 Port", all_available, index=0 if len(all_available) > 0 else None)
com2 = st.sidebar.selectbox("RX2 Port", all_available, index=1 if len(all_available) > 1 else None)

if "running" not in st.session_state:
    st.session_state.running = False

col_start, col_stop = st.sidebar.columns(2)
if col_start.button("🚀 Start", type="primary", use_container_width=True):
    st.session_state.running = True
if col_stop.button("🛑 Stop", use_container_width=True):
    st.session_state.running = False

st.sidebar.divider()
st.sidebar.info("Model: TEDNet Transformer (51-Dim)\n\nWindow Size: 30 Frames")

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
act_spot = m1.empty()
gate_spot = m2.empty()
buf_spot = m3.empty()
status_spot = m4.empty()

st.divider()
plot_spot = st.empty()

# Initialize empty state
act_spot.markdown(update_metric_html("Activity Level", "0.00"), unsafe_allow_html=True)
gate_spot.markdown(update_metric_html("Kinetic Gate", "Standby"), unsafe_allow_html=True)
buf_spot.markdown(update_metric_html("Data Buffer", "0/30"), unsafe_allow_html=True)
status_spot.markdown(update_metric_html("System Status", "OFFLINE", "#888888"), unsafe_allow_html=True)
plot_spot.pyplot(render_plot(message="SYSTEM OFFLINE\n\nSelect COM Ports and Press Start", msg_color="#888888"))

# --- 6. CORE ENGINE ---
if st.session_state.running:
    if com1 == com2 or com1 is None or com2 is None:
        st.error("Error: You must select two distinct COM ports.")
        st.session_state.running = False
        st.rerun()

    model = TEDNet()
    try:
        model.load_state_dict(torch.load("models/tednet_3d_best.pth", map_location="cpu", weights_only=True))
        model.eval()
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        st.stop()

    try:
        ser1 = serial.Serial(com1, 115200, timeout=0.01)
        ser2 = serial.Serial(com2, 115200, timeout=0.01)
    except Exception as e:
        st.error(f"Serial Error: {e}\n\nClose the Arduino IDE and replug the cables.")
        st.stop()

    buf1, buf2 = deque(maxlen=30), deque(maxlen=30)
    is_calibrated = False
    calibration_mads = []
    CALIBRATION_FRAMES = 200
    dynamic_threshold = 0.0

    while st.session_state.running:
        for ser, buf in [(ser1, buf1), (ser2, buf2)]:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors="ignore")
                    match = re.search(r'\[(.*?)\]', line)
                    if match:
                        nums = re.findall(r'-?\d+', match.group(1))
                        if len(nums) >= 128:
                            buf.append(np.array([float(n) for n in nums[:128]], dtype=np.float32))
            except serial.SerialException:
                st.error("USB Interrupt! Please stop and restart.")
                st.stop()

        if len(buf1) >= 30 and len(buf2) >= 30:
            b1, b2 = np.array(list(buf1)), np.array(list(buf2))
            x_raw = np.concatenate([b1, b2], axis=1)
            
            activity = np.mean(np.abs(np.diff(x_raw, axis=0)))
            act_spot.markdown(update_metric_html("Activity Level", f"{activity:.2f}"), unsafe_allow_html=True)

            # --- STAGE 1: CALIBRATION ---
            if not is_calibrated:
                calibration_mads.append(activity)
                
                gate_spot.markdown(update_metric_html("Kinetic Gate", "Calibrating...", "#FFA500"), unsafe_allow_html=True)
                buf_spot.markdown(update_metric_html("Calibration", f"{len(calibration_mads)}/{CALIBRATION_FRAMES}"), unsafe_allow_html=True)
                status_spot.markdown(update_metric_html("System Status", "CALIBRATING", "#FFA500"), unsafe_allow_html=True)

                fig = render_plot(message=f"⚙️ AUTO-CALIBRATING GATE\n\nPlease stand still.\n\nProgress: {len(calibration_mads)} / {CALIBRATION_FRAMES}", msg_color="#FFA500")
                plot_spot.pyplot(fig)
                plt.close(fig) # Critical to prevent memory leaks

                if len(calibration_mads) >= CALIBRATION_FRAMES:
                    is_calibrated = True
                    dynamic_threshold = np.mean(calibration_mads) + (0.3 * np.std(calibration_mads))
                
                time.sleep(0.01)
                continue 

            # --- STAGE 2: INFERENCE ---
            gate_spot.markdown(update_metric_html("Kinetic Gate", f"{dynamic_threshold:.2f}", "#00FF00"), unsafe_allow_html=True)
            buf_spot.markdown(update_metric_html("Data Buffer", "READY", "#00FF00"), unsafe_allow_html=True)

            if activity < dynamic_threshold:
                status_spot.markdown(update_metric_html("System Status", "EMPTY ROOM", "#FF3333"), unsafe_allow_html=True)
                fig = render_plot(message=f"EMPTY ROOM\n\nWaiting for Wi-Fi Disturbance...\n\n(Activity: {activity:.1f} / Gate: {dynamic_threshold:.1f})", msg_color="#FF3333")
                plot_spot.pyplot(fig)
                plt.close(fig)
            else:
                std_val = max(x_raw.std(), 10.0)
                x_np = (x_raw - x_raw.mean()) / std_val
                x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    pred = model(x_tensor).numpy()[0]
                kps_3d = pred.reshape(17, 3)

                ptp_mean = np.mean(np.ptp(x_raw, axis=0))
                w_ratio = ptp_mean / (activity + 1e-6)

                pose_status, pose_color = classify_state(kps_3d, activity, dynamic_threshold, w_ratio)
                status_spot.markdown(update_metric_html("System Status", pose_status, pose_color), unsafe_allow_html=True)

                fig = render_plot(kps_3d=kps_3d)
                plot_spot.pyplot(fig)
                plt.close(fig)
                
        else:
            buf_spot.markdown(update_metric_html("Data Buffer", f"{len(buf1)}/30", "#FFA500"), unsafe_allow_html=True)
        
        time.sleep(0.01)

    # If the loop breaks gracefully via the stop button
    ser1.close()
    ser2.close()