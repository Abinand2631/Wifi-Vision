# Wi-Fi Vision 📡🧍

**Device-free, camera-free human pose estimation and activity recognition using Wi-Fi CSI and deep learning.**

> Built on 3× ESP32 microcontrollers + a custom TEDNet Transformer architecture. No cameras. No wearables. Just Wi-Fi signals.

### 🎬 Project Demo
https://github.com/Abinand2631/Wifi-Vision/raw/main/Demo/Video.mp4

---

## What It Does

Wi-Fi Vision detects and classifies human posture in real time by analyzing how a person's body disturbs Wi-Fi radio signals — a technique called **Channel State Information (CSI) sensing**.

| Activity | Detection Method |
|---|---|
| 🧍 Standing | CSI amplitude pattern + keypoint thresholding |
| 🪑 Sitting | Kinetic gating on vertical displacement |
| 🚶 Walking | Temporal variance in CSI subcarrier phase |
| 🛌 Lying Down | Low-amplitude, low-variance CSI signature |

No camera feed. No body-worn sensors. Completely **privacy-preserving.**

---

## System Architecture

```
[ESP32 TX] ──── Wi-Fi CSI ────► [ESP32 RX #1]  ──► Serial (115200 baud)
                                [ESP32 RX #2]  ──► Serial (115200 baud)
                                                         │
                                                   1_capture.py
                                                         │
                                                  3_extraction.py
                                                   (preprocessing)
                                                         │
                                                      train.py
                                                  (TEDNet Transformer)
                                                         │
                                                   dashboard.py
                                               (Streamlit live inference)
```

**TEDNet Model:**
- **1D CNN** extracts spatial features from raw CSI subcarrier amplitudes
- **Transformer Encoder** captures temporal dependencies over a 30-frame sliding window
- **Classification head** maps pose embeddings to activity labels with kinetic gating

---

## Hardware Requirements

| Component | Qty | Notes |
|---|---|---|
| ESP32 (any variant) | 3× | 1× TX, 2× RX. External antenna modules improve accuracy. |
| NVIDIA GPU | 1× | Required for training `train.py`. Inference runs on CPU. |
| Webcam | 1× | Used only during data collection for ground-truth labelling. |
| USB Cables | 3× | For flashing and serial data streaming. |

> **Tip:** The project was developed and tested with ESP32's built-in PCB antennas. Using external antenna modules will significantly improve CSI signal quality and classification accuracy.

---

## Software Dependencies

```bash
pip install torch torchvision numpy pyserial matplotlib tqdm streamlit
```

> For GPU-accelerated training, install the CUDA-enabled PyTorch build matching your NVIDIA driver version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## ESP32 Firmware Setup

The ESP32s must be flashed with CSI extraction firmware via ESP-IDF.

**1. Download the firmware**

Download and extract the [ESP CSI Master firmware](https://drive.google.com/file/d/15BBSO7Kxio0WqTDHiqB7bMMBoByQiWyD/view?usp=drive_link).

Target projects inside `esp-csi-master/examples/get-started/`:
- `csi_send` → flash to your **1 TX board**
- `csi_recv` → flash to your **2 RX boards**

**2. Flash each board**

```bash
# Navigate to csi_recv or csi_send directory
idf.py set-target esp32
idf.py menuconfig
# → Serial flasher config → Default baud rate: 115200
idf.py -p COM_PORT flash   # Replace COM_PORT with e.g. COM3 or /dev/ttyUSB0
```

Repeat for all 3 boards.

---

## Running the Project

### Step 1 — Collect Data
```bash
python 1_capture.py
```
Records synchronized CSI streams from both RX boards alongside webcam ground truth for pose labelling.

### Step 2 — Preprocess
```bash
python 3_extraction.py
```
Extracts and normalizes CSI features, aligns frames, and outputs training-ready tensors to `data/`.

### Step 3 — Train the Model
```bash
python train.py
```
Trains the TEDNet Transformer. Best model weights are saved to `models/`. Requires NVIDIA GPU.

### Step 4 — Live Inference Dashboard
```bash
streamlit run dashboard.py
```
Or double-click `Start_WiFi_Pose_dashboard.bat` on Windows.

Opens a real-time Streamlit dashboard showing live posture classification from your ESP32 array.

> **Note on Standalone Inference:** If you prefer a simpler UI or want to test different rendering methods, you can run the standalone scripts (`6_test_live_v1.py` and `6_test_live_v2.py`). We recommend trying both versions to see which one works best for your specific setup!

---

## Results

<!-- 📊 FILL IN your actual accuracy numbers before publishing -->

| Metric | Value |
|---|---|
| Overall Accuracy | _Add your result_ |
| Standing | _Add your result_ |
| Sitting | _Add your result_ |
| Walking | _Add your result_ |
| Lying Down | _Add your result_ |

> Results measured on self-collected dataset. See `train.py` for evaluation code.

---

## Potential Applications

- 🏥 **Healthcare monitoring** — fall detection and activity tracking without cameras in patient rooms
- 🏠 **Smart home automation** — presence and posture-aware environments
- 🔒 **Security** — perimeter intrusion detection without video surveillance
- 👴 **Elderly care** — passive wellness monitoring

---

## Project Structure

```
Wifi-Vision/
├── 1_capture.py              # CSI + webcam data collection
├── 3_extraction.py           # Feature extraction & preprocessing
├── train.py                  # TEDNet Transformer training
├── 6_test_live_v1.py         # Live inference (standalone)
├── 6_test_live_v2.py         # Live inference (standalone)
├── dashboard.py              # Streamlit real-time dashboard
├── launcher.py               # Application launcher
├── Start_WiFi_Pose_dashboard.bat  # Windows one-click launcher
├── csi_video_capture_external/    # External CSI capture utilities
├── data/                     # Preprocessed training data
└── models/                   # Saved model weights
```

---

## About This Project

Built as a B.Tech Final Year Project (Electronics and Communication Engineering, 2026). All hardware testing, data collection, model training, and dashboard development were done independently.

If you build on this work, a mention or citation is appreciated.

**Contributions welcome** — open a PR for improvements, optimizations, or new activity classes.

---

*Keywords: ESP32, CSI, Channel State Information, Human Activity Recognition, Pose Estimation, Deep Learning, Transformer, IoT, Privacy-Preserving Sensing, Edge AI*
