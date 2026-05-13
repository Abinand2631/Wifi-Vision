# Wi-Fi Vision: 3D Digital Twin

**Note:** This is a fully functional, working model developed as my B.Tech final year project.

Wi-Fi Vision is a vision-free, real-time human posture inference system. It utilizes Wi-Fi Channel State Information (CSI) extracted from three ESP32 devices to infer 3D human pose and classify activity states (e.g., Standing, Sitting, Walking, Lying Down). The project features a deep learning architecture (TEDNet Transformer) and a sleek Streamlit dashboard for real-time visualization.

> **Note on Dataset Privacy:** The raw video datasets and accompanying recorded media used during the data collection process have **not** been included in this repository due to privacy constraints. 

## Hardware Requirements

- **GPU:** An NVIDIA Graphics Card is highly recommended/required for training the deep learning model (`train.py`) and running smooth inference.
- **ESP32:** 3x ESP32 microcontrollers are required (1 acting as a Transmitter/TX, and 2 acting as Receivers/RX for extracting the CSI data).
- **USB Cables:** For connecting the ESP32 boards to the PC.

## Software Dependencies

Ensure you have Python installed. You can install the required packages using `pip`:

```bash
pip install torch torchvision numpy pyserial matplotlib tqdm streamlit
```

*Note: For the best performance, ensure you install the CUDA-enabled version of PyTorch corresponding to your NVIDIA GPU drivers.*

## ESP32 CSI Setup

To extract the Wi-Fi CSI data, you need to flash both ESP32 microcontrollers with the CSI extraction firmware.

1. **Download the Firmware:** [ESP CSI Master Code](https://drive.google.com/file/d/15BBSO7Kxio0WqTDHiqB7bMMBoByQiWyD/view?usp=drive_link)
2. **Environment:** You must use **ESP-IDF** (Espressif IoT Development Framework) to compile and upload the firmware to the ESP32 boards.
3. **Configuration:** Make sure to set the Baud Rate to **115200** in the ESP-IDF monitor and in your device manager settings to ensure stable communication.

## Steps to Reproduce & Run

Once your hardware is set up and the ESP32s are flashed and connected:

1. **Data Collection:** Use `1_capture.py` to record CSI data along with ground truth (if you are creating your own dataset).
2. **Data Processing:** Run `3_extraction.py` to preprocess the captured data and prepare it for training.
3. **Model Training:** Run `train.py` to train the TEDNet 3D Transformer model. This will output the best weights into the `models/` directory.
4. **Live Dashboard:** 
   Execute the live Streamlit dashboard to see the real-time posture inference:
   ```bash
   streamlit run dashboard.py
   ```
   Or simply double-click the `Start_WiFi_Pose_dashboard.bat` script.

## System Architecture

- **TEDNet Model:** 1D CNN for feature extraction coupled with a Transformer Encoder to capture temporal dependencies over a 30-frame window.
- **Classification Engine:** Advanced thresholding and kinetic gating to detect specific human activities based on the 3D keypoints derived from the Wi-Fi CSI.
