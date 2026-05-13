# Wi-Fi Vision: 3D Digital Twin

**Note:** This is a fully functional, working model developed as my B.Tech final year project.

Wi-Fi Vision is a vision-free, real-time human posture inference system. It utilizes Wi-Fi Channel State Information (CSI) extracted from three ESP32 devices to infer 3D human pose and classify activity states (e.g., Standing, Sitting, Walking, Lying Down). The project features a deep learning architecture (TEDNet Transformer) and a sleek Streamlit dashboard for real-time visualization.

> **Note on Dataset Privacy (Authenticity):** Unlike many simulated or theoretical repositories online, this is a fully genuine, hardware-tested project. The raw video datasets and accompanying recorded media used during the data collection process have simply been removed from this repository strictly due to personal privacy constraints. 

## Hardware Requirements

- **GPU:** An NVIDIA Graphics Card is highly recommended/required for training the deep learning model (`train.py`) and running smooth inference.
- **ESP32:** 3x ESP32 microcontrollers are required (1 acting as a Transmitter/TX, and 2 acting as Receivers/RX for extracting the CSI data).
  - *Pro Tip:* This project was successfully developed and tested using the ESP32's built-in PCB antennas. However, using ESP32 modules with **external antennas** will significantly improve signal quality, range, and overall model accuracy.
- **Webcam:** A webcam is required to generate the ground-truth video dataset during the data collection and training phase.
- **USB Cables:** For connecting the ESP32 boards to the PC.

## Software Dependencies

Ensure you have Python installed. You can install the required packages using `pip`:

```bash
pip install torch torchvision numpy pyserial matplotlib tqdm streamlit
```

*Note: For the best performance, ensure you install the CUDA-enabled version of PyTorch corresponding to your NVIDIA GPU drivers.*

## ESP32 CSI Setup

To extract the Wi-Fi CSI data, you need to flash the 3 ESP32 microcontrollers with the CSI extraction firmware using ESP-IDF (Espressif IoT Development Framework).

### Firmware Uploading Steps:
1. **Download the Firmware:** Download and extract the [ESP CSI Master Code](https://drive.google.com/file/d/15BBSO7Kxio0WqTDHiqB7bMMBoByQiWyD/view?usp=drive_link).
2. **Open ESP-IDF:** Open your ESP-IDF Command Prompt / Terminal and navigate to the extracted firmware directory.
3. **Set the Target:** Run `idf.py set-target esp32` to configure the environment for the ESP32 chip.
4. **Configuration:** Run `idf.py menuconfig`. 
   - Navigate to **Serial flasher config** and ensure the default baud rate is set strictly to **115200**.
5. **Build and Flash:** Connect your ESP32 via USB and run `idf.py -p COM_PORT flash` (replace `COM_PORT` with your actual port, e.g., `COM3`).
6. **Repeat:** Repeat this process for all 3 ESP32 boards (1 TX, 2 RX).

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

## Disclaimer & Contribution

**Disclaimer:** This is an original B.Tech final year project. Please **do not directly copy** or plagiarize this repository for your own academic submissions or commercial use without permission and proper attribution.

**Contributions:** We absolutely welcome valid developers, researchers, and hobbyists to fork, experiment, and build upon this project! If you have improvements, bug fixes, or optimizations, feel free to open a Pull Request.
