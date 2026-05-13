"""
STEP 1 — CSI + Video Capture (Unlocked for Isolated Poses)
==========================================================
Records CSI from 2 ESP32 receivers simultaneously with 720p webcam video.
Now supports isolated pose datasets (Sitting, Standing, Walking).
"""

import cv2
import serial
import time
import csv
import os
import re
import threading
from collections import defaultdict

# ===============================
# CONFIGURATION
# ===============================
SERIAL_PORTS = {
    1: "COM10",   # Receiver 1 
    2: "COM4",   # Receiver 2 
}

BAUD_RATE = 115200

# 3 minutes to provide enough training data per pose
RECORD_DURATION = 180        

OUTPUT_FOLDER = "csi_video_capture_external"

# *** SET THIS BEFORE EACH RECORDING SESSION ***
# Options: "Person", "Empty", "Sitting", "Standing", "Walking"
SESSION_LABEL = "Walking"    # <-- Change this to record your isolated datasets!

# Regex matches CSI line format
CSI_PATTERN = re.compile(r'CSI_DATA,\d+,[^"]*,"\[([^][]*)\]"')

# ===============================
# Global variables
# ===============================
csi_data   = defaultdict(list)   
running    = True
start_time = None
start_lock = threading.Lock()    
csi_status = {1: {"count": 0, "last_time": 0}, 
              2: {"count": 0, "last_time": 0}}  

def capture_csi(dev_id, port):
    global running, start_time, csi_status
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"[Receiver {dev_id}] Connected to {port} at {BAUD_RATE} baud")

        while running:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors='ignore').rstrip('\x00')
                    if line and "CSI_DATA" in line:
                        match = CSI_PATTERN.search(line)
                        if match:
                            amps_str = match.group(1).strip()
                            try:
                                amps_str_clean = re.sub(r'[^\d,-]', '', amps_str)
                                amps = [int(x.strip()) for x in amps_str_clean.split(',') if x.strip()]
                                if amps:
                                    with start_lock:
                                        t0 = start_time
                                    if t0 is None:
                                        continue
                                    ts = time.time() - t0
                                    csi_data[dev_id].append((ts, amps))
                                    
                                    csi_status[dev_id]["count"] = len(csi_data[dev_id])
                                    csi_status[dev_id]["last_time"] = time.time()
                                    
                            except Exception as e:
                                print(f"[Receiver {dev_id}] Parse error: {e} | snippet: {amps_str[:80]}")
                else:
                    time.sleep(0.005)
            except Exception as e:
                print(f"[Receiver {dev_id}] Serial error: {e}")
                break

        ser.close()
        print(f"[Receiver {dev_id}] Port closed")

    except Exception as e:
        print(f"[Receiver {dev_id}] Failed to open {port}: {e}")

def draw_overlay(frame, label, elapsed_time, remaining_time):
    """Draw informative overlay on video frame"""
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Title Color Logic: Red for Empty Room, Green for any Human Pose
    title_color = (100, 100, 255) if label == "Empty" else (0, 255, 0)
    cv2.putText(frame, f"Recording: {label.upper()}", (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, title_color, 2)
    
    timer_text = f"Time: {int(elapsed_time)}s / {RECORD_DURATION}s"
    cv2.putText(frame, timer_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    progress = min(elapsed_time / RECORD_DURATION, 1.0)
    bar_width = w - 40
    bar_height = 20
    bar_x, bar_y = 20, 100
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (50, 50, 50), -1)
    fill_width = int(bar_width * progress)
    bar_color = (0, 255, 0) if progress < 0.9 else (0, 165, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                  bar_color, -1)
    cv2.putText(frame, f"{int(progress * 100)}%", 
                (bar_x + bar_width + 10, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    status_y = h - 120
    cv2.rectangle(frame, (0, status_y), (w, h), (0, 0, 0), -1)
    frame[status_y:h, 0:w] = cv2.addWeighted(frame[status_y:h, 0:w], 0.6, 
                                              frame[status_y:h, 0:w], 0.4, 0)
    
    rx1_color = (0, 255, 0) if time.time() - csi_status[1]["last_time"] < 1.0 else (0, 0, 255)
    cv2.putText(frame, f"RX1: {csi_status[1]['count']} packets", 
                (20, status_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, rx1_color, 2)
    
    rx2_color = (0, 255, 0) if time.time() - csi_status[2]["last_time"] < 1.0 else (0, 0, 255)
    cv2.putText(frame, f"RX2: {csi_status[2]['count']} packets", 
                (20, status_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, rx2_color, 2)
    
    if int(time.time() * 2) % 2 == 0: 
        cv2.circle(frame, (w - 40, 40), 15, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (w - 85, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, "Press 'q' to stop early", 
                (20, status_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return frame

def main():
    global running, start_time

    # Unlocked Validation logic
    label = SESSION_LABEL.strip().capitalize()
    allowed_labels = ["Person", "Empty", "Sitting", "Standing", "Walking"]
    
    if label not in allowed_labels:
        print(f"[ERROR] SESSION_LABEL must be one of {allowed_labels}, got: '{SESSION_LABEL}'")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    threads = []
    for dev_id, port in SERIAL_PORTS.items():
        t = threading.Thread(target=capture_csi, args=(dev_id, port), daemon=True)
        t.start()
        threads.append(t)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[WARNING] External webcam (index 1) not found. Trying built-in (index 0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No camera available. Exiting.")
            running = False
            for t in threads:
                t.join(timeout=2)
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps <= 0 or cam_fps > 120:
        cam_fps = 30.0

    video_filename = f"video_{label.lower()}.avi"
    video_path     = os.path.join(OUTPUT_FOLDER, video_filename)

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        cam_fps,
        (frame_width, frame_height)
    )

    print(f"\n{'='*60}")
    print(f"[INFO] Session   : {label}")
    print(f"[INFO] Camera    : {frame_width}x{frame_height} @ {cam_fps:.1f} FPS")
    print(f"[INFO] Video out : {video_path}")
    print(f"[INFO] Duration  : {RECORD_DURATION}s")
    print(f"{'='*60}")
    print(f"\nStarting in 3 seconds...")
    
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if ret:
            countdown_frame = frame.copy()
            h, w = countdown_frame.shape[:2]
            cv2.putText(countdown_frame, str(i), (w//2 - 50, h//2),
                       cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 255), 10)
            cv2.imshow(f'{label} Recording', countdown_frame)
            cv2.waitKey(1000)
    
    print("\n🔴 RECORDING STARTED!\n")

    with start_lock:
        start_time = time.time()

    cv2.namedWindow(f'{label} Recording', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'{label} Recording', 1280, 720)

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Camera feed ended.")
            break

        elapsed = time.time() - start_time
        remaining = max(0, RECORD_DURATION - elapsed)

        display_frame = draw_overlay(frame.copy(), label, elapsed, remaining)
        
        out.write(frame)
        cv2.imshow(f'{label} Recording', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            print("\n[INFO] Stopped by user.")
            break

        if elapsed >= RECORD_DURATION:
            print(f"\n[INFO] Reached {RECORD_DURATION}s duration.")
            running = False
            break

    running = False
    time.sleep(0.5)
    for t in threads:
        t.join(timeout=2.0)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("[INFO] Recording stopped. Saving CSI data...")
    print("="*60)

    for dev_id, data_list in csi_data.items():
        if not data_list:
            print(f"[Receiver {dev_id}] ⚠️  No data received — skipping")
            continue

        csv_filename = f"{label}_RX{dev_id}.csv"
        csv_path     = os.path.join(OUTPUT_FOLDER, csv_filename)

        num_vals = len(data_list[0][1])
        header   = ["timestamp"] + [f"subcarrier_{i}" for i in range(num_vals)]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for ts, buf in data_list:
                writer.writerow([ts] + buf)

        print(f"[Receiver {dev_id}] ✅ Saved {len(data_list)} rows → {csv_filename}")

    print(f"\n✅ Video saved → {video_filename}")
    
    print(f"\n{'='*60}")
    print("FILES CREATED:")
    print(f"{'='*60}")
    for dev_id in SERIAL_PORTS:
        print(f"  ✓ {label}_RX{dev_id}.csv")
    print(f"  ✓ video_{label.lower()}.avi")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()