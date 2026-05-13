"""
STEP 3 — CSI Synchronization & Extraction
=======================================
Synchronizes dual-receiver ESP32 CSI data with the 51-value 3D MotionBERT ground truth.
Outputs: data/processed/X.npy and data/processed/Y.npy
"""

import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt

# --- CONFIGURATION ---
WINDOW_SIZE = 30  # Slashed from 100 down to 30 for snappy, zero-delay real-time inference!
STEP_SIZE = 3
OUTPUT_DIR = "data/processed"

LABELS = ["Sitting", "Standing", "Walking"]
GT_DIR = "data/gt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FILTERING LOGIC ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut=0.1, highcut=10.0, fs=100.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def hampel_filter(data, window_size=5, n_sigmas=3):
    """Basic outlier removal for CSI amplitudes."""
    filtered_data = np.copy(data)
    for i in range(data.shape[1]):
        series = pd.Series(data[:, i])
        rolling_median = series.rolling(window=window_size, center=True).median()
        rolling_std = series.rolling(window=window_size, center=True).std()
        
        outliers = np.abs(series - rolling_median) > (n_sigmas * rolling_std)
        filtered_data[outliers, i] = rolling_median[outliers]
    
    return np.nan_to_num(filtered_data)

# --- SYNCHRONIZATION ---
def process_data():
    X_master, Y_master = [], []
    
    from scipy.interpolate import interp1d

    for label in LABELS:
        gt_file = os.path.join(GT_DIR, f"gt_{label.lower()}.npy")
        rx1_file = f"data/{label}_RX1.csv"
        rx2_file = f"data/{label}_RX2.csv"
        
        if not os.path.exists(gt_file) or not os.path.exists(rx1_file):
            print(f"⚠️ Skipping {label} - Missing CSV or GT files!")
            continue

        print(f"\n--- Extracting and Synchronizing {label} ---")
        gt_3d = np.load(gt_file) 
        
        rx1_df = pd.read_csv(rx1_file, on_bad_lines='skip')
        rx2_df = pd.read_csv(rx2_file, on_bad_lines='skip')

        # Extract timestamps
        t_rx1 = rx1_df['timestamp'].values
        t_rx2 = rx2_df['timestamp'].values
        
        csi_rx1 = rx1_df.drop(columns=['timestamp']).values[:, :128].astype(np.float32)
        csi_rx2 = rx2_df.drop(columns=['timestamp']).values[:, :128].astype(np.float32)

        print("Applying Hampel and Butterworth filters to clean RF noise...")
        csi_rx1 = apply_bandpass(hampel_filter(csi_rx1))
        csi_rx2 = apply_bandpass(hampel_filter(csi_rx2))

        print("Performing SciPy Temporal Synchronization for this session...")
        max_duration = max(t_rx1[-1], t_rx2[-1])
        t_video = np.linspace(0, max_duration, len(gt_3d))
        
        f1 = interp1d(t_rx1, csi_rx1, axis=0, bounds_error=False, fill_value="extrapolate")
        f2 = interp1d(t_rx2, csi_rx2, axis=0, bounds_error=False, fill_value="extrapolate")
        
        csi_combined = np.concatenate([f1(t_video), f2(t_video)], axis=1) 
        gt_aligned = gt_3d

        print("Building temporal sliding windows...")
        for start in range(0, len(gt_aligned) - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            mid = (start + end) // 2
            
            label_frame = gt_aligned[mid]
            
            if np.all(label_frame == 0):
                continue
                
            flat_label = label_frame.flatten()
            
            X_master.append(csi_combined[start:end])
            Y_master.append(flat_label)

    X = np.array(X_master, dtype=np.float32)
    Y = np.array(Y_master, dtype=np.float32)

    print(f"\n✅ Extraction Complete!")
    print(f"Features (X) Shape: {X.shape} -> (Samples, Time Window, Subcarriers)")
    print(f"Labels (Y) Shape:   {Y.shape} -> (Samples, 51-Value 3D Coordinates)")
    
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "Y.npy"), Y)

if __name__ == "__main__":
    process_data()