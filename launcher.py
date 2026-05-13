import tkinter as tk
from tkinter import ttk, messagebox
import serial.tools.list_ports
import subprocess
import sys
import os

def get_com_ports():
    """Scans Windows for active COM ports."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def refresh_ports():
    """Updates the dropdown menus with currently plugged-in devices."""
    ports = get_com_ports()
    if not ports:
        ports = ["No Ports Found"]
    
    combo_rx1['values'] = ports
    combo_rx2['values'] = ports
    
    # Auto-select the first two if available
    if len(ports) >= 1 and ports[0] != "No Ports Found":
        combo_rx1.set(ports[0])
    if len(ports) >= 2:
        combo_rx2.set(ports[1])
    elif len(ports) == 1 and ports[0] != "No Ports Found":
        combo_rx2.set(ports[0])

def launch_system():
    """Triggers your main live script with the selected ports."""
    port1 = combo_rx1.get().strip()
    port2 = combo_rx2.get().strip()
    
    if not port1 or not port2 or port1 == "No Ports Found":
        messagebox.showerror("Error", "Please select valid COM ports for both receivers.")
        return
        
    if port1 == port2:
        response = messagebox.askyesno("Warning", "You selected the SAME port for both RX1 and RX2. Are you sure?")
        if not response:
            return

    print(f"Launching WiFi-Pose with RX1: {port1} | RX2: {port2}")
    
    # Launch the main script and close the launcher
    subprocess.Popen([sys.executable, "6_test_live.py", port1, port2])
    root.destroy()

# --- Build the UI ---
root = tk.Tk()
root.title("WiFi-Pose Launch Control")
root.geometry("400x250")
root.configure(padx=20, pady=20)
root.resizable(False, False)

# Styling
style = ttk.Style()
style.configure("TLabel", font=("Arial", 11))
style.configure("TButton", font=("Arial", 10, "bold"))

# Title
ttk.Label(root, text="🚀 WiFi-Pose System", font=("Arial", 16, "bold")).pack(pady=(0, 15))

# RX1 Selection
frame1 = ttk.Frame(root)
frame1.pack(fill="x", pady=5)
ttk.Label(frame1, text="Receiver 1 (RX1):", width=15).pack(side="left")
combo_rx1 = ttk.Combobox(frame1, width=20)
combo_rx1.pack(side="left", padx=10)

# RX2 Selection
frame2 = ttk.Frame(root)
frame2.pack(fill="x", pady=5)
ttk.Label(frame2, text="Receiver 2 (RX2):", width=15).pack(side="left")
combo_rx2 = ttk.Combobox(frame2, width=20)
combo_rx2.pack(side="left", padx=10)

# Buttons
btn_frame = ttk.Frame(root)
btn_frame.pack(fill="x", pady=25)

refresh_btn = ttk.Button(btn_frame, text="🔄 Refresh Ports", command=refresh_ports)
refresh_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

launch_btn = ttk.Button(btn_frame, text="▶ START DEMO", command=launch_system)
launch_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

# Initialize
refresh_ports()
root.eval('tk::PlaceWindow . center')
root.mainloop()