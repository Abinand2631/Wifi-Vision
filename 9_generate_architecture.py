"""
STEP 9 — IEEE Architecture Diagram Generator
============================================
Generates a high-resolution 300 DPI block diagram of the 
WiFi-Vision system using Graphviz.
"""

import graphviz
import os

os.makedirs("figures", exist_ok=True)

def generate_architecture():
    print("Generating Figure 1: System Architecture...")
    
    # Initialize the directed graph
    dot = graphviz.Digraph('Architecture', format='png')
    dot.attr(rankdir='LR', size='10,6', dpi='300', fontname='Times-Roman')
    
    # Default styling for all boxes and arrows
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', 
             fontname='Times-Roman', fontsize='12', margin='0.2')
    dot.attr('edge', fontname='Times-Roman', fontsize='10', color='#333333')

    # ---------------------------------------------------------
    # PART 1: Live Inference Pipeline (Top)
    # ---------------------------------------------------------
    with dot.subgraph(name='cluster_live') as live:
        live.attr(label='Real-Time Vision-Free Inference Pipeline', 
                  style='dashed', color='#1f77b4', fontname='Times-Roman', fontsize='14')
        
        # Hardware & RF Environment
        live.node('Tx', 'ESP32 (Tx)\n2.4 GHz', fillcolor='#e1f5fe')
        live.node('Human', 'Human Subject\n(Multipath Reflections)', shape='ellipse', fillcolor='#fff9c4')
        live.node('Rx', 'ESP32 (Rx1 & Rx2)\n128 Subcarriers', fillcolor='#e1f5fe')
        
        # Signal Processing
        live.node('Filter', 'Signal Processing\n(Hampel & Butterworth)', fillcolor='#f3e5f5')
        live.node('Gate', 'Temporal MAD\nKinetic Gate', shape='diamond', fillcolor='#ffcdd2')
        live.node('Window', '100-Packet\nSliding Window', fillcolor='#f3e5f5')

        # Routing
        live.edge('Tx', 'Human', label=' Wi-Fi Waves')
        live.edge('Human', 'Rx', label=' Reflections')
        live.edge('Rx', 'Filter')
        live.edge('Filter', 'Gate')
        live.edge('Gate', 'Window', label=' Activity > 1200')

        # TED-Net Architecture Sub-cluster
        with live.subgraph(name='cluster_tednet') as tednet:
            tednet.attr(label='TED-Net', style='solid', color='black', bgcolor='#f8f9fa')
            tednet.node('CNN', '1D CNN Encoder', fillcolor='#c8e6c9')
            tednet.node('Trans', 'Transformer Encoder\n(4-Layer, 8-Head)', fillcolor='#c8e6c9')
            tednet.node('Reg', 'Linear Regression', fillcolor='#c8e6c9')
            
            tednet.edge('CNN', 'Trans')
            tednet.edge('Trans', 'Reg')

        live.edge('Window', 'CNN')
        
        # Final Output
        live.node('Output', '17-Joint 3D\nSpatial Skeleton', shape='cylinder', fillcolor='#fff3e0', penwidth='2')
        live.edge('Reg', 'Output')

    # ---------------------------------------------------------
    # PART 2: Offline Training Pipeline (Bottom)
    # ---------------------------------------------------------
    with dot.subgraph(name='cluster_offline') as offline:
        offline.attr(label='Offline Camera-Supervised Training Pipeline', 
                     style='dashed', color='#2ca02c', fontname='Times-Roman', fontsize='14')
        
        offline.node('Cam', 'Webcam (RGB)', fillcolor='#e1f5fe')
        offline.node('D2', 'Detectron2\n(2D Keypoints)', fillcolor='#f3e5f5')
        offline.node('MB', 'MotionBERT\n(3D Temporal Lifting)', fillcolor='#f3e5f5')
        offline.node('Target', '51-Value 3D\nTarget Coordinates', shape='cylinder', fillcolor='#fff3e0')

        offline.edge('Cam', 'D2')
        offline.edge('D2', 'MB')
        offline.edge('MB', 'Target')

    # Connect the Training Target to the Neural Network
    dot.edge('Target', 'Reg', label=' MSE Loss Supervision', style='dotted', color='#d62728', constraint='false')

    # Save to the figures folder
    output_path = 'figures/architecture'
    dot.render(output_path, cleanup=True)
    print(f"✅ Success! Architecture diagram saved to: {output_path}.png")

if __name__ == "__main__":
    generate_architecture()