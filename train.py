"""
STEP 4 — TEDNet 3D Training (SOTA Transformer)
==============================================
Maps 256 Wi-Fi subcarriers (2 RX) over 100 frames to 51 continuous 3D values.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
N_FEATURES = 256  # 128 subcarriers * 2 receivers
WINDOW_SIZE = 30
OUTPUT_DIM = 51   # 17 joints * 3 dimensions (X, Y, Z)

# --- 1. DATASET DEFINITION ---
class CSIDataset(Dataset):
    def __init__(self, x_path, y_path):
        print(f"Loading data into RAM...")
        self.X = np.load(x_path)
        self.Y = np.load(y_path)
        
        # Basic normalization for CSI data
        self.X = (self.X - np.mean(self.X)) / (np.std(self.X) + 1e-8)
        print(f"Dataset ready. Samples: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

# --- 2. NEURAL NETWORK ARCHITECTURE ---
class TEDNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D CNN for subcarrier feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(N_FEATURES, 128, kernel_size=3, padding=1), 
            nn.BatchNorm1d(128), 
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), 
            nn.BatchNorm1d(128), 
            nn.GELU()
        )
        # Learnable position embedding for the 100-frame time window
        self.pos_embed = nn.Parameter(torch.randn(1, WINDOW_SIZE, 128) * 0.02)
        
        # Transformer for temporal CSI dependencies
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        
        # Regressor to 51 3D spatial values
        self.regressor = nn.Sequential(
            nn.LayerNorm(128), 
            nn.Linear(128, 64), 
            nn.GELU(),
            nn.Linear(64, OUTPUT_DIM) 
            # Note: No Sigmoid here because 3D coordinates can span negative values
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features) -> CNN expects (Batch, Features, Time)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1) # Back to (Batch, Time, Features)
        
        x = x + self.pos_embed
        x = self.transformer(x)
        
        # Global Average Pooling over time dimension
        x = x.mean(dim=1) 
        
        return self.regressor(x)

# --- 3. TRAINING LOOP ---
def train():
    os.makedirs("models", exist_ok=True)
    
    # Load Data
    try:
        dataset = CSIDataset("data/processed/X.npy", "data/processed/Y.npy")
    except FileNotFoundError:
        print("⚠️ Could not find X.npy or Y.npy. Did you run 3_extraction.py?")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Initialize Model, Loss, and Optimizer
    model = TEDNet().to(DEVICE)
    criterion = nn.MSELoss() # Mean Squared Error is best for 3D coordinate regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    print(f"🔥 Starting Training on {DEVICE.type.upper()}...")
    
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        # Progress bar for the epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch:02d}/{EPOCHS} | Average MSE Loss: {avg_loss:.4f}")
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/tednet_3d_best.pth")
            print(f"   -> 🌟 New Best Model Saved! (Loss: {best_loss:.4f})")

    print("\n✅ Training Complete! Model saved as 'models/tednet_3d_best.pth'")

if __name__ == "__main__":
    train()