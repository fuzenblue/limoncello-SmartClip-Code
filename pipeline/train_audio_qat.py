
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

N_MFCC = 13
NUM_CLASSES = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS_FLOAT = 20
EPOCHS_QAT = 10
QUANT_DELAY = 900  # Start fake-quantization after ~900 steps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "audio")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATASET_FILE = os.path.join(DATA_DIR, "audio_features.csv")
MODEL_PTH = os.path.join(MODEL_DIR, "audio_classifier_qat.pth")

# ──────────────────────────────────────────────────────────────────────────────
# FAKE QUANTIZATION MODULES
# ──────────────────────────────────────────────────────────────────────────────

class FakeQuantize(nn.Module):
    """
    Simulates INT8 quantization effects during training.
    Implements symmetric quantization with a scale and zero_point.
    """
    def __init__(self, per_channel=False, num_channels=1, symmetric=True):
        super(FakeQuantize, self).__init__()
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.register_buffer('enabled', torch.tensor([0], dtype=torch.uint8))
        
        # In symmetric quantization, we usually track max absolute value
        # For Conv1d output (N, C, L), we want scale shape (1, C, 1)
        shape = (1, num_channels, 1) if per_channel else (1,)
        self.register_buffer('scale', torch.ones(shape))
        self.register_buffer('zero_point', torch.zeros(shape))

    def forward(self, x):
        if not self.enabled:
            return x
        
        # Calculate scale (simplified EMA or min-max)
        # For symmetric INT8: scale = max(abs(x)) / 127
        if self.training:
            with torch.no_grad():
                if self.per_channel:
                    # Assume x is (N, C, L)
                    axes = list(range(len(x.shape)))
                    axes.remove(1) # Don't reduce over channel
                    abs_max = x.abs().amax(dim=axes, keepdim=True) # shape (1, C, 1)
                    new_scale = abs_max / 127.0
                    self.scale.copy_(0.9 * self.scale + 0.1 * new_scale)
                else:
                    abs_max = x.abs().max()
                    new_scale = abs_max / 127.0
                    self.scale.copy_(0.9 * self.scale + 0.1 * new_scale)

        # Apply fake quantization
        # x_q = clamp(round(x / scale)) * scale
        x_q = torch.round(x / (self.scale + 1e-8))
        x_q = torch.clamp(x_q, -128, 127)
        return x_q * self.scale

    def enable(self):
        self.enabled.fill_(1)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (PyTorch with QAT)
# ──────────────────────────────────────────────────────────────────────────────

class AudioClassifier1DCNN(nn.Module):
    """
    1D-CNN for Audio Classification (MFCC input).
    Designed for ESP32-S3 deployment with QAT Readiness.
    """
    def __init__(self, num_classes=3, use_qat=True):
        super(AudioClassifier1DCNN, self).__init__()
        self.use_qat = use_qat
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.q1 = FakeQuantize(per_channel=True, num_channels=32) if use_qat else nn.Identity()
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.q2 = FakeQuantize(per_channel=True, num_channels=64) if use_qat else nn.Identity()
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.q3 = FakeQuantize(per_channel=True, num_channels=128) if use_qat else nn.Identity()
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Dense classification
        self.dense1 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.q4 = FakeQuantize() if use_qat else nn.Identity()
        
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (N, 1, 13)
        x = self.q1(self.relu1(self.bn1(self.conv1(x))))
        x = self.q2(self.relu2(self.bn2(self.conv2(x))))
        x = self.q3(self.relu3(self.bn3(self.conv3(x))))
        
        x = self.gap(x).squeeze(-1) # (N, 128)
        
        x = self.q4(self.relu4(self.dense1(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

    def enable_qat(self):
        for m in self.modules():
            if isinstance(m, FakeQuantize):
                m.enable()

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path):
    if not os.path.exists(path):
        # Generate dummy data for testing if dataset is missing
        print(f"⚠ Dataset not found at {path}. Generating dummy data.")
        X = np.random.randn(1000, 1, N_MFCC).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, 1000).astype(np.int64)
    else:
        print(f"📂 Loading audio features from: {path}")
        df = pd.read_csv(path)
        mfcc_cols = [f"mfcc_{i+1:02d}" for i in range(N_MFCC)]
        X = df[mfcc_cols].values.astype(np.float32)
        y = df["sc_class"].values.astype(np.int64)
        X = X.reshape(-1, 1, N_MFCC)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, step_counter=None):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        # Enable QAT after quant_delay steps
        if step_counter is not None:
            step_counter[0] += 1
            if step_counter[0] == QUANT_DELAY:
                print(f"\n   [INFO] Step {QUANT_DELAY} reached: Enabling Fake Quantization (QAT)")
                model.enable_qat()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1

def run_qat_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"── Running QAT Pipeline on {device} ──")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data(DATASET_FILE)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 2. Build Model
    model = AudioClassifier1DCNN(NUM_CLASSES, use_qat=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training with QAT awareness
    print(f"\nPhase 1: Float32 Pre-training & QAT Warm-up ({EPOCHS_FLOAT} epochs)")
    steps = [0]
    for epoch in range(EPOCHS_FLOAT):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, steps)
        acc, f1 = evaluate(model, test_loader, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:02d}: Loss={loss:.4f}, Test Acc={acc:.4f}, Macro-F1={f1:.4f}")

    print(f"\nPhase 2: Full QAT Fine-tuning ({EPOCHS_QAT} epochs)")
    # Ensure QAT is enabled if warm-up was short
    if steps[0] < QUANT_DELAY:
        print("   [INFO] Forcing QAT enable for fine-tuning phase")
        model.enable_qat()
        
    for epoch in range(EPOCHS_QAT):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc, f1 = evaluate(model, test_loader, device)
        print(f"   QAT Epoch {epoch+1:02d}: Loss={loss:.4f}, Test Acc={acc:.4f}, Macro-F1={f1:.4f}")

    # 4. Final Evaluation
    acc_final, f1_final = evaluate(model, test_loader, device)
    print("\n" + "="*40)
    print("FINAL QAT RESULTS")
    print("="*40)
    print(f"Target Macro-F1: >= 0.75")
    print(f"Actual Macro-F1:    {f1_final:.4f} {'✅' if f1_final >= 0.75 else '⚠️'}")
    print(f"Final Accuracy:     {acc_final:.4f}")
    
    # 5. Export
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PTH)
    print(f"\n📁 Saved QAT model state: {MODEL_PTH}")
    
    # Note on Exporting to ESP32:
    # In a real pipeline, the FakeQuantize layers' scale and zero_point 
    # would be used to export the final static quantised model (INT8).

if __name__ == "__main__":
    run_qat_pipeline()
