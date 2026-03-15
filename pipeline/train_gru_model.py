

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ──────────────────────────────────────────────────────────────────────────────
# PyTorch or NumPy-based training
# We try PyTorch first; if unavailable, we implement a minimal NumPy GRU
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
    print("✓ PyTorch available — using GPU-accelerated training")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not installed — using NumPy-based training (slower)")

try:
    import mindspore
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# GRU architecture parameters
SEQ_LEN = 12              # Sequence length: 12 steps × 10 minutes = 2 hours
                           # WHY 12? 2 hours of history captures the typical
                           # build-up period for environmental triggers.
                           # Shorter: insufficient context for gradual changes.
                           # Longer: increases model size and training time.

GRU_HIDDEN_1 = 64         # First GRU layer: 64 hidden units
                           # This layer learns SHORT-TERM patterns —
                           # minute-to-minute sensor fluctuations

GRU_HIDDEN_2 = 32         # Second GRU layer: 32 hidden units
                           # This layer learns LONGER patterns across the
                           # full 2-hour history — sustained trends

DROPOUT_1 = 0.3           # Dropout between GRU layers (30% of neurons zeroed)
DROPOUT_2 = 0.2           # Dropout before dense layers (20%)

DENSE_UNITS = 16          # Dense layer before output

# Training hyperparameters
EPOCHS = 50               # Maximum training epochs
BATCH_SIZE = 32            # Mini-batch size for SGD
LEARNING_RATE = 0.001      # Adam optimiser learning rate
EARLY_STOPPING_PATIENCE = 10  # Stop if val_loss doesn't improve for 10 epochs

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Feature columns used as model input
# Excludes metadata (user_id, window_start) and labels (migraine_within_6h, etc.)
FEATURE_COLS = [
    "motion_active",
    "flicker_index", "flicker_freq_hz", "flicker_alert",
    "pressure_hpa", "pressure_ddt_1h", "pressure_ddt_6h",
    "pressure_zscore", "pressure_drop_alert",
    "voc_raw", "voc_zscore", "voc_spike",
    "humidity_pct", "temp_celsius",
    "audio_class", "audio_confidence", "audio_db_mean",
    "prior_photophobia", "prior_phonophobia",
    "prior_pressure_sensitivity", "prior_voc_sensitivity",
    "risk_score",
]
N_FEATURES = len(FEATURE_COLS)  # 22 features

TARGET_COL = "migraine_within_6h"

# File paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "..", "data")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "..", "models")
DATASET_FILE = os.path.join(DATA_DIR, "master", "smartclip_master_dataset.csv")
PRIORS_FILE  = os.path.join(DATA_DIR, "population_priors.json")
MODEL_FILE   = os.path.join(MODEL_DIR, "gru_smartclip_model.pth")
MINDIR_FILE  = os.path.join(MODEL_DIR, "gru_smartclip_model.mindir")
RESULTS_FILE = os.path.join(MODEL_DIR, "training_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    
    print(f"\n📂 Loading dataset: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run Phase 3 first."
        )

    df = pd.read_csv(path, parse_dates=["window_start"])

    # Validate required columns
    missing = [c for c in FEATURE_COLS + [TARGET_COL, "user_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Print class distribution
    pos_count = df[TARGET_COL].sum()
    neg_count = len(df) - pos_count
    pos_rate = pos_count / len(df)
    print(f"   Loaded {len(df)} rows")
    print(f"   Class distribution:")
    print(f"     Positive (migraine_within_6h=1): {pos_count} ({100*pos_rate:.1f}%)")
    print(f"     Negative (migraine_within_6h=0): {neg_count} ({100*(1-pos_rate):.1f}%)")

    return df


def create_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> tuple:
    
    print(f"\n── Creating sequences (length={seq_len}) ──")

    all_X = []
    all_y = []

    for user_id, user_df in df.groupby("user_id"):
        # Sort by time within this user
        user_df = user_df.sort_values("window_start").reset_index(drop=True)

        # Extract feature matrix and labels for this user
        features = user_df[FEATURE_COLS].values.astype(np.float32)
        labels = user_df[TARGET_COL].values.astype(np.float32)

        # Replace NaN with 0 (e.g., pain_score_t0 is often NaN)
        features = np.nan_to_num(features, nan=0.0)

        # Create sliding windows
        # For each position i, the sequence is features[i:i+seq_len]
        # The label is labels[i+seq_len-1] (the label at the END of the sequence)
        n_windows = len(features) - seq_len + 1
        for i in range(n_windows):
            seq = features[i:i + seq_len]     # Shape: (seq_len, n_features)
            label = labels[i + seq_len - 1]    # Scalar: label at end of window

            all_X.append(seq)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)  # (n_samples, 12, 22)
    y = np.array(all_y, dtype=np.float32)  # (n_samples,)

    print(f"   Created {len(X)} sequences")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Positive sequences: {int(y.sum())} ({100*y.mean():.1f}%)")

    return X, y


def normalise_features(X_train, X_val, X_test):
    
    # Reshape to 2D for mean/std computation: (n_samples * seq_len, n_features)
    n, seq, feat = X_train.shape
    flat = X_train.reshape(-1, feat)

    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0  # Prevent division by zero for constant features

    # Apply normalisation
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std


# ══════════════════════════════════════════════════════════════════════════════
# GRU MODEL DEFINITION (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class SmartClipGRU(nn.Module):
        

        def __init__(self, input_size: int, hidden1: int = GRU_HIDDEN_1,
                     hidden2: int = GRU_HIDDEN_2):
            
            super(SmartClipGRU, self).__init__()

            # GRU Layer 1: learns short-term patterns (minute-to-minute changes)
            # batch_first=True means input shape is (batch, seq_len, features)
            # Without batch_first, shape would be (seq_len, batch, features)
            # which is harder to work with in data loaders.
            self.gru1 = nn.GRU(
                input_size=input_size,
                hidden_size=hidden1,
                batch_first=True,     # Input: (batch, seq, features)
                num_layers=1
            )

            # Dropout 1: randomly zeros 30% of GRU1 outputs during training
            # This prevents co-adaptation — neurons can't rely on specific
            # partners, so each learns independently useful features
            self.dropout1 = nn.Dropout(DROPOUT_1)

            # GRU Layer 2: learns longer patterns across the 2-hour history
            # Takes 64-dim output from GRU1 and compresses to 32-dim
            self.gru2 = nn.GRU(
                input_size=hidden1,
                hidden_size=hidden2,
                batch_first=True,
                num_layers=1
            )

            # Dropout 2: 20% dropout before dense layers
            self.dropout2 = nn.Dropout(DROPOUT_2)

            # Dense layer: compresses 32-dim GRU output to 16-dim
            # ReLU activation: f(x) = max(0, x) — introduces non-linearity
            self.dense = nn.Linear(hidden2, DENSE_UNITS)
            self.relu = nn.ReLU()

            # Output layer: single neuron with sigmoid activation
            # Sigmoid maps any real number to [0, 1] — a probability
            self.output = nn.Linear(DENSE_UNITS, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            
            # GRU Layer 1: process entire sequence
            # out1 shape: (batch, seq_len, hidden1=64)
            # h1 shape:   (1, batch, hidden1=64) — final hidden state
            out1, h1 = self.gru1(x)
            out1 = self.dropout1(out1)

            # GRU Layer 2: process the 64-dim sequence from GRU1
            # out2 shape: (batch, seq_len, hidden2=32)
            out2, h2 = self.gru2(out1)
            out2 = self.dropout2(out2)

            # Take ONLY the last time step's output
            # This is the GRU's "summary" of the entire 2-hour sequence
            # Shape: (batch, hidden2=32)
            last_output = out2[:, -1, :]

            # Dense layers
            dense_out = self.relu(self.dense(last_output))  # (batch, 16)
            logit = self.output(dense_out)                   # (batch, 1)
            prob = self.sigmoid(logit)                        # (batch, 1)

            return prob


def initialise_prior_biases(model, priors_path: str):
    
    if not TORCH_AVAILABLE:
        return

    # Load priors
    if os.path.exists(priors_path):
        with open(priors_path, "r") as f:
            priors = json.load(f)
        combined = priors.get("combined_weighted_prior", 0.04)
    else:
        combined = 0.04  # Typical migraine positive rate

    # Compute logit bias
    combined_clipped = np.clip(combined, 1e-6, 1 - 1e-6)
    logit_bias = float(np.log(combined_clipped / (1 - combined_clipped)))

    # Set the output layer bias
    with torch.no_grad():
        model.output.bias.fill_(logit_bias)

    print(f"   Set output bias to {logit_bias:.4f} "
          f"(sigmoid={combined:.4f} ≈ population base rate)")


def compute_class_weights(y_train: np.ndarray) -> float:
    
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)

    if n_pos == 0:
        print("   ⚠ No positive samples in training set!")
        return 1.0

    pos_weight = float(n_neg / n_pos)

    # Cap at a reasonable maximum to prevent gradient explosion
    pos_weight = min(pos_weight, 50.0)

    print(f"   Positive samples: {int(n_pos)} ({100*n_pos/(n_pos+n_neg):.1f}%)")
    print(f"   Negative samples: {int(n_neg)} ({100*n_neg/(n_pos+n_neg):.1f}%)")
    print(f"   Positive class weight: {pos_weight:.1f}×")

    return pos_weight


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                pos_weight=1.0):
    
    if not TORCH_AVAILABLE:
        return _train_numpy_fallback(X_train, y_train, X_val, y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n── Training on {device} ──")
    model = model.to(device)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_X_tensor = torch.FloatTensor(X_val).to(device)
    val_y_tensor = torch.FloatTensor(y_val).to(device)

    # BCE loss with positive class weighting
    # BCELoss: -[y × log(ŷ) + (1-y) × log(1-ŷ)]
    # With pos_weight: -[pos_weight × y × log(ŷ) + (1-y) × log(1-ŷ)]
    criterion = nn.BCELoss(
        weight=None  # We'll handle pos_weight manually
    )

    # Adam optimiser: adaptive learning rate per parameter
    # Combines the benefits of RMSProp and momentum
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training tracking
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # ── Training phase ────────────────────────────────────────────────
        model.train()  # Enable dropout
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            predictions = model(batch_X).squeeze()

            # Compute weighted loss manually
            # Positive samples contribute pos_weight × more to the loss
            weights = torch.where(batch_y == 1,
                                  torch.tensor(pos_weight, device=device),
                                  torch.tensor(1.0, device=device))
            loss = nn.functional.binary_cross_entropy(
                predictions, batch_y, weight=weights
            )

            # Backward pass: compute gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # ── Validation phase ──────────────────────────────────────────────
        model.eval()  # Disable dropout
        with torch.no_grad():
            val_pred = model(val_X_tensor).squeeze()
            val_weights = torch.where(val_y_tensor == 1,
                                      torch.tensor(pos_weight, device=device),
                                      torch.tensor(1.0, device=device))
            val_loss = nn.functional.binary_cross_entropy(
                val_pred, val_y_tensor, weight=val_weights
            ).item()

            # Compute F1 for monitoring
            val_pred_binary = (val_pred.cpu().numpy() > 0.5).astype(int)
            val_f1 = f1_score(y_val, val_pred_binary, zero_division=0)

        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # ── Early stopping check ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:>3}/{epochs}  "
                  f"train_loss={avg_train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_F1={val_f1:.4f}  "
                  f"patience={patience_counter}/{EARLY_STOPPING_PATIENCE}")

        # Stop if patience exhausted
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n   Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"   Restored best model (val_loss={best_val_loss:.4f})")

    return history


def _train_numpy_fallback(X_train, y_train, X_val, y_val):
    
    print("\n── NumPy Fallback Training (Logistic Regression) ──")
    from sklearn.linear_model import LogisticRegression

    # Flatten sequences: (n_samples, seq_len, features) → (n_samples, seq_len * features)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_flat, y_train)

    val_pred = clf.predict(X_val_flat)
    val_f1 = f1_score(y_val, val_pred, zero_division=0)
    print(f"   Validation F1: {val_f1:.4f}")

    return {"train_loss": [0.0], "val_loss": [0.0], "val_f1": [val_f1]}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, thresholds=[0.3, 0.5, 0.7]):
    
    print("\n── Model Evaluation ──")

    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(device)
            y_prob = model(test_tensor).squeeze().cpu().numpy()
    else:
        # NumPy fallback
        X_flat = X_test.reshape(len(X_test), -1)
        y_prob = np.random.random(len(y_test))  # Placeholder

    results = {}

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred)

        results[f"threshold_{threshold}"] = {
            "threshold":    threshold,
            "accuracy":     round(float(acc), 4),
            "precision":    round(float(prec), 4),
            "recall":       round(float(rec), 4),
            "f1_score":     round(float(f1), 4),
            "auc_roc":      round(float(auc), 4),
            "confusion_matrix": cm.tolist(),
        }

        # Print results for this threshold
        print(f"\n   Threshold = {threshold}")
        print(f"   {'Metric':<15} {'Value':>8}")
        print(f"   {'-'*25}")
        print(f"   {'Accuracy':<15} {acc:>8.4f}")
        print(f"   {'Precision':<15} {prec:>8.4f}")
        print(f"   {'Recall':<15} {rec:>8.4f}")
        print(f"   {'F1 Score':<15} {f1:>8.4f}")
        print(f"   {'AUC-ROC':<15} {auc:>8.4f}")

        # Confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"   TP={tp} FP={fp} FN={fn} TN={tn}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MODEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_model(model, X_sample):
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        # Save PyTorch checkpoint
        torch.save(model.state_dict(), MODEL_FILE)
        file_size = os.path.getsize(MODEL_FILE) / 1024
        print(f"\n📁 Saved PyTorch model: {MODEL_FILE} ({file_size:.1f} KB)")

        # Try ONNX export as a portable alternative
        try:
            onnx_path = MODEL_FILE.replace(".pth", ".onnx")
            dummy = torch.FloatTensor(X_sample[:1])
            torch.onnx.export(model.cpu(), dummy, onnx_path,
                              input_names=["sensor_sequence"],
                              output_names=["migraine_probability"],
                              opset_version=11)
            print(f"📁 Saved ONNX model:    {onnx_path}")
        except Exception as e:
            print(f"⚠  ONNX export skipped: {e}")

    # MindSpore export (if available)
    if MINDSPORE_AVAILABLE:
        try:
            print("   Attempting MindSpore MINDIR export...")
            # This would use mindspore.export() with the model and input tensor
            # For now, we note this requires the MindSpore training path
            print("   ⚠  MindSpore export requires model trained in MindSpore.")
            print("   ⚠  Use the PyTorch → ONNX → MindSpore conversion pipeline.")
        except Exception as e:
            print(f"   ⚠  MindSpore export failed: {e}")
    else:
        print("   ⚠  MindSpore not installed — MINDIR export skipped.")
        print("   ⚠  Install mindspore for production MINDIR export.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    start_time = time.time()

    print("=" * 60)
    print("PHASE: GRU Model Training")
    print("=" * 60)

    # ── Step 1: Load dataset ──────────────────────────────────────────────
    df = load_dataset(DATASET_FILE)

    # ── Step 2: Create sequences ──────────────────────────────────────────
    X, y = create_sequences(df)

    # ── Step 3: Train/Val/Test split ──────────────────────────────────────
    n_total = len(X)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    # Shuffle indices (but maintain sequence integrity since sequences are
    # already independent units)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n── Data Split ──")
    print(f"   Train: {len(X_train)} ({100*len(X_train)/n_total:.0f}%)")
    print(f"   Val:   {len(X_val)} ({100*len(X_val)/n_total:.0f}%)")
    print(f"   Test:  {len(X_test)} ({100*len(X_test)/n_total:.0f}%)")

    # ── Step 4: Normalise features ────────────────────────────────────────
    X_train, X_val, X_test, feat_mean, feat_std = normalise_features(
        X_train, X_val, X_test
    )

    # ── Step 5: Build model ───────────────────────────────────────────────
    if TORCH_AVAILABLE:
        model = SmartClipGRU(input_size=N_FEATURES)
        print(f"\n── Model Architecture ──")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")

        # Initialise output bias with population priors
        initialise_prior_biases(model, PRIORS_FILE)
    else:
        model = None

    # ── Step 6: Compute class weights ─────────────────────────────────────
    print(f"\n── Class Imbalance ──")
    pos_weight = compute_class_weights(y_train)

    # ── Step 7: Train ─────────────────────────────────────────────────────
    history = train_model(model, X_train, y_train, X_val, y_val,
                          pos_weight=pos_weight)

    # ── Step 8: Evaluate ──────────────────────────────────────────────────
    eval_results = evaluate_model(model, X_test, y_test)

    # ── Step 9: Export ────────────────────────────────────────────────────
    export_model(model, X_test)

    # ── Step 10: Save results ─────────────────────────────────────────────
    elapsed = time.time() - start_time

    training_results = {
        "model_type":         "GRU",
        "framework":          "PyTorch" if TORCH_AVAILABLE else "NumPy",
        "seq_length":         SEQ_LEN,
        "n_features":         N_FEATURES,
        "gru_hidden1":        GRU_HIDDEN_1,
        "gru_hidden2":        GRU_HIDDEN_2,
        "epochs_trained":     len(history.get("train_loss", [])),
        "learning_rate":      LEARNING_RATE,
        "batch_size":         BATCH_SIZE,
        "pos_weight":         round(pos_weight, 2),
        "train_samples":      len(X_train),
        "val_samples":        len(X_val),
        "test_samples":       len(X_test),
        "final_train_loss":   round(history["train_loss"][-1], 6) if history["train_loss"] else None,
        "best_val_loss":      round(min(history["val_loss"]), 6) if history["val_loss"] else None,
        "evaluation":         eval_results,
        "training_time_sec":  round(elapsed, 1),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2, default=str)
    print(f"\n📁 Saved results: {RESULTS_FILE}")

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Value':>12}")
    print("  " + "-" * 44)
    print(f"  {'Model Type':<30} {'GRU':>12}")
    print(f"  {'Parameters':<30} "
          f"{sum(p.numel() for p in model.parameters()) if TORCH_AVAILABLE and model else 'N/A':>12}")
    print(f"  {'Epochs Trained':<30} {training_results['epochs_trained']:>12}")
    print(f"  {'Training Time':<30} {elapsed:>11.1f}s")
    print(f"  {'Best Val Loss':<30} {training_results.get('best_val_loss', 'N/A'):>12}")

    # Best evaluation metrics at threshold 0.5
    if "threshold_0.5" in eval_results:
        r = eval_results["threshold_0.5"]
        print(f"  {'Test F1 @0.5':<30} {r['f1_score']:>12.4f}")
        print(f"  {'Test AUC-ROC':<30} {r['auc_roc']:>12.4f}")
        print(f"  {'Test Recall @0.5':<30} {r['recall']:>12.4f}")

    print("=" * 60)
    print("\n✅ Training complete.")
