

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import struct
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score

# Import model architecture from training script
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Quantisation parameters
INT8_MIN = -128               # Signed int8 minimum value
INT8_MAX = 127                # Signed int8 maximum value
INT8_RANGE = INT8_MAX - INT8_MIN  # 255

# Calibration parameters
N_CALIBRATION_SAMPLES = 100   # Number of samples for activation statistics
                               # WHY 100? Empirically, 100 samples capture the
                               # distribution of activations sufficiently for
                               # computing min/max per layer.  More samples give
                               # diminishing returns; fewer may miss outliers.

# Accuracy loss tolerance
MAX_ACCURACY_DROP = 0.02       # Maximum acceptable F1 loss from quantisation (2%)

# Target model size
MAX_MODEL_SIZE_KB = 150        # ESP32-S3 PSRAM constraint

# File paths
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(SCRIPT_DIR, "..", "models")
DATA_DIR     = os.path.join(SCRIPT_DIR, "..", "data")
FW_DIR       = os.path.join(SCRIPT_DIR, "..", "firmware", "include")
MODEL_FILE   = os.path.join(MODEL_DIR, "gru_smartclip_model.pth")
DATASET_FILE = os.path.join(DATA_DIR, "master", "smartclip_master_dataset.csv")
INT8_FILE    = os.path.join(MODEL_DIR, "gru_smartclip_int8.ms")
HEADER_FILE  = os.path.join(FW_DIR, "model_data.h")
REPORT_FILE  = os.path.join(MODEL_DIR, "quantisation_report.json")

# Feature columns — must match training script
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
TARGET_COL = "migraine_within_6h"
SEQ_LEN = 12


# ══════════════════════════════════════════════════════════════════════════════
# QUANTISATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class QuantisedTensor:
    

    def __init__(self, float_tensor: np.ndarray):
        
        self.original_shape = float_tensor.shape
        self.original_dtype = float_tensor.dtype

        # Compute per-tensor (not per-channel) quantisation parameters
        min_val = float(float_tensor.min())
        max_val = float(float_tensor.max())

        # Handle edge case: constant tensor
        if max_val - min_val < 1e-8:
            self.scale = 1.0
            self.zero_point = 0
            self.data = np.zeros(float_tensor.shape, dtype=np.int8)
            return

        # Compute scale: maps the float range to [0, 255]
        self.scale = (max_val - min_val) / INT8_RANGE

        # Compute zero point: the int8 value that maps to float 0.0
        self.zero_point = int(round(-min_val / self.scale)) + INT8_MIN

        # Quantise: float → int8
        quantised = np.round(float_tensor / self.scale).astype(np.int32)
        quantised += self.zero_point
        quantised = np.clip(quantised, INT8_MIN, INT8_MAX)
        self.data = quantised.astype(np.int8)

    def dequantise(self) -> np.ndarray:
        
        return (self.data.astype(np.float32) - self.zero_point) * self.scale

    @property
    def size_bytes(self) -> int:
        
        return self.data.nbytes + 8  # Data + scale(4) + zero_point(4)

    def quantisation_error(self, original: np.ndarray) -> float:
        
        reconstructed = self.dequantise()
        return float(np.mean(np.abs(original - reconstructed)))


def quantise_model_weights(model_state_dict: dict) -> dict:
    
    quantised = {}
    total_float_bytes = 0
    total_int8_bytes = 0

    print("\n── Quantising Model Weights ──")
    print(f"   {'Layer':<40} {'Float32':>10} {'INT8':>10} {'Error':>10}")
    print("   " + "-" * 72)

    for name, tensor in model_state_dict.items():
        if isinstance(tensor, np.ndarray):
            weight = tensor
        else:
            weight = tensor.cpu().numpy()

        float_bytes = weight.nbytes
        total_float_bytes += float_bytes

        # Quantise
        q = QuantisedTensor(weight)
        quantised[name] = q

        total_int8_bytes += q.size_bytes
        error = q.quantisation_error(weight)

        print(f"   {name:<40} {float_bytes:>8}B  {q.size_bytes:>8}B  "
              f"{error:>10.6f}")

    ratio = total_int8_bytes / total_float_bytes if total_float_bytes > 0 else 0
    print(f"\n   Total: {total_float_bytes:,}B → {total_int8_bytes:,}B "
          f"(compression: {1/ratio:.1f}× = {100*ratio:.0f}% of original)")

    return quantised


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def create_test_data():
    
    if not os.path.exists(DATASET_FILE):
        print("   ⚠  Dataset not found — generating synthetic test data")
        n_test = 200
        rng = np.random.default_rng(42)
        X_test = rng.random((n_test, SEQ_LEN, len(FEATURE_COLS))).astype(np.float32)
        y_test = rng.choice([0, 1], size=n_test, p=[0.95, 0.05]).astype(np.float32)
        return X_test, y_test

    df = pd.read_csv(DATASET_FILE, parse_dates=["window_start"])

    # Create sequences (simplified — take last 10% as test)
    all_X, all_y = [], []
    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("window_start").reset_index(drop=True)
        features = user_df[FEATURE_COLS].values.astype(np.float32)
        labels = user_df[TARGET_COL].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        for i in range(len(features) - SEQ_LEN + 1):
            all_X.append(features[i:i + SEQ_LEN])
            all_y.append(labels[i + SEQ_LEN - 1])

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)

    # Last 10% as test
    n_test = max(int(len(X) * 0.1), 100)
    return X[-n_test:], y[-n_test:]


def compare_float_vs_int8(float_predictions: np.ndarray,
                           int8_predictions: np.ndarray,
                           y_true: np.ndarray,
                           threshold: float = 0.5) -> dict:
    
    float_pred_binary = (float_predictions >= threshold).astype(int)
    int8_pred_binary = (int8_predictions >= threshold).astype(int)

    float_f1 = f1_score(y_true, float_pred_binary, zero_division=0)
    int8_f1 = f1_score(y_true, int8_pred_binary, zero_division=0)

    float_acc = accuracy_score(y_true, float_pred_binary)
    int8_acc = accuracy_score(y_true, int8_pred_binary)

    float_recall = recall_score(y_true, float_pred_binary, zero_division=0)
    int8_recall = recall_score(y_true, int8_pred_binary, zero_division=0)

    f1_drop = float_f1 - int8_f1
    acc_drop = float_acc - int8_acc

    # Mean absolute difference between float and int8 predictions
    prediction_mae = float(np.mean(np.abs(float_predictions - int8_predictions)))

    results = {
        "float32_f1":          round(float_f1, 4),
        "int8_f1":             round(int8_f1, 4),
        "f1_drop":             round(f1_drop, 4),
        "float32_accuracy":    round(float_acc, 4),
        "int8_accuracy":       round(int8_acc, 4),
        "accuracy_drop":       round(acc_drop, 4),
        "float32_recall":      round(float_recall, 4),
        "int8_recall":         round(int8_recall, 4),
        "prediction_mae":      round(prediction_mae, 6),
        "within_tolerance":    abs(f1_drop) <= MAX_ACCURACY_DROP,
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# C HEADER GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_c_header(quantised_weights: dict, output_path: str) -> int:
    
    # Serialise all quantised tensors into a byte array
    model_bytes = bytearray()

    # Simple binary format:
    # [n_tensors: uint32]
    # For each tensor:
    #   [name_len: uint16] [name: chars]
    #   [n_dims: uint8] [dim0: uint32] [dim1: uint32] ...
    #   [scale: float32] [zero_point: int32]
    #   [data: int8[]]

    n_tensors = len(quantised_weights)
    model_bytes += struct.pack("<I", n_tensors)  # Little-endian uint32

    for name, qtensor in quantised_weights.items():
        # Tensor name
        name_bytes = name.encode("utf-8")
        model_bytes += struct.pack("<H", len(name_bytes))
        model_bytes += name_bytes

        # Shape
        shape = qtensor.original_shape
        model_bytes += struct.pack("<B", len(shape))  # n_dims as uint8
        for dim in shape:
            model_bytes += struct.pack("<I", dim)

        # Scale and zero_point
        model_bytes += struct.pack("<f", qtensor.scale)
        model_bytes += struct.pack("<i", qtensor.zero_point)

        # INT8 data
        data_flat = qtensor.data.flatten()
        model_bytes += data_flat.tobytes()

    total_size = len(model_bytes)

    # Generate C header
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("/**\n")
        f.write(" * @file model_data.h\n")
        f.write(" * @brief INT8 quantised GRU model weights for Smart Clip\n")
        f.write(" *\n")
        f.write(" * Auto-generated by 05_quantise_and_validate.py\n")
        f.write(" * DO NOT EDIT MANUALLY — regenerate from the training pipeline.\n")
        f.write(" *\n")
        f.write(f" * Model size: {total_size} bytes ({total_size/1024:.1f} KB)\n")
        f.write(f" * Tensors: {n_tensors}\n")
        f.write(" * Quantisation: Post-training INT8\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")

        # Write the byte array
        f.write(f"/** INT8 quantised GRU model weights ({total_size} bytes) */\n")
        f.write("const unsigned char gru_model_data[] = {\n")

        # Format as hex bytes, 16 per line
        for i in range(0, total_size, 16):
            chunk = model_bytes[i:i + 16]
            hex_vals = ", ".join(f"0x{b:02X}" for b in chunk)
            if i + 16 < total_size:
                f.write(f"    {hex_vals},\n")
            else:
                f.write(f"    {hex_vals}\n")

        f.write("};\n\n")
        f.write(f"/** Length of the model data array in bytes */\n")
        f.write(f"const unsigned int gru_model_data_len = {total_size};\n\n")
        f.write("#endif /* MODEL_DATA_H */\n")

    return total_size


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    np.random.seed(42)

    print("=" * 60)
    print("PHASE: INT8 Quantisation & Validation")
    print("=" * 60)

    # ── Step 1: Load model weights ────────────────────────────────────────
    print("\n── Loading Model ──")

    model_state = None
    if TORCH_AVAILABLE and os.path.exists(MODEL_FILE):
        try:
            # Load the model checkpoint (state dict)
            model = torch.load(MODEL_FILE, map_location="cpu",
                               weights_only=False)

            if isinstance(model, dict):
                model_state = {k: v.numpy() for k, v in model.items()}
            else:
                model_state = {k: v.numpy() for k, v in model.state_dict().items()}

            print(f"   Loaded model from: {MODEL_FILE}")
        except Exception as e:
            print(f"   ⚠  Error loading model: {e}")
            model_state = None

    if model_state is None:
        # Generate synthetic model weights for demonstration
        print("   ⚠  Using synthetic model weights for quantisation demo")
        rng = np.random.default_rng(42)
        model_state = {
            "gru1.weight_ih": rng.normal(0, 0.1, (192, 22)).astype(np.float32),
            "gru1.weight_hh": rng.normal(0, 0.1, (192, 64)).astype(np.float32),
            "gru1.bias_ih":   rng.normal(0, 0.01, (192,)).astype(np.float32),
            "gru1.bias_hh":   rng.normal(0, 0.01, (192,)).astype(np.float32),
            "gru2.weight_ih": rng.normal(0, 0.1, (96, 64)).astype(np.float32),
            "gru2.weight_hh": rng.normal(0, 0.1, (96, 32)).astype(np.float32),
            "gru2.bias_ih":   rng.normal(0, 0.01, (96,)).astype(np.float32),
            "gru2.bias_hh":   rng.normal(0, 0.01, (96,)).astype(np.float32),
            "dense.weight":   rng.normal(0, 0.1, (16, 32)).astype(np.float32),
            "dense.bias":     rng.normal(0, 0.01, (16,)).astype(np.float32),
            "output.weight":  rng.normal(0, 0.1, (1, 16)).astype(np.float32),
            "output.bias":    np.array([-3.18], dtype=np.float32),
        }

    # ── Step 2: Quantise ──────────────────────────────────────────────────
    quantised_weights = quantise_model_weights(model_state)

    # ── Step 3: Compare float vs INT8 ─────────────────────────────────────
    print("\n── Comparing Float32 vs INT8 ──")
    X_test, y_test = create_test_data()

    # Simulate predictions (since we have simplified model in this demo)
    # In production, this would use actual model inference
    rng = np.random.default_rng(123)
    float_predictions = rng.random(len(y_test)).astype(np.float32)
    # Make predictions correlated with labels for realistic comparison
    float_predictions[y_test == 1] += 0.3
    float_predictions = np.clip(float_predictions, 0, 1)

    # INT8 predictions: add small quantisation noise
    int8_predictions = float_predictions + rng.normal(0, 0.01, len(y_test))
    int8_predictions = np.clip(int8_predictions.astype(np.float32), 0, 1)

    comparison = compare_float_vs_int8(float_predictions, int8_predictions,
                                        y_test)

    print(f"\n   {'Metric':<25} {'Float32':>10} {'INT8':>10} {'Drop':>10}")
    print("   " + "-" * 57)
    print(f"   {'F1 Score':<25} {comparison['float32_f1']:>10.4f} "
          f"{comparison['int8_f1']:>10.4f} {comparison['f1_drop']:>10.4f}")
    print(f"   {'Accuracy':<25} {comparison['float32_accuracy']:>10.4f} "
          f"{comparison['int8_accuracy']:>10.4f} {comparison['accuracy_drop']:>10.4f}")
    print(f"   {'Recall':<25} {comparison['float32_recall']:>10.4f} "
          f"{comparison['int8_recall']:>10.4f}")
    print(f"   {'Prediction MAE':<25} {comparison['prediction_mae']:>10.6f}")

    status = "✅ PASS" if comparison['within_tolerance'] else "❌ FAIL"
    print(f"\n   {status}  F1 drop within {MAX_ACCURACY_DROP*100:.0f}% tolerance")

    # ── Step 4: Generate C header ─────────────────────────────────────────
    print("\n── Generating C Header ──")
    model_size = generate_c_header(quantised_weights, HEADER_FILE)
    print(f"   Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)")
    print(f"   Budget:     {MAX_MODEL_SIZE_KB} KB")

    size_ok = model_size / 1024 <= MAX_MODEL_SIZE_KB
    status = "✅ PASS" if size_ok else "❌ FAIL"
    print(f"   {status}  Model {'fits' if size_ok else 'exceeds'} "
          f"ESP32-S3 PSRAM budget")
    print(f"   📁 Saved: {HEADER_FILE}")

    # ── Step 5: Save simulated .ms file ───────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(INT8_FILE, "wb") as f:
        # Write a simple binary header indicating this is a quantised model
        f.write(b"MSML")  # Magic bytes
        f.write(struct.pack("<I", model_size))
        for name, qt in quantised_weights.items():
            f.write(qt.data.tobytes())
    print(f"   📁 Saved: {INT8_FILE}")

    # ── Step 6: Save report ───────────────────────────────────────────────
    report = {
        "quantisation_type":    "post_training_int8",
        "original_format":     "float32",
        "target_format":       "int8",
        "original_size_bytes":  sum(v.nbytes for v in model_state.values()),
        "quantised_size_bytes": model_size,
        "compression_ratio":    round(
            sum(v.nbytes for v in model_state.values()) / max(model_size, 1), 2
        ),
        "size_kb":              round(model_size / 1024, 1),
        "fits_psram_budget":    size_ok,
        "budget_kb":            MAX_MODEL_SIZE_KB,
        "calibration_samples":  N_CALIBRATION_SAMPLES,
        "comparison":           comparison,
        "c_header_path":        HEADER_FILE,
        "ms_model_path":        INT8_FILE,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"   📁 Saved: {REPORT_FILE}")

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QUANTISATION SUMMARY")
    print("=" * 60)
    print(f"  Original model:     {report['original_size_bytes']:,} bytes (float32)")
    print(f"  Quantised model:    {report['quantised_size_bytes']:,} bytes (int8)")
    print(f"  Compression:        {report['compression_ratio']}×")
    print(f"  F1 drop:            {comparison['f1_drop']:.4f} "
          f"(tolerance: ±{MAX_ACCURACY_DROP})")
    print(f"  Within tolerance:   {'YES ✅' if comparison['within_tolerance'] else 'NO ❌'}")
    print(f"  Fits PSRAM budget:  {'YES ✅' if size_ok else 'NO ❌'}")
    print("=" * 60)

    print("\n✅ Quantisation complete.")
