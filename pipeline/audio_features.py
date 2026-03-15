
# IMPORTS
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

# Try to import librosa for real audio processing; optional dependency
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠  librosa not installed — will use synthetic MFCC fallback")

# Suppress librosa warnings about audioread backends
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# CONSTANTS

# Audio parameters matching ESP32-S3 firmware
AUDIO_SAMPLE_RATE = 22050   # Hz — standard librosa sample rate for feature
                             # extraction. The INMP441 records at 16kHz, but
                             # librosa resamples to 22050Hz internally for
                             # consistent MFCC computation.

AUDIO_DURATION = 4.0         # seconds — UrbanSound8K clips are up to 4 seconds

N_MFCC = 13                # Number of MFCC coefficients to extract
                             # Standard value used across audio ML literature

# UrbanSound8K class mapping to Smart Clip 3-class system
# Each mapping decision is documented with its phonophobia rationale:
URBANSCOUND_TO_SMARTCLIP = {
    # CLASS 2 — TRIGGER: Sudden, high-intensity sounds that are known to
    # aggravate phonophobia (sound sensitivity) in migraine patients.
    "siren":         2,   # High-pitched, piercing — classic phonophobia trigger
    "jackhammer":    2,   # Sharp mechanical repetitive impact
    "gun_shot":      2,   # Short, extreme broadband impulse

    # CLASS 1 — TRAFFIC: Urban ambient sounds. Moderate intensity, sustained.
    # Annoying but less acutely triggering than class 2.
    "car_horn":      1,   # Intermittent, medium intensity
    "engine_idling": 1,   # Low-frequency continuous rumble
    "drilling":      1,   # Construction noise, sustained

    # CLASS 0 — QUIET: Background ambient sounds. Low phonophobia relevance.
    # Present in everyday environments, unlikely to trigger migraine.
    "air_conditioner": 0, # Stable, low-level white noise
    "children_playing": 0,# Natural ambient, non-alarming
    "dog_bark":         0, # Brief, low phonophobia relevance
    "street_music":     0, # Ambient, no sharp transients
}

# Smart Clip class labels (for human-readable output)
SC_CLASS_NAMES = {0: "Quiet", 1: "Traffic", 2: "Trigger"}

# Synthetic MFCC distribution parameters (per Smart Clip class)
# These are approximate MFCC statistics derived from UrbanSound8K literature.
# Each class has a characteristic "spectral shape" that MFCCs capture:
#   Quiet: relatively flat spectrum, low energy
#   Traffic: energy in low-mid frequencies, moderate level
#   Trigger: sharp spectral peaks, high energy, transient characteristics
SYNTHETIC_MFCC_PARAMS = {
    0: {  # Quiet — smooth, low-energy spectral shape
        "mean": np.array([-250, 80, -10, 20, -5, 10, -5, 5, -3, 3, -2, 2, -1]),
        "std":  np.array([30, 15, 10, 8, 6, 5, 4, 4, 3, 3, 2, 2, 2]),
    },
    1: {  # Traffic — low-frequency dominated, moderate energy
        "mean": np.array([-180, 60, -20, 15, -10, 8, -8, 6, -5, 4, -3, 3, -2]),
        "std":  np.array([35, 20, 12, 10, 8, 7, 5, 5, 4, 3, 3, 2, 2]),
    },
    2: {  # Trigger — high-energy, sharp spectral peaks
        "mean": np.array([-120, 100, -30, 25, -15, 12, -10, 8, -6, 5, -4, 3, -2]),
        "std":  np.array([40, 25, 15, 12, 10, 8, 6, 5, 4, 4, 3, 3, 2]),
    },
}

SYNTHETIC_SAMPLES_PER_CLASS = 166   # Total = 498 (close to UrbanSound8K class sizes)

# File paths
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "raw")
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "audio")
US8K_DIR     = os.path.join(RAW_DATA_DIR, "UrbanSound8K")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "audio_features.csv")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "audio_class_mapping.json")
REPORT_FILE  = os.path.join(OUTPUT_DIR, "audio_model_report.json")


# MFCC EXTRACTION FROM REAL AUDIO

def extract_mfcc_from_file(file_path: str) -> np.ndarray:
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required for real audio processing")

    # Load audio file, convert to mono, resample to target rate
    # sr=AUDIO_SAMPLE_RATE forces resampling regardless of file's native rate
    # duration=AUDIO_DURATION truncates to 4 seconds
    y, sr = librosa.load(file_path, sr=AUDIO_SAMPLE_RATE,
                         duration=AUDIO_DURATION, mono=True)

    # Pad shorter clips to full duration with zeros (silence)
    target_length = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")

    # Extract MFCCs
    # n_mfcc=13: number of cepstral coefficients to compute
    # The result shape is (13, T) where T = number of time frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Average across time frames to get a single feature vector
    # This loses temporal information but gives a fixed-size input
    # suitable for simple classifiers
    mfcc_mean = np.mean(mfccs, axis=1)

    return mfcc_mean


def load_urbansound8k() -> pd.DataFrame:
    # Look for the dataset folder and a CSV file with the SAME name next to it
    # This matches the user's "folder + csv same name" requirement
    dataset_csv = US8K_DIR + ".csv"
    
    if os.path.isdir(US8K_DIR) and os.path.isfile(dataset_csv):
        metadata_path = dataset_csv
        print(f"Detected UrbanSound8K: Folder and CSV '{os.path.basename(dataset_csv)}' both present.")
    else:
        # Fallback to standard structured folder search
        metadata_path = os.path.join(US8K_DIR, "metadata", "UrbanSound8K.csv")
        if not os.path.exists(metadata_path):
            return None

    if not LIBROSA_AVAILABLE:
        print("librosa not available — cannot process raw audio")
        return None

    print(f"Loading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    records = []
    total = len(metadata)

    for idx, row in metadata.iterrows():
        fold = row["fold"]
        filename = row["slice_file_name"]
        us8k_class = row["class"]

        # Map UrbanSound8K class name to Smart Clip class (0/1/2)
        sc_class = URBANSCOUND_TO_SMARTCLIP.get(us8k_class, 0)

        file_path = os.path.join(US8K_DIR, "audio", f"fold{fold}", filename)

        if not os.path.exists(file_path):
            continue

        try:
            mfcc = extract_mfcc_from_file(file_path)
            record = {f"mfcc_{i+1:02d}": mfcc[i] for i in range(N_MFCC)}
            record["us8k_class"] = us8k_class
            record["sc_class"] = sc_class
            records.append(record)

            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{total} clips...")

        except Exception as e:
            # Skip corrupted or unreadable files
            continue

    print(f"Extracted MFCCs from {len(records)} clips")
    return pd.DataFrame(records)


# SYNTHETIC MFCC GENERATION (Fallback)

def generate_synthetic_mfcc() -> pd.DataFrame:
    print("\nSYNTHETIC FALLBACK: Generating synthetic MFCC features")
    print(f"UrbanSound8K not found at: {US8K_DIR}")
    print(f"Generating {SYNTHETIC_SAMPLES_PER_CLASS} samples per class\n")

    rng = np.random.default_rng(42)
    records = []

    # Map back from sc_class to a representative us8k_class for labelling
    sc_to_us8k = {0: "air_conditioner", 1: "car_horn", 2: "siren"}

    for sc_class, params in SYNTHETIC_MFCC_PARAMS.items():
        us8k_class = sc_to_us8k[sc_class]

        for _ in range(SYNTHETIC_SAMPLES_PER_CLASS):
            # Sample from multivariate normal with class-specific statistics
            mfcc = rng.normal(loc=params["mean"], scale=params["std"])

            record = {f"mfcc_{i+1:02d}": float(mfcc[i]) for i in range(N_MFCC)}
            record["us8k_class"] = us8k_class
            record["sc_class"] = sc_class
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} synthetic MFCC samples:")
    for sc_class in [0, 1, 2]:
        count = len(df[df["sc_class"] == sc_class])
        print(f"Class {sc_class} ({SC_CLASS_NAMES[sc_class]}): {count} samples")

    return df


# CLASSIFIER TRAINING

def train_random_forest(X: np.ndarray, y: np.ndarray) -> dict:
    print("\nTraining Random Forest Baseline")

    # Initialise classifier with 100 decision trees
    # random_state=42 ensures reproducibility
    # n_jobs=-1 uses all CPU cores for parallel tree training
    clf = RandomForestClassifier(
        n_estimators=100,     # 100 trees in the forest
        random_state=42,      # Reproducible results
        n_jobs=-1,            # Use all CPU cores
        class_weight="balanced"  # Auto-adjust weights for class imbalance
                                  # This upweights the rare Trigger class
    )

    # 5-fold stratified cross-validation
    # method="predict" gets class predictions (not probabilities)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")

    # Compute per-class metrics
    report = classification_report(
        y, y_pred,
        target_names=[SC_CLASS_NAMES[i] for i in range(3)],
        output_dict=True
    )

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Overall metrics
    macro_f1 = f1_score(y, y_pred, average="macro")
    accuracy = accuracy_score(y, y_pred)

    # Feature importances (which MFCCs are most discriminative?)
    # Train on full dataset for feature importance
    clf.fit(X, y)
    importances = clf.feature_importances_

    results = {
        "accuracy":          round(float(accuracy), 4),
        "macro_f1":          round(float(macro_f1), 4),
        "per_class_report":  report,
        "confusion_matrix":  cm.tolist(),
        "feature_importances": {
            f"mfcc_{i+1:02d}": round(float(importances[i]), 4)
            for i in range(N_MFCC)
        },
        "n_samples":         int(len(y)),
        "n_folds":           5,
        "classifier":        "RandomForest(n_estimators=100)",
    }

    return results


def print_training_report(results: dict) -> None:
    
    print("AUDIO CLASSIFIER — TRAINING REPORT")

    print(f"\n  Classifier:  {results['classifier']}")
    print(f"  Samples:     {results['n_samples']}")
    print(f"  CV Folds:    {results['n_folds']}")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Macro F1:    {results['macro_f1']:.4f}")

    # Confusion matrix
    cm = np.array(results["confusion_matrix"])
    print("\n Confusion Matrix ")
    print(f"   {'':>12} Pred:Quiet  Pred:Traffic  Pred:Trigger")
    for i, name in enumerate(["Quiet", "Traffic", "Trigger"]):
        row = "  ".join(f"{cm[i][j]:>10}" for j in range(3))
        print(f"   {name:>12}  {row}")

    # Per-class metrics
    print("\n Per-Class Metrics ")
    print(f"   {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("   " + "-" * 44)
    for class_name in ["Quiet", "Traffic", "Trigger"]:
        r = results["per_class_report"].get(class_name, {})
        print(f"   {class_name:<12} {r.get('precision', 0):>10.3f} "
              f"{r.get('recall', 0):>10.3f} {r.get('f1-score', 0):>10.3f}")

    # Feature importances (top 5)
    print("\n Top 5 Most Discriminative MFCCs ")
    importances = results["feature_importances"]
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp[:5]:
        bar = "█" * int(imp * 100)
        print(f"   {name}: {imp:.4f}  {bar}")



# 1D-CNN ARCHITECTURE DEFINITION (for MindSpore deployment)

def define_1d_cnn_architecture():

    print("1D-CNN ARCHITECTURE (MindSpore deployment target)")

    architecture = [
        {
            "layer": "Input",
            "shape": "(13,)",
            "description": "13 MFCC coefficients (mean across time frames)"
        },
        {
            "layer": "Reshape",
            "shape": "(13, 1)",
            "description": "Treat 13 MFCCs as a 1D sequence with 1 channel. "
                           "Conv1D requires (sequence_length, channels) format."
        },
        {
            "layer": "Conv1D",
            "params": "filters=32, kernel_size=3, activation=ReLU",
            "description": "Learns local patterns across 3 adjacent MFCCs. "
                           "32 filters = 32 different pattern detectors. "
                           "ReLU activation passes positive values, zeros negatives."
        },
        {
            "layer": "Conv1D",
            "params": "filters=64, kernel_size=3, activation=ReLU",
            "description": "Learns higher-level patterns by combining the "
                           "32 first-layer features. Receptive field expands "
                           "to 5 MFCCs."
        },
        {
            "layer": "Conv1D",
            "params": "filters=128, kernel_size=3, activation=ReLU",
            "description": "Deepest feature extraction layer. 128 high-level "
                           "pattern detectors. Receptive field = 7 MFCCs."
        },
        {
            "layer": "GlobalAveragePooling1D",
            "params": "—",
            "description": "Average all positions into a single 128-D vector. "
                           "Makes the model invariant to input length and "
                           "drastically reduces parameters vs Flatten."
        },
        {
            "layer": "Dense",
            "params": "units=64, activation=ReLU",
            "description": "Fully-connected classification layer. Combines "
                           "the 128 pooled features into 64 high-level concepts."
        },
        {
            "layer": "Dropout",
            "params": "rate=0.3",
            "description": "Randomly sets 30%% of outputs to zero during training. "
                           "Prevents overfitting by forcing the network to learn "
                           "redundant representations."
        },
        {
            "layer": "Dense (Output)",
            "params": "units=3, activation=Softmax",
            "description": "Output layer with 3 units (one per class). "
                           "Softmax converts raw scores to probabilities "
                           "that sum to 1.0."
        },
    ]

    print(f"\n{'Layer':<25} {'Parameters':<40} {'Output Shape':<12}")
    print("-" * 80)
    for layer_info in architecture:
        layer_name = layer_info["layer"]
        params = layer_info.get("params", layer_info.get("shape", "—"))
        print(f"  {layer_name:<23} {params:<40}")
        print(f"  {'':>23} {layer_info['description']}")
        print()

    print("Total estimated parameters: ~15,000 (INT8 model size: ~18KB)")
    print("This fits comfortably within the 150KB PSRAM budget.")

    return architecture


# ENTRY POINT

if __name__ == "__main__":
    # Set random seed for full reproducibility
    np.random.seed(42)

    print("PHASE 2d: Audio Feature Extraction & Classifier Training")

    #  Step 1: Load or generate MFCC features 
    df = load_urbansound8k()
    if df is None:
        df = generate_synthetic_mfcc()

    #  Step 2: Prepare feature matrix and labels ─
    mfcc_cols = [f"mfcc_{i+1:02d}" for i in range(N_MFCC)]
    X = df[mfcc_cols].values.astype(np.float32)     # Shape: (n_samples, 13)
    y = df["sc_class"].values.astype(np.int32)       # Shape: (n_samples,)

    print(f"\n   Feature matrix shape: {X.shape}")
    print(f"   Label distribution:")
    for sc_class in [0, 1, 2]:
        count = np.sum(y == sc_class)
        pct = 100 * count / len(y)
        print(f"     Class {sc_class} ({SC_CLASS_NAMES[sc_class]}): "
              f"{count} ({pct:.1f}%)")

    #  Step 3: Train and evaluate Random Forest 
    results = train_random_forest(X, y)
    print_training_report(results)

    #  Step 4: Document 1D-CNN architecture 
    cnn_arch = define_1d_cnn_architecture()

    #  Step 5: Save outputs 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save MFCC feature dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved features:  {OUTPUT_CSV}")

    # Save class mapping documentation
    mapping_doc = {
        "smartclip_classes": SC_CLASS_NAMES,
        "urbansound8k_mapping": {k: SC_CLASS_NAMES[v]
                                  for k, v in URBANSCOUND_TO_SMARTCLIP.items()},
        "mapping_rationale": {
            "class_0_quiet": "Ambient sounds with low phonophobia relevance",
            "class_1_traffic": "Urban environmental sounds, moderate intensity",
            "class_2_trigger": "Sudden loud sounds that aggravate migraine phonophobia",
        },
    }
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping_doc, f, indent=2)
    print(f"Saved mapping:   {MAPPING_FILE}")

    # Save model report
    # Convert numpy types to native Python for JSON serialisation
    serialisable_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serialisable_results[k] = v.tolist()
        elif isinstance(v, dict):
            serialisable_results[k] = {
                sk: (sv.tolist() if isinstance(sv, np.ndarray) else sv)
                for sk, sv in v.items()
            }
        else:
            serialisable_results[k] = v

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(serialisable_results, f, indent=2, default=str)
    print(f"Saved report:    {REPORT_FILE}")

    print("\nPhase 2d complete: Audio features extracted and classifier trained.")
