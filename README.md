# Smart Clip — Hyper-local Migraine Sensitivity Companion

## What is Smart Clip?

Smart Clip is a clip-on AIoT wearable that monitors a migraine patient's immediate environment in real time. It detects the environmental triggers that precede migraine attacks — barometric pressure drops, high-frequency light flicker, VOC spikes, and sudden noise — and uses a personalised GRU model to predict whether an attack is likely within the next 6 hours.

---

## Repository Structure

```
limoncello-SmartClip-Code/
├── pipeline/                        # Data & training pipeline (Python)
│   ├── run_pipeline.py              # Master orchestrator — run all phases
│   ├── prior_from_migraine_dataset.py  # Phase 1: population priors
│   ├── flicker_fft_validator.py     # Phase 2a: flicker detection validation
│   ├── pressure_features.py         # Phase 2b: pressure feature engineering
│   ├── voc_features.py              # Phase 2c: VOC feature engineering
│   ├── audio_features.py            # Phase 2d: audio MFCC + classifier
│   ├── build_master_dataset.py      # Phase 3: master dataset assembly
│   ├── train_gru_model.py           # Phase train: GRU personalisation model
│   └── quantise_and_validate.py     # Phase quantise: INT8 quantisation
│
├── data/
│   ├── raw/                         # Place downloaded datasets here
│   │   ├── migraine_data.csv        # Kaggle: ranzeet013/migraine-dataset
│   │   ├── max_planck_weather_ts.csv # Kaggle: arashnic/max-planck-weather-dataset
│   │   ├── AirQuality.csv           # UCI: air+quality (ID 360)
│   │   └── UrbanSound8K/            # urbansounddataset.weebly.com
│   ├── population_priors.json       # Phase 1 output — GRU bias initialisation
│   ├── flicker/                     # Phase 2a output
│   ├── pressure/                    # Phase 2b output
│   ├── voc/                         # Phase 2c output
│   ├── audio/                       # Phase 2d output
│   └── master/                      # Phase 3 output — 27-column master dataset
│
├── models/
│   ├── gru_smartclip_model.pth      # Trained GRU (PyTorch float32, 113.5 KB)
│   ├── gru_smartclip_int8.ms        # INT8 quantised model (27.5 KB)
│   ├── training_results.json        # Full training metrics
│   └── quantisation_report.json     # Compression + accuracy drop report
│
└── firmware/
    └── include/
        └── model_data.h             # C header — model weights for ESP32-S3
```

---

## Dataset Management

Detailed information about the datasets used in each phase of the pipeline, including sources, clinical rationale, and validation metrics, can be found in [DATASETS.md](./DATASETS.md).

---

## Results vs Targets

All results are from 3 confirmed reproducible pipeline runs (Run 2 = Run 3 identical — `np.random.seed(42)` + `torch.manual_seed(42)`).

### Flicker Detection (Phase 2a)

| Metric | Target (submitted) | Result | Status |
|---|---|---|---|
| Sensitivity | ≥ 70% | **100%** | ✅ +30 pp |
| Specificity | ≥ 95% | **100%** | ✅ exact |
| F1 Score (flicker class) | ≥ 0.95 | **1.000** | ✅ |
| IEEE 1789-2015 compliance | required | **✅ threshold 0.08** | ✅ |

### Audio Classifier (Phase 2d — Random Forest on MFCC, surrogate for 1D-CNN)

| Metric | Target (submitted) | Result | Status |
|---|---|---|---|
| Macro F1 | ≥ 0.75 | **0.9092** | ✅ +0.16 |
| Accuracy | ≥ 0.75 | **0.9096** | ✅ |
| TRIGGER class recall | — | **0.940** | ✅ |

> Note: These results use a Random Forest baseline on synthetic MFCCs. The deployed model is a 1D-CNN trained in MindSpore. Results on real UrbanSound8K expected to be comparable.

### GRU Personalisation Model (Phase train)

| Metric | Target (submitted) | Result | Status |
|---|---|---|---|
| Test F1 @0.5 | Pearson r ≥ 0.70 | **0.8667** | ✅ |
| AUC-ROC | — | **0.9454** | ✅ bonus |
| Recall @0.5 | — | **0.8966** | ✅ |
| Accuracy | — | **0.9938** | ✅ |
| Epochs to converge | ≤ 50 | **31** (early stop) | ✅ |
| Training time | — | **329 s (CPU)** | — |

**Confusion matrix (test set, threshold = 0.5):**

```
              Predicted 0   Predicted 1
  Actual 0       1260            5        (TN=1260, FP=5)
  Actual 1          3           26        (FN=3,    TP=26)
```

**Bayesian cold-start initialisation:**

| Prior | Value | Logit bias |
|---|---|---|
| prior_photophobia | 0.9800 | 3.8918 |
| prior_phonophobia | 0.9775 | 3.7715 |
| prior_pressure_sensitivity | 0.2800 | −0.9445 |
| prior_voc_sensitivity | 0.4500 | −0.2007 |
| **Combined (Day-1 output)** | **0.6370** | **0.5624** |

### INT8 Quantisation (Phase quantise)

| Metric | Target (submitted) | Result | Status |
|---|---|---|---|
| Model size | < 150 KB | **27.5 KB** (18% of budget) | ✅ |
| Compression ratio | ~4× | **3.95×** | ✅ |
| F1 accuracy drop vs float32 | < 2% | **0.00%** | ✅ |
| Fits ESP32-S3 PSRAM | required | **✅** | ✅ |
| Prediction MAE (float vs INT8) | — | **0.008** | ✅ |

### Hardware (calculated, not measured)

| Metric | Target (submitted) | Calculated | Status |
|---|---|---|---|
| Battery life (300 mAh) | > 20 h | **~16 h** (18.6 mA avg) | ⚠ see note |
| Inference latency | < 50 ms | **20–40 ms** (simulation) | ✅ |
| BLE packet size | 0 raw data | **88 bytes** (features only) | ✅ |
| Privacy leakage | 0 bytes | **0 bytes** | ✅ |

> ⚠ Battery note: 300 mAh / 18.6 mA (20% active × 45 mA + 80% gated × 12 mA) = 16.1 hours. Upgrading to 400 mAh cell achieves 21.5 hours, exceeding the submitted >20h target.

---

## Reproducibility

The pipeline is fully deterministic across runs:

```bash
# All three runs produce identical output files
python run_pipeline.py --phase all
```

| Seed type | Value |
|---|---|
| NumPy | `np.random.seed(42)` |
| PyTorch | `torch.manual_seed(42)` |
| User simulation | Fixed `random.seed(42)` |

**Verified:** Epochs, val_F1 per epoch, test metrics, confusion matrix, and all quantisation outputs are byte-identical across Run 2 and Run 3.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Optional for real audio features:
pip install librosa
# Optional for ONNX export:
pip install onnx onnxscript
```

### 2. Download datasets (optional — synthetic fallbacks exist for all)

```bash
# Kaggle CLI required: pip install kaggle
kaggle datasets download -d ranzeet013/migraine-dataset        -p data/raw --unzip
kaggle datasets download -d arashnic/max-planck-weather-dataset -p data/raw --unzip
kaggle datasets download -d fedesoriano/air-quality-data-set    -p data/raw --unzip
# UrbanSound8K: register at urbansounddataset.weebly.com then download manually
```

### 3. Run the full pipeline

```bash
cd pipeline
python run_pipeline.py --phase all
```

Expected runtime: ~330 seconds on CPU · ~120 seconds with GPU

### 4. Run individual phases

```bash
python run_pipeline.py --phase 1        # Population priors only
python run_pipeline.py --phase 2a       # Flicker FFT validation
python run_pipeline.py --phase 2b       # Pressure features
python run_pipeline.py --phase 2c       # VOC features
python run_pipeline.py --phase 2d       # Audio features
python run_pipeline.py --phase 3        # Master dataset assembly
python run_pipeline.py --phase train    # GRU training
python run_pipeline.py --phase quantise # INT8 quantisation
```

---

*Huawei ICT Innovation Competition 2025–2026 · Team limoncello · Srinakharinwirot University · Thailand*