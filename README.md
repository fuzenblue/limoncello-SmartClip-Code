# Smart Clip — Hyper-local Migraine Sensitivity Companion

A wearable clip-on IoT device that monitors a migraine patient's immediate environment in real time, detecting environmental triggers and predicting migraine attacks using personalised AI.

## Project Structure

```
SmartClip/
├── data/                          # Generated datasets and features
│   ├── raw/                       # Place raw datasets here
│   ├── flicker/                   # Flicker detection outputs
│   ├── pressure/                  # Pressure feature outputs
│   ├── voc/                       # VOC feature outputs
│   ├── audio/                     # Audio feature outputs
│   └── master/                    # Final assembled dataset
├── models/                        # Trained model outputs
├── pipeline/                      # Python data pipeline scripts
│   ├── prior_from_migraine_dataset.py
│   ├── flicker_fft_validator.py
│   ├── pressure_features.py
│   ├── voc_features.py
│   ├── audio_features.py
│   ├── build_master_dataset.py
│   ├── train_gru_model.py
│   ├── quantise_and_validate.py
│   └── run_pipeline.py
└── firmware/                      # ESP32-S3 firmware (ESP-IDF)
    ├── CMakeLists.txt
    ├── main/
    │   ├── CMakeLists.txt
    │   ├── main.c
    │   ├── sensor_grid.c / .h
    │   ├── flicker_fft.c / .h
    │   ├── pressure_voc_calc.c / .h
    │   ├── audio_features.c / .h
    │   ├── feature_vector.c / .h
    │   ├── ble_transmit.c / .h
    │   └── motion_gate.c / .h
    └── components/
        └── mindspore_micro/
```

## Quick Start

### Python Pipeline
```bash
cd pipeline
pip install numpy pandas scipy scikit-learn librosa matplotlib
```

### ESP32-S3 Firmware
```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

## Architecture

- **Layer 1 — Device**: ESP32-S3 with 4 sensors, FFT flicker detection, MFCC audio features
- **Layer 2 — Mobile**: Companion app (not in this repo) runs GRU model via MindSpore Lite
- **Layer 3 — Cloud**: Huawei ModelArts for periodic model retraining
