

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os                             # File system operations
import json                           # JSON serialisation
import numpy as np                    # Numerical computation
import pandas as pd                   # Tabular data handling
from scipy.fft import fft, fftfreq    # Fast Fourier Transform implementation
from sklearn.metrics import (         # ML evaluation metrics
    confusion_matrix,
    classification_report,
    f1_score
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS — These match the ESP32-S3 firmware exactly
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 800          # Hz — ADC sampling rate for the photodiode
                            # WHY 800? Nyquist theorem requires ≥ 2× the max
                            # frequency we want to detect.  200Hz × 2 = 400Hz.
                            # 800Hz gives a comfortable 2× safety margin and is
                            # a clean multiple of 100/200Hz flicker frequencies.

WINDOW_SIZE = 512          # FFT window length in samples (must be power of 2)
                            # WHY 512? The FFT "butterfly" algorithm (Cooley-Tukey)
                            # is only efficient for power-of-2 sizes.  512 samples
                            # at 800Hz = 0.64 seconds — enough for several flicker
                            # cycles at 100Hz while keeping computation fast.

FLICKER_BAND_LOW = 90      # Hz — lower edge of the detection band
                            # Slightly below 100Hz to capture mains flicker even
                            # if the frequency drifts slightly (mains ≠ exactly 50Hz)

FLICKER_BAND_HIGH = 210    # Hz — upper edge of the detection band
                            # Covers 100Hz (50Hz mains), 120Hz (60Hz mains),
                            # 150Hz (common PWM), and 200Hz (high PWM)

FLICKER_THRESHOLD = 0.08   # IEEE 1789-2015 Flicker Index limit
                            # Above this value, the flicker is considered
                            # "potentially harmful" for photosensitive individuals.
                            # This is NOT a self-defined threshold — it comes from
                            # a published IEEE engineering standard.

SIGNAL_DURATION = 1.0       # seconds — duration of each test signal

NOISE_STD = 0.02            # Standard deviation of Gaussian ADC noise.
                            # Real ADC readings have electrical noise from the
                            # circuit — this simulates that effect.

SAMPLES_PER_CLASS = 50     # Number of synthetic signals to generate per class

# Output paths
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "data", "flicker")
DATASET_FILE = os.path.join(OUTPUT_DIR, "flicker_dataset.csv")
REPORT_FILE  = os.path.join(OUTPUT_DIR, "flicker_validation_report.json")


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def mains_led_signal(freq_hz: float,
                     duration_s: float = SIGNAL_DURATION,
                     sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    
    # Create time vector: n evenly spaced points from 0 to duration_s
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # Full-wave rectified sine: |sin(2πft)| is always positive, oscillates at freq_hz
    # The 0.5 + 0.5× scaling puts the signal in range [0.5, 1.0]
    signal = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * freq_hz * t))

    # Add Gaussian noise to simulate ADC measurement noise
    noise = np.random.normal(0, NOISE_STD, n_samples)
    signal += noise

    return signal


def pwm_led_signal(freq_hz: float,
                   duty_cycle: float = 0.5,
                   duration_s: float = SIGNAL_DURATION,
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # Period of one PWM cycle in seconds
    period = 1.0 / freq_hz

    # Fractional position within each PWM cycle [0.0, 1.0)
    # The modulo operation wraps the time around the period
    phase = (t % period) / period

    # Square wave: ON (=1.0) when phase < duty_cycle, OFF (=0.2) otherwise
    # 0.2 baseline (not 0.0) simulates residual light from phosphor persistence
    signal = np.where(phase < duty_cycle, 1.0, 0.2)

    # Add ADC noise
    noise = np.random.normal(0, NOISE_STD, n_samples)
    signal += noise

    return signal


def dc_light_signal(duration_s: float = SIGNAL_DURATION,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    
    n_samples = int(duration_s * sample_rate)

    # Constant brightness at ~75% of ADC range
    signal = np.full(n_samples, 0.75)

    # Add small noise to simulate real ADC readings
    noise = np.random.normal(0, NOISE_STD, n_samples)
    signal += noise

    return signal


def sunlight_signal(duration_s: float = SIGNAL_DURATION,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # Slow sinusoidal drift at 0.3Hz — simulates passing clouds
    # This is well below the 90Hz lower bound of our flicker detection band
    slow_drift = 0.1 * np.sin(2 * np.pi * 0.3 * t)

    # Base brightness + drift
    signal = 0.70 + slow_drift

    # Add noise — slightly more than indoor due to atmospheric turbulence
    noise = np.random.normal(0, NOISE_STD * 1.5, n_samples)
    signal += noise

    return signal


# ══════════════════════════════════════════════════════════════════════════════
# FFT PIPELINE — Exact mirror of the ESP32-S3 C implementation
# ══════════════════════════════════════════════════════════════════════════════

def compute_flicker_index(signal: np.ndarray,
                          sample_rate: int = SAMPLE_RATE,
                          window_size: int = WINDOW_SIZE) -> tuple:
    
    # ── Step 1: Extract FFT window ────────────────────────────────────────
    # Use only the first window_size samples.  On the ESP32, this is the
    # last window_size samples from the circular ADC buffer.
    segment = signal[:window_size].copy()

    # ── Step 2: Apply Hamming window ──────────────────────────────────────
    # WITHOUT the window function, the FFT treats the signal as if it repeats
    # infinitely.  The abrupt start/end of our finite window creates a
    # discontinuity — the FFT sees this as high-frequency content that
    # doesn't actually exist (= "spectral leakage").
    #
    # The Hamming window smoothly tapers the signal to zero at both ends:
    #   w[n] = 0.54 - 0.46 × cos(2π × n / (N - 1))
    #
    # This eliminates the discontinuity and gives us a clean, accurate
    # frequency spectrum.  The tradeoff is slightly wider frequency peaks
    # (worse frequency resolution), which is acceptable for our application.
    hamming_window = np.hamming(window_size)
    windowed = segment * hamming_window

    # ── Step 3: Compute FFT ───────────────────────────────────────────────
    # The FFT decomposes the time-domain signal into its constituent
    # frequencies.  For a real-valued input of length N, the output contains
    # N complex values, but only the first N/2 are unique (the second half
    # is the complex conjugate of the first half — a property of real signals).
    #
    # Each element spectrum[k] represents the amplitude at frequency:
    #   freq_k = k × (sample_rate / window_size)
    #
    # spectrum[0] = the DC component (average signal level)
    # spectrum[1] = amplitude at freq = sample_rate / window_size
    # spectrum[N/2-1] = amplitude at freq = sample_rate / 2 (Nyquist frequency)
    raw_fft = fft(windowed)

    # Take only the first half (positive frequencies) and compute magnitude
    # Magnitude = sqrt(real² + imaginary²) — the "strength" at each frequency
    spectrum = np.abs(raw_fft[:window_size // 2])

    # ── Step 4: Compute frequency axis ────────────────────────────────────
    # fftfreq returns the frequency corresponding to each FFT bin
    # freq_resolution = sample_rate / window_size = 800 / 512 = 1.5625 Hz
    # This means each bin covers 1.5625 Hz of bandwidth
    freqs = fftfreq(window_size, d=1.0 / sample_rate)[:window_size // 2]

    # ── Step 5: Compute power spectrum ────────────────────────────────────
    # Power = magnitude² — proportional to the energy at each frequency
    # Using power instead of magnitude emphasises strong peaks and
    # suppresses noise, making flicker detection more robust
    power = spectrum ** 2

    # ── Step 6: Identify the flicker band ─────────────────────────────────
    # Find frequency bins within the 90–210 Hz detection band
    band_mask = (freqs >= FLICKER_BAND_LOW) & (freqs <= FLICKER_BAND_HIGH)

    # ── Step 7: Compute Flicker Index ─────────────────────────────────────
    # Flicker Index = (power in flicker band) / (total power, excluding DC)
    #
    # We exclude DC (index 0) because the DC component represents the average
    # light level, not any oscillation.  A bright constant light would have
    # enormous DC power but zero flicker — including DC would make the
    # flicker index meaninglessly small.
    band_power = np.sum(power[band_mask])
    total_power = np.sum(power[1:])  # Skip DC component at index 0

    # Avoid division by zero (total silence/no light)
    if total_power < 1e-12:
        return (0.0, 0.0)

    flicker_index = float(band_power / total_power)

    # ── Step 8: Find dominant frequency ───────────────────────────────────
    # The dominant frequency is the frequency with the highest power
    # within the flicker band.  This tells us what TYPE of flicker source
    # is present (100Hz = 50Hz mains, 120Hz = 60Hz mains, etc.)
    if np.sum(band_mask) > 0 and np.max(power[band_mask]) > 0:
        # Find the index within the band that has maximum power
        band_indices = np.where(band_mask)[0]
        band_powers = power[band_indices]
        dominant_bin = band_indices[np.argmax(band_powers)]
        dominant_freq = float(freqs[dominant_bin])
    else:
        dominant_freq = 0.0

    return (flicker_index, dominant_freq)


def detect_flicker(flicker_index: float, dominant_freq: float) -> bool:
    
    is_above_threshold = flicker_index > FLICKER_THRESHOLD
    is_in_band = FLICKER_BAND_LOW <= dominant_freq <= FLICKER_BAND_HIGH
    return is_above_threshold and is_in_band


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION AND VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_validation_dataset() -> pd.DataFrame:
    
    records = []

    # Define signal types with their generator functions and expected labels
    # label=1 means the signal SHOULD trigger a flicker alert
    # label=0 means the signal should NOT trigger a flicker alert
    signal_specs = [
        # (signal_type_name, generator_function, expected_label)
        ("thai_mains_led_100hz",  lambda: mains_led_signal(100.0),     1),
        ("us_mains_led_120hz",    lambda: mains_led_signal(120.0),     1),
        ("pwm_led_150hz",         lambda: pwm_led_signal(150.0),       1),
        ("pwm_led_200hz",         lambda: pwm_led_signal(200.0, 0.4),  1),
        ("battery_led_dc",        lambda: dc_light_signal(),           0),
        ("sunlight",              lambda: sunlight_signal(),           0),
        ("phone_flashlight",      lambda: dc_light_signal(),           0),
    ]

    print("\n── Generating Synthetic Validation Dataset ──")
    print(f"   {len(signal_specs)} signal types × {SAMPLES_PER_CLASS} samples each\n")

    for sig_name, generator, label in signal_specs:
        detections = 0
        for i in range(SAMPLES_PER_CLASS):
            # Generate the signal with its inherent randomness (noise)
            signal = generator()

            # Run the FFT flicker detection pipeline
            fidx, ffreq = compute_flicker_index(signal)
            detected = detect_flicker(fidx, ffreq)

            if detected:
                detections += 1

            records.append({
                "signal_type":      sig_name,
                "label":            label,          # Ground truth
                "flicker_index":    round(fidx, 6),
                "dominant_freq_hz": round(ffreq, 2),
                "detected":         int(detected),  # Prediction
                "sample_id":        i,
            })

        label_str = "FLICKER" if label == 1 else "CLEAN"
        print(f"   {sig_name:<30}  [{label_str}]  "
              f"Detected: {detections}/{SAMPLES_PER_CLASS}")

    return pd.DataFrame(records)


def compute_validation_metrics(df: pd.DataFrame) -> dict:
    
    y_true = df["label"].values
    y_pred = df["detected"].values

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)

    # Extract individual counts
    tn, fp, fn, tp = cm.ravel()

    # Compute metrics with safe division (avoid divide-by-zero)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0    # = Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = f1_score(y_true, y_pred)
    accuracy    = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    metrics = {
        "total_samples":     int(len(df)),
        "true_positives":    int(tp),
        "true_negatives":    int(tn),
        "false_positives":   int(fp),
        "false_negatives":   int(fn),
        "sensitivity":       round(sensitivity, 4),  # = recall
        "specificity":       round(specificity, 4),
        "precision":         round(precision, 4),
        "f1_score":          round(f1, 4),
        "accuracy":          round(accuracy, 4),
        "flicker_threshold": FLICKER_THRESHOLD,
        "band_low_hz":       FLICKER_BAND_LOW,
        "band_high_hz":      FLICKER_BAND_HIGH,
        "sample_rate_hz":    SAMPLE_RATE,
        "window_size":       WINDOW_SIZE,
        "freq_resolution_hz": round(SAMPLE_RATE / WINDOW_SIZE, 4),
    }

    return metrics


def print_validation_report(metrics: dict, df: pd.DataFrame) -> None:
    
    print("\n" + "=" * 60)
    print("FLICKER FFT VALIDATOR — VALIDATION REPORT")
    print("Standard: IEEE 1789-2015")
    print("=" * 60)

    # Confusion matrix
    print("\n── Confusion Matrix ──")
    print(f"   {'':>15} Predicted:0  Predicted:1")
    print(f"   {'Actual:0':>15}    {metrics['true_negatives']:>5}       "
          f"{metrics['false_positives']:>5}")
    print(f"   {'Actual:1':>15}    {metrics['false_negatives']:>5}       "
          f"{metrics['true_positives']:>5}")

    # Performance metrics
    print("\n── Performance Metrics ──")
    print(f"   {'Metric':<20} {'Value':>10}")
    print("   " + "-" * 32)
    metric_display = [
        ("Sensitivity",   metrics["sensitivity"]),
        ("Specificity",   metrics["specificity"]),
        ("Precision",     metrics["precision"]),
        ("F1 Score",      metrics["f1_score"]),
        ("Accuracy",      metrics["accuracy"]),
    ]
    for name, val in metric_display:
        bar = "█" * int(val * 20)
        print(f"   {name:<20} {val:>8.4f}  {bar}")

    # Configuration
    print("\n── DSP Configuration ──")
    print(f"   Sample Rate:         {metrics['sample_rate_hz']} Hz")
    print(f"   FFT Window Size:     {metrics['window_size']} samples")
    print(f"   Frequency Resolution:{metrics['freq_resolution_hz']} Hz/bin")
    print(f"   Detection Band:      {metrics['band_low_hz']}–{metrics['band_high_hz']} Hz")
    print(f"   Flicker Threshold:   {metrics['flicker_threshold']} "
          f"(IEEE 1789-2015)")

    # Per signal type breakdown
    print("\n── Per-Signal-Type Results ──")
    print(f"   {'Signal Type':<30} {'Label':>5} {'Mean FI':>8} "
          f"{'Mean Freq':>10} {'Det Rate':>8}")
    print("   " + "-" * 65)
    for sig_type in df["signal_type"].unique():
        subset = df[df["signal_type"] == sig_type]
        print(f"   {sig_type:<30} {subset['label'].iloc[0]:>5} "
              f"{subset['flicker_index'].mean():>8.4f} "
              f"{subset['dominant_freq_hz'].mean():>10.2f} "
              f"{subset['detected'].mean():>8.1%}")

    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Set random seed for reproducibility
    np.random.seed(42)

    print("=" * 60)
    print("PHASE 2a: Flicker FFT Validation")
    print("=" * 60)

    # ── Step 1: Generate validation dataset ───────────────────────────────
    df = generate_validation_dataset()

    # ── Step 2: Compute validation metrics ────────────────────────────────
    metrics = compute_validation_metrics(df)

    # ── Step 3: Print formatted report ────────────────────────────────────
    print_validation_report(metrics, df)

    # ── Step 4: Save outputs ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save labelled dataset
    df.to_csv(DATASET_FILE, index=False)
    print(f"\n📁 Saved dataset: {DATASET_FILE}")

    # Save validation report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"📁 Saved report:  {REPORT_FILE}")

    # ── Step 5: Pass/Fail verdict ─────────────────────────────────────────
    print("\n── QUALITY GATE ──")
    checks = [
        ("Sensitivity ≥ 0.95", metrics["sensitivity"] >= 0.95),
        ("Specificity ≥ 0.95", metrics["specificity"] >= 0.95),
        ("F1 Score ≥ 0.95",    metrics["f1_score"] >= 0.95),
    ]
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}  {check_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ Phase 2a complete: Flicker FFT validator PASSED all checks.")
    else:
        print("\n⚠  Phase 2a complete with warnings: some quality checks failed.")
