

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Sampling interval in minutes (matches Smart Clip 10-minute window)
SAMPLE_INTERVAL_MIN = 10

# Steps per hour: 60 / 10 = 6
STEPS_PER_HOUR = 60 // SAMPLE_INTERVAL_MIN

# Personal baseline window: 7 days of 10-minute readings
# 7 × 24 × 6 = 1008 steps
# WHY 7 DAYS? A 7-day window captures a full week cycle (weekdays vs weekends
# have different activity patterns), while being short enough to adapt to
# environmental changes like moving to a new apartment or starting a new job.
BASELINE_WINDOW_7D = 7 * 24 * STEPS_PER_HOUR  # = 1008

# Z-score threshold for VOC spike detection
# 2.0 standard deviations = top ~2.3% of personal readings
# This is the statistical definition of "unusual" — not "dangerous" per se,
# but significantly above what this specific user normally experiences.
SPIKE_ZSCORE_THRESHOLD = 2.0

# Persistent spike: at least 3 consecutive windows with spike = True
# 3 windows × 10 minutes = 30 minutes of sustained elevated VOC
# Brief spikes (< 30 min) are often transient (opening a cleaning product,
# walking past exhaust) and less likely to trigger a migraine.
PERSISTENT_SPIKE_WINDOW = 3

# BME680 simulator parameters
BME680_BASELINE_OHM = 50000     # Typical gas resistance in clean indoor air
                                 # Real BME680 outputs: 20,000–500,000 Ω
                                 # depending on air quality and sensor age

BME680_NOISE_STD = 500          # Gaussian noise standard deviation (Ω)
                                 # BME680 has moderate measurement noise

BME680_CIRCADIAN_AMP = 5000     # Amplitude of circadian drift (Ω)
                                 # Gas resistance varies throughout the day due to
                                 # ventilation patterns, cooking, commuting, etc.

VOC_EVENTS_PER_DAY = 0.8        # Average number of VOC spike events per day
                                 # ~1 significant exposure per day is realistic for
                                 # urban living (cooking, cleaning, traffic exposure)

VOC_DROP_RANGE = (10000, 30000)  # Drop in gas resistance during VOC event (Ω)
                                  # This translates to a significant VOC presence

VOC_RECOVERY_HOURS = (1, 3)      # Time to recover from VOC event (hours)
                                  # VOCs dissipate as air circulates

# Epsilon to prevent division by zero in z-score computation
EPSILON = 1e-6

# File paths
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "raw")
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "voc")
UCI_FILE     = os.path.join(RAW_DATA_DIR, "AirQualityUCI.csv")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "voc_features.csv")
OUTPUT_STATS = os.path.join(OUTPUT_DIR, "voc_stats.json")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING / SYNTHETIC GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def load_uci_air_quality() -> pd.DataFrame:
    
    if not os.path.exists(UCI_FILE):
        return None

    try:
        print(f"📂 Loading UCI Air Quality dataset: {UCI_FILE}")

        # The UCI Air Quality dataset uses semicolons as separators
        # and commas as decimal points (European format)
        df = pd.read_csv(UCI_FILE, sep=";", decimal=",",
                         na_values=[-200, -200.0])

        # Find VOC-related column
        voc_col = None
        for candidate in ["PT08.S2(NMHC)", "PT08_S2_NMHC", "NMHC(GT)"]:
            if candidate in df.columns:
                voc_col = candidate
                break

        if voc_col is None:
            print("   ⚠  Could not find VOC column in UCI dataset.")
            return None

        # Create timestamp from Date + Time columns
        if "Date" in df.columns and "Time" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                format="mixed", dayfirst=True, errors="coerce"
            )
        else:
            df["timestamp"] = pd.date_range(
                start="2024-01-01", periods=len(df), freq="1h"
            )

        # Convert NMHC sensor values to approximate BME680 gas resistance
        # The NMHC sensor gives higher values for more pollution
        # BME680 gives LOWER resistance for more pollution
        # Inversion formula: resistance ≈ max_value - sensor_value + baseline
        voc_raw = pd.to_numeric(df[voc_col], errors="coerce").dropna()
        max_sensor = voc_raw.max()

        result = pd.DataFrame({
            "timestamp": df.loc[voc_raw.index, "timestamp"],
            "voc_raw":   (max_sensor - voc_raw + BME680_BASELINE_OHM / 2).values,
        })
        result = result.dropna()

        # Resample to 10-minute intervals (UCI data is hourly)
        result = result.set_index("timestamp").resample("10min").interpolate()
        result = result.reset_index()

        # Limit to 30 days
        max_rows = 30 * 24 * STEPS_PER_HOUR
        if len(result) > max_rows:
            result = result.iloc[:max_rows].copy()

        print(f"   Loaded and resampled to {len(result)} records "
              f"({len(result) / (24 * STEPS_PER_HOUR):.1f} days)")
        return result

    except Exception as e:
        print(f"   ❌ Error reading UCI dataset: {e}")
        return None


def generate_synthetic_voc(n_days: int = 30) -> pd.DataFrame:
    
    print(f"\n⚠  SYNTHETIC FALLBACK: Generating {n_days}-day VOC series")
    print(f"   UCI Air Quality dataset not found at: {UCI_FILE}\n")

    rng = np.random.default_rng(42)

    n_steps = n_days * 24 * STEPS_PER_HOUR
    timestamps = pd.date_range(start="2024-01-01", periods=n_steps, freq="10min")

    # ── Component 1: Baseline ─────────────────────────────────────────────
    voc = np.full(n_steps, float(BME680_BASELINE_OHM))

    # ── Component 2: Circadian drift ─────────────────────────────────────
    # Gas resistance varies throughout the day following human activity:
    #   - Morning (7am): slight drop from cooking/shower products
    #   - Midday:        recovery as space ventilates
    #   - Evening (7pm): drop from cooking dinner
    #   - Night:         highest resistance (no activity, windows closed)
    t_hours = np.arange(n_steps) / STEPS_PER_HOUR
    # Double sine wave: peaks at ~3am (max clean), troughs at 7am and 7pm
    circadian = BME680_CIRCADIAN_AMP * np.sin(2 * np.pi * (t_hours - 3) / 24.0)
    voc += circadian

    # ── Component 3: Gaussian measurement noise ──────────────────────────
    noise = rng.normal(0, BME680_NOISE_STD, n_steps)
    voc += noise

    # ── Component 4: VOC spike events ─────────────────────────────────────
    # Each event simulates a significant VOC exposure:
    #   - Cooking: resistance drops ~15,000 Ω for 1–2 hours
    #   - Cleaning products: resistance drops ~20,000 Ω for 2–3 hours
    #   - Traffic exhaust: resistance drops ~10,000 Ω for 1 hour
    #
    # The drop is sharp (onset in 1–2 steps) but recovery is gradual
    # (exponential recovery over 1–3 hours) — this matches real VOC dynamics.
    n_events = 0
    total_event_prob = VOC_EVENTS_PER_DAY / (24 * STEPS_PER_HOUR)  # Per-step prob

    i = 0
    while i < n_steps:
        if rng.random() < total_event_prob:
            # Determine event magnitude and duration
            drop_magnitude = rng.integers(VOC_DROP_RANGE[0], VOC_DROP_RANGE[1])
            recovery_hours = rng.uniform(VOC_RECOVERY_HOURS[0], VOC_RECOVERY_HOURS[1])
            recovery_steps = int(recovery_hours * STEPS_PER_HOUR)

            # Sharp onset: resistance drops immediately
            # Exponential recovery: resistance gradually returns to baseline
            for j in range(min(recovery_steps, n_steps - i)):
                # Exponential decay: drop_magnitude × e^(-t/τ)
                # τ (tau) = recovery_steps / 3 — time constant
                decay = np.exp(-3.0 * j / recovery_steps)
                voc[i + j] -= drop_magnitude * decay

            n_events += 1
            i += recovery_steps
        else:
            i += 1

    # Ensure no negative resistance values
    voc = np.maximum(voc, 1000)

    print(f"   Generated {n_steps} readings over {n_days} days")
    print(f"   Injected {n_events} VOC spike events")
    print(f"   Resistance range: {voc.min():.0f} – {voc.max():.0f} Ω\n")

    return pd.DataFrame({
        "timestamp": timestamps,
        "voc_raw":   voc,
    })


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_voc_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Personal Baseline: 7-day rolling mean and std ─────────────────────
    # The baseline adapts over time:
    #   - If you move to a new city, the baseline adjusts within 7 days
    #   - If there's a seasonal change (e.g., wildfire smoke), it adapts
    #   - If you get a new air purifier, the baseline rises within a week
    #
    # min_periods prevents NaN values at the start of the series —
    # we use at least STEPS_PER_HOUR samples (1 hour) before computing stats
    df["voc_baseline_mean"] = (
        df["voc_raw"]
        .rolling(window=BASELINE_WINDOW_7D, min_periods=STEPS_PER_HOUR)
        .mean()
    )

    df["voc_baseline_std"] = (
        df["voc_raw"]
        .rolling(window=BASELINE_WINDOW_7D, min_periods=STEPS_PER_HOUR)
        .std()
    )

    # ── VOC Z-Score ───────────────────────────────────────────────────────
    # z = (current_reading - personal_mean) / personal_std
    #
    # IMPORTANT: For BME680, LOWER resistance = MORE VOCs = WORSE air
    # So a large NEGATIVE z-score (resistance dropped far below mean)
    # indicates a VOC event.  However, we want positive z-scores to
    # indicate "worse" conditions (for consistency with other features),
    # so we NEGATE the z-score:
    #   voc_zscore = -(voc_raw - mean) / std
    #
    # Now: positive voc_zscore = air quality worsened (resistance dropped)
    #       negative voc_zscore = air quality improved (resistance rose)
    df["voc_zscore"] = -(
        (df["voc_raw"] - df["voc_baseline_mean"]) /
        (df["voc_baseline_std"] + EPSILON)
    )

    # ── Rate of Change (dR/dt per minute) ─────────────────────────────────
    # How fast is the gas resistance changing?
    # A sudden drop indicates a new VOC source appeared nearby.
    # Units: Ohms per minute (negative = resistance dropping = air worsening)
    df["voc_ddt_10min"] = df["voc_raw"].diff() / SAMPLE_INTERVAL_MIN

    # ── VOC Spike Detection ───────────────────────────────────────────────
    # A "spike" occurs when the z-score exceeds our threshold.
    # z > 2.0 means the current reading is in the top ~2.3% of this
    # user's 7-day distribution — statistically "unusual".
    #
    # Note: this is NOT "dangerous" in an absolute sense — it just means
    # "unusual for YOU".  For a migraineur with osmophobia (odour sensitivity),
    # even brief exposure to unusual VOC levels can trigger an attack.
    df["voc_spike"] = (df["voc_zscore"] > SPIKE_ZSCORE_THRESHOLD).astype(int)

    # ── Persistent Spike Detection ────────────────────────────────────────
    # A persistent spike = spike sustained for 30+ consecutive minutes
    # (3 consecutive 10-minute windows all above threshold).
    #
    # Brief transient spikes (e.g., walking past a bus exhaust) are filtered
    # out — they rarely trigger migraines.  Sustained exposure is the concern.
    df["voc_persistent_spike"] = (
        df["voc_spike"]
        .rolling(window=PERSISTENT_SPIKE_WINDOW, min_periods=PERSISTENT_SPIKE_WINDOW)
        .sum() >= PERSISTENT_SPIKE_WINDOW
    ).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY AND REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary_stats(df: pd.DataFrame) -> dict:
    
    n_total = len(df)
    n_spike = int(df["voc_spike"].sum())
    n_persistent = int(df["voc_persistent_spike"].sum())

    stats = {
        "total_windows":          n_total,
        "voc_raw_mean_ohm":       round(float(df["voc_raw"].mean()), 0),
        "voc_raw_std_ohm":        round(float(df["voc_raw"].std()), 0),
        "voc_raw_min_ohm":        round(float(df["voc_raw"].min()), 0),
        "voc_raw_max_ohm":        round(float(df["voc_raw"].max()), 0),
        "voc_zscore_mean":        round(float(df["voc_zscore"].mean()), 4),
        "voc_zscore_max":         round(float(df["voc_zscore"].max()), 4),
        "spike_count":            n_spike,
        "spike_rate_pct":         round(100 * n_spike / n_total, 2),
        "persistent_spike_count": n_persistent,
        "persistent_spike_rate_pct": round(100 * n_persistent / n_total, 2),
        "spike_threshold_zscore": SPIKE_ZSCORE_THRESHOLD,
        "baseline_window_days":   7,
        "sample_interval_min":    SAMPLE_INTERVAL_MIN,
    }
    return stats


def print_summary(stats: dict) -> None:
    
    print("\n" + "=" * 60)
    print("VOC FEATURES — SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<40} {'Value':>12}")
    print("-" * 54)
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key:<38} {val:>12.4f}")
        else:
            print(f"  {key:<38} {val:>12}")
    print("-" * 54)
    print("\nNOTE: Higher gas resistance = cleaner air.")
    print("      VOC spikes correspond to DROPS in resistance.")
    print("      Z-score is negated so positive = worse air quality.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Set random seed for reproducibility
    np.random.seed(42)

    print("=" * 60)
    print("PHASE 2c: VOC Feature Engineering")
    print("=" * 60)

    # ── Step 1: Load or generate VOC data ─────────────────────────────────
    df = load_uci_air_quality()
    if df is None:
        df = generate_synthetic_voc(n_days=30)

    # ── Step 2: Compute VOC features ──────────────────────────────────────
    print("\n── Computing VOC Features ──")
    df = compute_voc_features(df)
    print(f"   Computed features for {len(df)} windows.")
    voc_cols = [c for c in df.columns if c.startswith("voc_")]
    print(f"   Feature columns: {voc_cols}")

    # ── Step 3: Compute and print summary ─────────────────────────────────
    stats = compute_summary_stats(df)
    print_summary(stats)

    # ── Step 4: Save outputs ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📁 Saved features: {OUTPUT_CSV}")

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"📁 Saved stats:    {OUTPUT_STATS}")

    print("\n✅ Phase 2c complete: VOC features computed successfully.")
