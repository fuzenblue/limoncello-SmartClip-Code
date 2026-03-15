

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

# Sampling interval in minutes.  The Smart Clip sensor samples every 10 min.
SAMPLE_INTERVAL_MIN = 10

# Number of 10-minute steps per hour: 60 / 10 = 6
STEPS_PER_HOUR = 60 // SAMPLE_INTERVAL_MIN  # = 6

# dP/dt window sizes (in number of 10-minute steps)
WINDOW_1H = STEPS_PER_HOUR           # 6 steps  = 1 hour
WINDOW_6H = STEPS_PER_HOUR * 6       # 36 steps = 6 hours

# Rolling baseline window: 30 days of 10-minute readings
# 30 days × 24 hours × 6 steps/hour = 4320 steps
BASELINE_WINDOW_30D = 30 * 24 * STEPS_PER_HOUR  # = 4320

# Rolling volatility window: 6 hours
STD_WINDOW_6H = WINDOW_6H  # 36 steps

# Clinical thresholds from Katsuki et al. (2023)
RAPID_DROP_1H_THRESHOLD = -1.5    # hPa/hr — 1-hour rapid drop threshold
                                   # A drop faster than 1.5 hPa/hr indicates
                                   # a fast-moving weather front, which is
                                   # strongly associated with migraine onset.

SUSTAINED_DROP_6H_THRESHOLD = -0.5  # hPa/hr — 6-hour sustained drop threshold
                                     # Even a slow sustained drop can trigger
                                     # migraines if prolonged over 6 hours.

# Synthetic pressure generation parameters
BASELINE_PRESSURE_HPA = 1013.25  # Standard atmosphere at sea level (hPa)
                                  # Bangkok average: ~1010 hPa
                                  # Will be used as the mean for synthetic data

RANDOM_WALK_STD = 0.05            # Standard deviation of pressure random walk
                                  # per 10-minute step (hPa). Generates realistic
                                  # hour-to-hour pressure variation.

DROP_EVENT_PROB = 0.02            # Probability per step of starting a rapid
                                  # pressure drop event. ≈ 2% per step =
                                  # roughly 2.9 events per day × 2% ≈ 0.6
                                  # significant drops per day (realistic for
                                  # tropical monsoon climate).

DROP_MAGNITUDE_RANGE = (0.2, 0.5)  # hPa per 10-minute step during a drop event
                                    # A severe drop: 0.5 hPa/step × 6 steps/hr
                                    # = 3.0 hPa/hr (major frontal passage)

DROP_DURATION_RANGE = (6, 18)      # Duration of drop events in steps (1–3 hours)
                                    # 6 steps = 1 hour, 18 steps = 3 hours

# File paths
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "raw")
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "pressure")
JENA_FILE    = os.path.join(RAW_DATA_DIR, "jena_climate.csv")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "pressure_features.csv")
OUTPUT_STATS = os.path.join(OUTPUT_DIR, "pressure_stats.json")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING / SYNTHETIC GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def load_jena_climate() -> pd.DataFrame:
    
    if not os.path.exists(JENA_FILE):
        return None

    try:
        print(f"📂 Loading Jena Climate dataset: {JENA_FILE}")
        df = pd.read_csv(JENA_FILE)

        # Identify pressure column — try several common names
        pressure_col = None
        for candidate in ["p (mbar)", "p_mbar", "pressure", "Press"]:
            if candidate in df.columns:
                pressure_col = candidate
                break

        if pressure_col is None:
            print("   ⚠  Could not find pressure column in Jena dataset.")
            return None

        # Identify timestamp column
        time_col = None
        for candidate in ["Date Time", "datetime", "date_time", "Date"]:
            if candidate in df.columns:
                time_col = candidate
                break

        # Build clean dataframe
        result = pd.DataFrame()
        if time_col:
            result["timestamp"] = pd.to_datetime(df[time_col])
        else:
            # Generate synthetic timestamps at 10-min intervals
            result["timestamp"] = pd.date_range(
                start="2024-01-01", periods=len(df), freq="10min"
            )

        result["pressure_hpa"] = pd.to_numeric(df[pressure_col], errors="coerce")
        result = result.dropna(subset=["pressure_hpa"])

        # Limit to 60 days for manageable processing
        max_rows = 60 * 24 * STEPS_PER_HOUR  # 60 days
        if len(result) > max_rows:
            result = result.iloc[:max_rows].copy()

        print(f"   Loaded {len(result)} records "
              f"({len(result) / (24 * STEPS_PER_HOUR):.1f} days)")
        return result

    except Exception as e:
        print(f"   ❌ Error reading Jena dataset: {e}")
        return None


def generate_synthetic_pressure(n_days: int = 60) -> pd.DataFrame:
    
    print(f"\n⚠  SYNTHETIC FALLBACK: Generating {n_days}-day pressure series")
    print(f"   Jena Climate dataset not found at: {JENA_FILE}\n")

    rng = np.random.default_rng(42)

    n_steps = n_days * 24 * STEPS_PER_HOUR
    timestamps = pd.date_range(start="2024-01-01", periods=n_steps, freq="10min")

    # ── Component 1: Seasonal drift (30-day sine wave) ────────────────────
    # Barometric pressure has a ~semiannual cycle with ~5 hPa variation
    t_days = np.arange(n_steps) / (24 * STEPS_PER_HOUR)  # Convert steps to days
    seasonal = 5.0 * np.sin(2 * np.pi * t_days / 30.0)

    # ── Component 2: Diurnal cycle (24-hour sine wave) ────────────────────
    # Atmospheric pressure changes by ~1 hPa over the day due to solar heating
    t_hours = np.arange(n_steps) / STEPS_PER_HOUR  # Convert steps to hours
    diurnal = 1.0 * np.sin(2 * np.pi * t_hours / 24.0)

    # ── Component 3: Random walk ──────────────────────────────────────────
    # Small random steps simulating chaotic atmospheric pressure changes
    # Cumulative sum of Gaussian noise creates a realistic random walk
    random_steps = rng.normal(0, RANDOM_WALK_STD, n_steps)
    random_walk = np.cumsum(random_steps)

    # Combine all components with the baseline
    pressure = BASELINE_PRESSURE_HPA + seasonal + diurnal + random_walk

    # ── Component 4: Rapid drop events ────────────────────────────────────
    # These simulate weather fronts passing through.  During a front passage,
    # pressure drops rapidly (0.2–0.5 hPa per 10 minutes) for 1–3 hours,
    # then partially recovers.
    n_events = 0
    i = 0
    while i < n_steps:
        if rng.random() < DROP_EVENT_PROB:
            # Start a drop event
            duration = rng.integers(DROP_DURATION_RANGE[0], DROP_DURATION_RANGE[1])
            magnitude = rng.uniform(DROP_MAGNITUDE_RANGE[0], DROP_MAGNITUDE_RANGE[1])

            # Apply progressive drop
            for j in range(min(duration, n_steps - i)):
                pressure[i + j] -= magnitude * (j + 1)

            # Partial recovery (pressure rises back, but not fully)
            recovery_steps = duration
            for j in range(min(recovery_steps, n_steps - i - duration)):
                recovery_amount = magnitude * duration * 0.7 * (j + 1) / recovery_steps
                pressure[i + duration + j] += recovery_amount

            n_events += 1
            i += duration + recovery_steps  # Skip past this event
        else:
            i += 1

    print(f"   Generated {n_steps} readings over {n_days} days")
    print(f"   Injected {n_events} rapid-drop events")
    print(f"   Pressure range: {pressure.min():.1f} – {pressure.max():.1f} hPa\n")

    return pd.DataFrame({
        "timestamp":    timestamps,
        "pressure_hpa": pressure,
    })


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Ensure data is sorted by time (required for rolling windows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Feature 1: pressure_ddt_1h (dP/dt over 1 hour) ───────────────────
    # Rate of change = (current pressure - pressure 1 hour ago) / 1 hour
    # At 10-minute intervals, "1 hour ago" = 6 steps back.
    #
    # A negative value means pressure is DROPPING.
    # Clinical threshold: < -1.5 hPa/hr indicates a rapid frontal passage.
    #
    # .diff(periods=WINDOW_1H) computes: value[i] - value[i - 6]
    # Dividing by 1 gives units of hPa/hour.
    df["pressure_ddt_1h"] = df["pressure_hpa"].diff(periods=WINDOW_1H) / 1.0

    # ── Feature 2: pressure_ddt_6h (dP/dt over 6 hours) ──────────────────
    # Rate of change over a longer window captures sustained pressure changes
    # that may trigger migraines even if the hourly rate is moderate.
    #
    # Katsuki et al. (2023) found the 6-hour window had the highest predictive
    # gain (odds ratio = 11.7) across 40,617 migraine patient diaries.
    #
    # .diff(periods=WINDOW_6H) computes: value[i] - value[i - 36]
    # Dividing by 6 gives units of hPa/hour (average rate over 6 hours).
    df["pressure_ddt_6h"] = df["pressure_hpa"].diff(periods=WINDOW_6H) / 6.0

    # ── Feature 3: pressure_std_6h (6-hour rolling standard deviation) ────
    # Measures pressure VOLATILITY — rapid up-and-down swings even if the
    # net change is zero.  Some migraineurs are sensitive to instability
    # itself, not just drops.
    #
    # min_periods=1 ensures we get values even at the start of the series
    # (where there aren't enough data points for a full 36-step window).
    df["pressure_std_6h"] = (
        df["pressure_hpa"]
        .rolling(window=STD_WINDOW_6H, min_periods=1)
        .std()
    )

    # ── Feature 4: pressure_zscore (z-score vs 30-day rolling mean) ───────
    # Z-score normalisation removes the effect of:
    #   - Altitude (Denver vs Bangkok — different absolute pressures)
    #   - Climate zone (tropical vs temperate)
    #   - Seasonal baseline shifts (winter high vs summer low)
    #
    # Formula: z = (x - μ_30d) / σ_30d
    # where μ_30d and σ_30d are the rolling mean and std over 30 days.
    #
    # A z-score of -2.0 means "this pressure is 2 standard deviations below
    # my recent 30-day average" — regardless of location or season.
    rolling_mean = (
        df["pressure_hpa"]
        .rolling(window=BASELINE_WINDOW_30D, min_periods=WINDOW_6H)
        .mean()
    )
    rolling_std = (
        df["pressure_hpa"]
        .rolling(window=BASELINE_WINDOW_30D, min_periods=WINDOW_6H)
        .std()
    )

    # Add small epsilon (1e-6) to denominator to prevent division by zero
    # This can happen if all pressure readings are identical (unlikely in
    # practice, but possible with very short synthetic series).
    df["pressure_zscore"] = (
        (df["pressure_hpa"] - rolling_mean) / (rolling_std + 1e-6)
    )

    # ── Feature 5: pressure_drop_alert_1h (boolean, rapid 1-hour drop) ────
    # True when dP/dt(1h) < -1.5 hPa/hr (Katsuki threshold)
    # This indicates a fast-moving weather front is passing NOW.
    df["pressure_drop_alert_1h"] = (
        df["pressure_ddt_1h"] < RAPID_DROP_1H_THRESHOLD
    ).astype(int)

    # ── Feature 6: pressure_drop_alert_6h (boolean, sustained 6-hour drop) ─
    # True when dP/dt(6h) < -0.5 hPa/hr
    # This indicates sustained pressure decline over half a day.
    df["pressure_drop_alert_6h"] = (
        df["pressure_ddt_6h"] < SUSTAINED_DROP_6H_THRESHOLD
    ).astype(int)

    # ── Feature 7: pressure_trigger (combined alert) ──────────────────────
    # True when BOTH the 1-hour and 6-hour alerts fire simultaneously.
    # This indicates a severe, sustained pressure event — the strongest
    # trigger condition.  Having both alerts ensures we don't false-alarm
    # on brief gusts (1h only) or slow seasonal trends (6h only).
    df["pressure_trigger"] = (
        (df["pressure_drop_alert_1h"] == 1) &
        (df["pressure_drop_alert_6h"] == 1)
    ).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY AND REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary_stats(df: pd.DataFrame) -> dict:
    
    n_total = len(df)
    # Count only rows where features are fully computed (exclude warmup period)
    n_valid = df["pressure_ddt_6h"].notna().sum()
    n_alert_1h = int(df["pressure_drop_alert_1h"].sum())
    n_alert_6h = int(df["pressure_drop_alert_6h"].sum())
    n_trigger  = int(df["pressure_trigger"].sum())

    stats = {
        "total_windows":           n_total,
        "valid_windows":           int(n_valid),
        "warmup_skipped":          int(n_total - n_valid),
        "pressure_mean_hpa":       round(float(df["pressure_hpa"].mean()), 2),
        "pressure_std_hpa":        round(float(df["pressure_hpa"].std()), 2),
        "pressure_min_hpa":        round(float(df["pressure_hpa"].min()), 2),
        "pressure_max_hpa":        round(float(df["pressure_hpa"].max()), 2),
        "ddt_1h_mean":             round(float(df["pressure_ddt_1h"].mean()), 4),
        "ddt_1h_min":              round(float(df["pressure_ddt_1h"].min()), 4),
        "ddt_6h_mean":             round(float(df["pressure_ddt_6h"].mean()), 4),
        "ddt_6h_min":              round(float(df["pressure_ddt_6h"].min()), 4),
        "alert_1h_count":          n_alert_1h,
        "alert_1h_rate_pct":       round(100 * n_alert_1h / n_total, 2),
        "alert_6h_count":          n_alert_6h,
        "alert_6h_rate_pct":       round(100 * n_alert_6h / n_total, 2),
        "trigger_count":           n_trigger,
        "trigger_rate_pct":        round(100 * n_trigger / n_total, 2),
        "rapid_drop_threshold_hpa_hr": RAPID_DROP_1H_THRESHOLD,
        "sustained_drop_threshold_hpa_hr": SUSTAINED_DROP_6H_THRESHOLD,
        "sample_interval_min":     SAMPLE_INTERVAL_MIN,
    }
    return stats


def print_summary(stats: dict) -> None:
    
    print("\n" + "=" * 60)
    print("PRESSURE FEATURES — SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<40} {'Value':>12}")
    print("-" * 54)
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key:<38} {val:>12.4f}")
        else:
            print(f"  {key:<38} {val:>12}")
    print("-" * 54)
    print("\nTrigger events are windows where BOTH 1-hour rapid drop")
    print("AND 6-hour sustained drop are active simultaneously.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Set random seed for reproducibility of synthetic data
    np.random.seed(42)

    print("=" * 60)
    print("PHASE 2b: Pressure Feature Engineering")
    print("=" * 60)

    # ── Step 1: Load or generate pressure data ────────────────────────────
    df = load_jena_climate()
    if df is None:
        df = generate_synthetic_pressure(n_days=60)

    # ── Step 2: Compute pressure features ─────────────────────────────────
    print("\n── Computing Pressure Features ──")
    df = compute_pressure_features(df)
    print(f"   Computed features for {len(df)} windows.")
    print(f"   Feature columns: {[c for c in df.columns if c.startswith('pressure_')]}")

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

    print("\n✅ Phase 2b complete: Pressure features computed successfully.")
