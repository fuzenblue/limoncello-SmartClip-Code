import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Time constants
SAMPLE_INTERVAL_MIN = 10                        # 10-minute window
STEPS_PER_HOUR = 60 // SAMPLE_INTERVAL_MIN      # 6 steps per hour
STEPS_PER_DAY = 24 * STEPS_PER_HOUR             # 144 steps per day
PREDICTION_WINDOW_HOURS = 6                      # Predict migraine within 6h
PREDICTION_WINDOW_STEPS = PREDICTION_WINDOW_HOURS * STEPS_PER_HOUR  # 36 steps

# Simulation parameters
N_USERS = 3                   # Number of simulated users
N_DAYS_PER_USER = 30          # 30 days per user (1 month)
ATTACKS_PER_MONTH = 4         # Target attack rate (chronic migraine)

# Sensor baseline parameters
PRESSURE_BASELINE_HPA = 1013.25    # Standard sea-level pressure
PRESSURE_WALK_STD = 0.03           # Random walk step std (hPa per step)

VOC_BASELINE_OHM = 50000           # BME680 clean air resistance
VOC_DAILY_AMP = 5000               # Circadian variation amplitude

FLICKER_PROB_CLEAN = 0.60          # 60% of windows: no flicker (clean DC)
FLICKER_PROB_MILD = 0.35           # 35% of windows: noise floor flicker
FLICKER_PROB_ALERT = 0.05          # 5% of windows: mains-powered LED flicker

AUDIO_CLASS_PROBS = [0.70, 0.25, 0.05]  # P(Quiet), P(Traffic), P(Trigger)

# Escalation magnitude (how much sensors change before an attack)
ESCALATION_HOURS = 6              # Escalation begins 6 hours before attack
ESCALATION_STEPS = ESCALATION_HOURS * STEPS_PER_HOUR  # 36 steps

# File paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "..", "data")
PRIORS_FILE = os.path.join(DATA_DIR, "population_priors.json")
OUTPUT_DIR  = os.path.join(DATA_DIR, "master")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "smartclip_master_dataset.csv")
OUTPUT_STATS = os.path.join(OUTPUT_DIR, "dataset_summary.json")


# HELPER FUNCTIONS

def load_priors() -> dict:
    if os.path.exists(PRIORS_FILE):
        with open(PRIORS_FILE, "r", encoding="utf-8") as f:
            priors = json.load(f)
        print(f"Loaded priors from: {PRIORS_FILE}")
        return priors
    else:
        print("Priors file not found — using literature defaults")
        return {
            "prior_photophobia":           0.80,
            "prior_phonophobia":           0.75,
            "prior_pressure_sensitivity":  0.28,
            "prior_voc_sensitivity":       0.45,
            "attack_freq_mean_per_month":  4.0,
            "pain_intensity_mean":         6.5,
            "pain_intensity_std":          1.8,
        }


def generate_user_id(index: int) -> str:
    return hashlib.sha256(f"smartclip_user_{index}".encode()).hexdigest()[:12]


def generate_attack_times(n_days: int, n_attacks: int,
                          rng: np.random.Generator) -> list:

    total_steps = n_days * STEPS_PER_DAY
    min_spacing = STEPS_PER_DAY  # Minimum 24 hours between attacks

    attacks = []
    attempts = 0
    max_attempts = 1000

    while len(attacks) < n_attacks and attempts < max_attempts:
        # Random step within simulation, avoiding first and last 6 hours
        candidate = rng.integers(PREDICTION_WINDOW_STEPS,
                                 total_steps - STEPS_PER_HOUR)

        # Check minimum spacing from existing attacks
        too_close = any(abs(candidate - a) < min_spacing for a in attacks)
        if not too_close:
            attacks.append(int(candidate))
        attempts += 1

    attacks.sort()
    return attacks

def simulate_user_timeline(user_id: str, n_days: int, attack_times: list,
                           priors: dict, rng: np.random.Generator) -> pd.DataFrame:
    
    n_steps = n_days * STEPS_PER_DAY
    start_time = datetime(2024, 1, 1)

    # ── Determine this user's sensitivities ───────────────────────────────
    # Each user randomly has or doesn't have each sensitivity based on priors
    is_photophobic    = rng.random() < priors["prior_photophobia"]
    is_phonophobic    = rng.random() < priors["prior_phonophobia"]
    is_pressure_sens  = rng.random() < priors["prior_pressure_sensitivity"]
    is_voc_sensitive  = rng.random() < priors["prior_voc_sensitivity"]

    # ── Generate baselines ────────────────────────────────────────────────

    # PRESSURE: random walk starting at sea-level baseline
    pressure_steps = rng.normal(0, PRESSURE_WALK_STD, n_steps)
    pressure = PRESSURE_BASELINE_HPA + np.cumsum(pressure_steps)
    # Add seasonal drift
    t_days = np.arange(n_steps) / STEPS_PER_DAY
    pressure += 3.0 * np.sin(2 * np.pi * t_days / 30)

    # VOC: circadian-modulated baseline
    t_hours = np.arange(n_steps) / STEPS_PER_HOUR
    voc = VOC_BASELINE_OHM + VOC_DAILY_AMP * np.sin(
        2 * np.pi * (t_hours - 3) / 24
    )
    voc += rng.normal(0, 500, n_steps)

    # FLICKER: probabilistic assignment per window
    flicker_index = np.zeros(n_steps)
    flicker_freq = np.zeros(n_steps)
    for i in range(n_steps):
        roll = rng.random()
        if roll < FLICKER_PROB_CLEAN:
            # Clean DC light — no flicker
            flicker_index[i] = rng.uniform(0.0, 0.02)
            flicker_freq[i] = 0.0
        elif roll < FLICKER_PROB_CLEAN + FLICKER_PROB_MILD:
            # Mild noise-floor flicker
            flicker_index[i] = rng.uniform(0.02, 0.06)
            flicker_freq[i] = rng.choice([0.0, 50.0, 80.0])
        else:
            # Mains-powered LED flicker (above IEEE threshold)
            flicker_index[i] = rng.uniform(0.10, 0.25)
            flicker_freq[i] = rng.choice([100.0, 120.0, 150.0, 200.0])

    # AUDIO: probabilistic class assignment
    audio_class = rng.choice([0, 1, 2], size=n_steps, p=AUDIO_CLASS_PROBS)
    audio_confidence = rng.uniform(0.65, 0.99, n_steps)
    audio_db = np.where(audio_class == 0, rng.uniform(30, 55, n_steps),
               np.where(audio_class == 1, rng.uniform(60, 80, n_steps),
                                          rng.uniform(80, 100, n_steps)))

    # MOTION: mostly active during day (6am–10pm), stationary at night
    motion = np.zeros(n_steps, dtype=int)
    for i in range(n_steps):
        hour_of_day = (i % STEPS_PER_DAY) / STEPS_PER_HOUR
        if 6 <= hour_of_day <= 22:
            motion[i] = 1 if rng.random() < 0.85 else 0
        else:
            motion[i] = 1 if rng.random() < 0.15 else 0

    # ── Pre-attack escalation ─────────────────────────────────────────────
    # For each attack, escalate relevant sensors in the 6 hours before onset
    migraine_label = np.zeros(n_steps, dtype=int)
    pain_score = np.full(n_steps, np.nan)
    attack_confirmed = np.zeros(n_steps, dtype=int)

    for attack_step in attack_times:
        # Label all windows in [attack - 6h, attack) as positive
        label_start = max(0, attack_step - PREDICTION_WINDOW_STEPS)
        migraine_label[label_start:attack_step] = 1

        # Set attack confirmation and pain score at attack time
        if attack_step < n_steps:
            attack_confirmed[attack_step] = 1
            pain_mean = priors.get("pain_intensity_mean", 6.5)
            pain_std = priors.get("pain_intensity_std", 1.8)
            pain_score[attack_step] = np.clip(
                rng.normal(pain_mean, pain_std), 1, 10
            )

        # ── Escalate sensors based on user sensitivities ──────────────────
        # The escalation is GRADUAL: sensor anomalies increase as the
        # attack approaches.  This creates the temporal pattern the GRU
        # learns to detect.

        for step_offset in range(ESCALATION_STEPS):
            idx = attack_step - ESCALATION_STEPS + step_offset
            if idx < 0 or idx >= n_steps:
                continue

            # Escalation intensity ramps up linearly from 0 to 1
            intensity = step_offset / ESCALATION_STEPS

            # PRESSURE escalation: gradual pressure drop before attack
            if is_pressure_sens:
                # Simulate weather front: pressure drops progressively
                # Increased intensity to 4.0 hPa to ensure we trigger the simulation threshold
                pressure[idx] -= intensity * 4.0  

            # VOC escalation: gas resistance drops (more VOCs)
            if is_voc_sensitive:
                voc[idx] -= intensity * 15000  # Significant resistance drop
                voc[idx] = max(voc[idx], 5000)  # Never go below 5kΩ

            # FLICKER escalation: increase flicker index
            if is_photophobic:
                if rng.random() < 0.5 + 0.4 * intensity:  # Increasing prob
                    flicker_index[idx] = rng.uniform(0.10, 0.25)
                    flicker_freq[idx] = rng.choice([100.0, 120.0])

            # AUDIO escalation: increase trigger class probability
            if is_phonophobic:
                if rng.random() < 0.2 + 0.5 * intensity:
                    audio_class[idx] = 2  # Trigger sound
                    audio_confidence[idx] = rng.uniform(0.70, 0.95)
                    audio_db[idx] = rng.uniform(85, 100)

    # ── Compute derived features ──────────────────────────────────────────

    # Pressure features
    pressure_series = pd.Series(pressure)
    pressure_ddt_1h = pressure_series.diff(periods=STEPS_PER_HOUR) / 1.0
    pressure_ddt_6h = pressure_series.diff(periods=STEPS_PER_HOUR * 6) / 6.0
    # 6-hour rolling volatility
    pressure_std_6h = pressure_series.rolling(window=STEPS_PER_HOUR * 6, min_periods=1).std()

    # 30-day rolling z-score
    baseline_30d = 30 * STEPS_PER_DAY
    rolling_mean = pressure_series.rolling(baseline_30d, min_periods=STEPS_PER_HOUR).mean()
    rolling_std = pressure_series.rolling(baseline_30d, min_periods=STEPS_PER_HOUR).std()
    pressure_zscore = (pressure_series - rolling_mean) / (rolling_std + 1e-6)

    pressure_drop_alert_1h = (pressure_ddt_1h < -0.5).astype(int)
    pressure_drop_alert_6h = (pressure_ddt_6h < -0.5).astype(int)
    pressure_trigger = ((pressure_drop_alert_1h == 1) & (pressure_drop_alert_6h == 1)).astype(int)

    # VOC features
    voc_series = pd.Series(voc)
    voc_baseline_7d = 7 * STEPS_PER_DAY
    voc_rolling_mean = voc_series.rolling(voc_baseline_7d, min_periods=STEPS_PER_HOUR).mean()
    voc_rolling_std = voc_series.rolling(voc_baseline_7d, min_periods=STEPS_PER_HOUR).std()
    # Negate: lower resistance (more VOC) = positive z-score
    voc_zscore = -(voc_series - voc_rolling_mean) / (voc_rolling_std + 1e-6)
    voc_spike = (voc_zscore > 2.0).astype(int)
    # VOC derivative and persistence
    voc_ddt_10min = voc_series.diff() / SAMPLE_INTERVAL_MIN
    voc_persistent_spike = (voc_spike.rolling(window=3, min_periods=1).sum() >= 3).astype(int)

    # Flicker alert
    flicker_alert = ((flicker_index > 0.08) &
                     (flicker_freq >= 90) &
                     (flicker_freq <= 210)).astype(int)

    # Humidity and temperature (simulated, correlated with pressure)
    humidity = 60 + 10 * np.sin(2 * np.pi * t_hours / 24) + rng.normal(0, 3, n_steps)
    humidity = np.clip(humidity, 20, 95)
    temp = 28 + 3 * np.sin(2 * np.pi * t_hours / 24) + rng.normal(0, 0.5, n_steps)

    # Risk score: weighted linear combination of boolean alerts
    risk_score = (
        flicker_alert * priors["prior_photophobia"] * 0.25 +
        pressure_drop_alert_6h.values * priors["prior_pressure_sensitivity"] * 0.30 +
        voc_persistent_spike.values * priors["prior_voc_sensitivity"] * 0.25 +
        (audio_class == 2).astype(float) * priors["prior_phonophobia"] * 0.20
    )

    # ── Assemble DataFrame ────────────────────────────────────────────────
    timestamps = [start_time + timedelta(minutes=i * SAMPLE_INTERVAL_MIN)
                  for i in range(n_steps)]

    df = pd.DataFrame({
        # Metadata
        "user_id":                     user_id,
        "window_start":                timestamps,
        "motion_active":               motion,

        # Light features
        "flicker_index":               flicker_index,
        "flicker_freq_hz":             flicker_freq,
        "flicker_alert":               flicker_alert,

        # Pressure features
        "pressure_hpa":                pressure,
        "pressure_ddt_1h":             pressure_ddt_1h.values,
        "pressure_ddt_6h":             pressure_ddt_6h.values,
        "pressure_std_6h":             pressure_std_6h.values,
        "pressure_zscore":             pressure_zscore.values,
        "pressure_drop_alert_1h":      pressure_drop_alert_1h.values,
        "pressure_drop_alert_6h":      pressure_drop_alert_6h.values,
        "pressure_trigger":            pressure_trigger.values,

        # VOC features
        "voc_raw":                     voc,
        "voc_zscore":                  voc_zscore.values,
        "voc_ddt_10min":               voc_ddt_10min.values,
        "voc_spike":                   voc_spike.values,
        "voc_persistent_spike":        voc_persistent_spike.values,
        "humidity_pct":                humidity,
        "temp_celsius":                temp,

        # Audio features
        "audio_class":                 audio_class,
        "audio_confidence":            audio_confidence,
        "audio_db_mean":               audio_db,

        # User labels
        "pain_score_t0":               pain_score,
        "migraine_within_6h":          migraine_label,
        "attack_confirmed":            attack_confirmed,

        # Population priors (constant per user, loaded from Phase 1)
        "prior_photophobia":           priors["prior_photophobia"],
        "prior_phonophobia":           priors["prior_phonophobia"],
        "prior_pressure_sensitivity":  priors["prior_pressure_sensitivity"],
        "prior_voc_sensitivity":       priors["prior_voc_sensitivity"],

        # Composite risk score
        "risk_score":                  risk_score,
    })

    return df



# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA ENFORCEMENT AND VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

# The master dataset schema — all 27 columns with expected dtypes
SCHEMA = {
    "user_id":                     "object",
    "window_start":                "datetime64[ns]",
    "motion_active":               "int",
    "flicker_index":               "float64",
    "flicker_freq_hz":             "float64",
    "flicker_alert":               "int",
    "pressure_hpa":                "float64",
    "pressure_ddt_1h":             "float64",
    "pressure_ddt_6h":             "float64",
    "pressure_std_6h":             "float64",
    "pressure_zscore":             "float64",
    "pressure_drop_alert_1h":      "int",
    "pressure_drop_alert_6h":      "int",
    "pressure_trigger":            "int",
    "voc_raw":                     "float64",
    "voc_zscore":                  "float64",
    "voc_ddt_10min":               "float64",
    "voc_spike":                   "int",
    "voc_persistent_spike":        "int",
    "humidity_pct":                "float64",
    "temp_celsius":                "float64",
    "audio_class":                 "int",
    "audio_confidence":            "float64",
    "audio_db_mean":               "float64",
    "pain_score_t0":               "float64",
    "migraine_within_6h":          "int",
    "attack_confirmed":            "int",
    "prior_photophobia":           "float64",
    "prior_phonophobia":           "float64",
    "prior_pressure_sensitivity":  "float64",
    "prior_voc_sensitivity":       "float64",
    "risk_score":                  "float64",
}


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    
    for col, dtype in SCHEMA.items():
        if col in df.columns:
            if dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col])
            elif dtype == "object":
                df[col] = df[col].astype(str)
            elif dtype == "int":
                # Fill NaN with 0 before int conversion (NaN can't be int)
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def verify_dataset(df: pd.DataFrame) -> bool:
    
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    all_passed = True

    # ── Check 1: Schema completeness ──────────────────────────────────────
    missing_cols = [col for col in SCHEMA if col not in df.columns]
    check1 = len(missing_cols) == 0
    status = "✅ PASS" if check1 else "❌ FAIL"
    print(f"\n  {status}  All 27 columns present")
    if not check1:
        print(f"           Missing: {missing_cols}")
        all_passed = False

    # ── Check 2: Positive rate in expected range ──────────────────────────
    pos_rate = df["migraine_within_6h"].mean()
    check2 = 0.02 <= pos_rate <= 0.08
    status = "✅ PASS" if check2 else "⚠  WARN"
    print(f"  {status}  Positive rate: {pos_rate:.4f} (expected 0.02–0.08)")
    if not check2:
        all_passed = False

    # ── Check 3: Pre-attack escalation visible ───────────────────────────
    pre_attack = df[df["migraine_within_6h"] == 1]
    baseline = df[df["migraine_within_6h"] == 0]

    if len(pre_attack) > 0 and len(baseline) > 0:
        # Flicker alert rate should be higher pre-attack
        pre_flicker = pre_attack["flicker_alert"].mean()
        base_flicker = baseline["flicker_alert"].mean()
        check3a = pre_flicker > base_flicker
        status = "✅ PASS" if check3a else "⚠  WARN"
        print(f"  {status}  Flicker escalation: pre-attack={pre_flicker:.4f} "
              f"> baseline={base_flicker:.4f}")

        # Pressure drop alert should be higher pre-attack
        pre_pressure = pre_attack["pressure_drop_alert_1h"].mean()
        base_pressure = baseline["pressure_drop_alert_1h"].mean()
        check3b = pre_pressure > base_pressure
        status = "✅ PASS" if check3b else "⚠  WARN"
        print(f"  {status}  Pressure escalation: pre-attack={pre_pressure:.4f} "
              f"> baseline={base_pressure:.4f}")

        # VOC spike should be higher pre-attack
        pre_voc = pre_attack["voc_spike"].mean()
        base_voc = baseline["voc_spike"].mean()
        check3c = pre_voc > base_voc
        status = "✅ PASS" if check3c else "⚠  WARN"
        print(f"  {status}  VOC escalation: pre-attack={pre_voc:.4f} "
              f"> baseline={base_voc:.4f}")

        # Risk score should be higher pre-attack
        pre_risk = pre_attack["risk_score"].mean()
        base_risk = baseline["risk_score"].mean()
        check3d = pre_risk > base_risk
        status = "✅ PASS" if check3d else "⚠  WARN"
        print(f"  {status}  Risk score: pre-attack={pre_risk:.4f} "
              f"> baseline={base_risk:.4f}")

    # ── Check 4: Value ranges ─────────────────────────────────────────────
    check4a = df["flicker_index"].between(0, 1).all()
    status = "✅ PASS" if check4a else "❌ FAIL"
    print(f"  {status}  Flicker index in [0, 1]")
    if not check4a:
        all_passed = False

    check4b = (df["pressure_hpa"] > 800).all() and (df["pressure_hpa"] < 1100).all()
    status = "✅ PASS" if check4b else "⚠  WARN"
    print(f"  {status}  Pressure in [800, 1100] hPa")

    check4c = df["audio_class"].isin([0, 1, 2]).all()
    status = "✅ PASS" if check4c else "❌ FAIL"
    print(f"  {status}  Audio class in {{0, 1, 2}}")
    if not check4c:
        all_passed = False

    print("=" * 60)
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Set random seed for reproducibility
    np.random.seed(42)
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("PHASE 3: Master Dataset Assembly")
    print("=" * 60)

    # ── Step 1: Load priors ───────────────────────────────────────────────
    priors = load_priors()

    # ── Step 2: Simulate users ────────────────────────────────────────────
    all_users = []

    for user_idx in range(N_USERS):
        user_id = generate_user_id(user_idx)
        print(f"\n── Simulating User {user_idx + 1}/{N_USERS}: {user_id} ──")

        # Generate random attack times for this user
        attack_times = generate_attack_times(N_DAYS_PER_USER, ATTACKS_PER_MONTH, rng)
        print(f"   Attack times (steps): {attack_times}")
        print(f"   Attack times (hours): "
              f"{[t / STEPS_PER_HOUR for t in attack_times]}")

        # Simulate the user's complete sensor timeline
        user_df = simulate_user_timeline(
            user_id=user_id,
            n_days=N_DAYS_PER_USER,
            attack_times=attack_times,
            priors=priors,
            rng=rng
        )

        print(f"   Generated {len(user_df)} windows")
        print(f"   Positive windows: "
              f"{user_df['migraine_within_6h'].sum()} "
              f"({100 * user_df['migraine_within_6h'].mean():.1f}%)")
        all_users.append(user_df)

    # ── Step 3: Concatenate all users ─────────────────────────────────────
    master_df = pd.concat(all_users, ignore_index=True)
    print(f"\n── Combined Dataset ──")
    print(f"   Total windows: {len(master_df)}")
    print(f"   Total users:   {master_df['user_id'].nunique()}")

    # ── Step 4: Enforce schema ────────────────────────────────────────────
    master_df = enforce_schema(master_df)

    # ── Step 5: Verify dataset quality ────────────────────────────────────
    all_passed = verify_dataset(master_df)

    # ── Step 6: Save outputs ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📁 Saved dataset: {OUTPUT_CSV}")
    print(f"   Size: {os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.1f} MB")

    # Summary statistics
    summary = {
        "total_rows":              len(master_df),
        "total_users":             int(master_df["user_id"].nunique()),
        "days_per_user":           N_DAYS_PER_USER,
        "attacks_per_month":       ATTACKS_PER_MONTH,
        "positive_rate":           round(float(master_df["migraine_within_6h"].mean()), 4),
        "positive_count":          int(master_df["migraine_within_6h"].sum()),
        "negative_count":          int((master_df["migraine_within_6h"] == 0).sum()),
        "flicker_alert_rate":      round(float(master_df["flicker_alert"].mean()), 4),
        "pressure_drop_rate":      round(float(master_df["pressure_drop_alert_1h"].mean()), 4),
        "voc_spike_rate":          round(float(master_df["voc_spike"].mean()), 4),
        "trigger_audio_rate":      round(float((master_df["audio_class"] == 2).mean()), 4),
        "mean_risk_score":         round(float(master_df["risk_score"].mean()), 4),
        "schema_columns":          len(SCHEMA),
        "all_checks_passed":       all_passed,
    }

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📁 Saved summary: {OUTPUT_STATS}")

    # Print final summary
    print("\n" + "=" * 60)
    print("MASTER DATASET — FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Value':>12}")
    print("  " + "-" * 49)
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key:<35} {val:>12.4f}")
        elif isinstance(val, bool):
            print(f"  {key:<35} {'YES' if val else 'NO':>12}")
        else:
            print(f"  {key:<35} {val:>12}")
    print("=" * 60)

    if all_passed:
        print("\n✅ Phase 3 complete: Master dataset assembled and verified.")
    else:
        print("\n⚠  Phase 3 complete with warnings. Review verification output.")
