

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os                       # File system operations
import sys                      # System-level utilities (exit codes)
import json                     # Read/write JSON files
import numpy as np              # Numerical operations
import pandas as pd             # Tabular data manipulation

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS — Literature-based fallback values
# ──────────────────────────────────────────────────────────────────────────────
# These are used when the Kaggle dataset is unavailable or when a specific
# column is missing from the dataset.  Each value cites a peer-reviewed source.

FALLBACK_PHOTOPHOBIA = 0.80     # Goadsby et al. 2017: ~80% of migraineurs
                                 # report light sensitivity during attacks

FALLBACK_PHONOPHOBIA = 0.75     # Goadsby et al. 2017: ~75% of migraineurs
                                 # report sound sensitivity during attacks

FALLBACK_PRESSURE    = 0.28     # Katsuki et al. 2023: 28% of migraineurs
                                 # report weather/barometric pressure triggers

FALLBACK_VOC         = 0.45     # Zanchin et al. 2007: 45% of migraineurs
                                 # report odour/VOC as a trigger

FALLBACK_ATTACK_FREQ = 4.0      # Average attacks per month for chronic migraine

FALLBACK_PAIN_MEAN   = 6.5      # Mean pain intensity on 0-10 NRS scale

FALLBACK_PAIN_STD    = 1.8      # Standard deviation of pain intensity

# Path configuration — relative to this script's directory
RAW_DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT_FILE     = os.path.join(RAW_DATA_DIR, "migraine.csv")
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "population_priors.json")


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Column Name Normalisation
# ──────────────────────────────────────────────────────────────────────────────
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    # Step 1 — Lowercase and strip whitespace from all column names
    df.columns = df.columns.str.lower().str.strip()

    # Step 2 — Replace spaces, hyphens, and dots with underscores for consistency
    df.columns = df.columns.str.replace(r'[\s\-\.]+', '_', regex=True)

    # Step 3 — Map known column name variants to canonical names
    # This dict maps every observed variant → canonical name used downstream
    variant_map = {
        # Photophobia variants
        "photo_phobia":         "photophobia",
        "light_sensitivity":    "photophobia",
        "lightsensitivity":     "photophobia",
        "visual_sensitivity":   "photophobia",

        # Phonophobia variants
        "phono_phobia":         "phonophobia",
        "sound_sensitivity":    "phonophobia",
        "noise_sensitivity":    "phonophobia",

        # Pressure/weather variants
        "weather_sensitivity":  "pressure_sensitivity",
        "barometric":           "pressure_sensitivity",
        "weather_trigger":      "pressure_sensitivity",
        "pressure_trigger":     "pressure_sensitivity",
        "weather":              "pressure_sensitivity",

        # VOC/odour variants
        "odor_sensitivity":     "voc_sensitivity",
        "odour_sensitivity":    "voc_sensitivity",
        "osmophobia":           "voc_sensitivity",
        "smell_sensitivity":    "voc_sensitivity",
        "smell_trigger":        "voc_sensitivity",

        # Frequency variants
        "frequency":            "attack_frequency",
        "attacks_per_month":    "attack_frequency",
        "freq":                 "attack_frequency",
        "monthly_frequency":    "attack_frequency",

        # Intensity variants
        "intensity":            "pain_intensity",
        "pain_level":           "pain_intensity",
        "severity":             "pain_intensity",
        "pain_score":           "pain_intensity",
    }

    # Apply the mapping — only rename columns that exist and have a known variant
    df = df.rename(columns={k: v for k, v in variant_map.items() if k in df.columns})

    return df


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Convert Yes/No Columns to Binary Float
# ──────────────────────────────────────────────────────────────────────────────
def convert_yes_no_to_binary(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    
    # Set of string values that should map to 1.0 (positive/true)
    positive_values = {"yes", "y", "true", "1", "1.0"}

    for col in columns:
        if col in df.columns:
            # Convert to string, lowercase, strip whitespace, then map
            df[col] = (
                df[col]
                .astype(str)             # Convert any type to string first
                .str.lower()             # Lowercase for case-insensitive matching
                .str.strip()             # Remove leading/trailing whitespace
                .apply(lambda x: 1.0 if x in positive_values else 0.0)
            )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Safe Column Mean with Fallback
# ──────────────────────────────────────────────────────────────────────────────
def safe_mean(df: pd.DataFrame, column: str, fallback: float, label: str) -> float:
    
    if column not in df.columns:
        print(f"  ⚠  Column '{column}' not found — using literature fallback: "
              f"{label} = {fallback}")
        return fallback

    # Drop NaN values before computing mean
    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(series) == 0:
        print(f"  ⚠  Column '{column}' has no valid data — using fallback: "
              f"{label} = {fallback}")
        return fallback

    computed = float(series.mean())
    print(f"  ✓  {label} = {computed:.4f}  (from {len(series)} records)")
    return computed


def safe_std(df: pd.DataFrame, column: str, fallback: float, label: str) -> float:
    
    if column not in df.columns:
        print(f"  ⚠  Column '{column}' not found — using fallback: {label} = {fallback}")
        return fallback

    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(series) < 2:
        print(f"  ⚠  Column '{column}' has insufficient data — using fallback: "
              f"{label} = {fallback}")
        return fallback

    computed = float(series.std())
    print(f"  ✓  {label} = {computed:.4f}  (from {len(series)} records)")
    return computed


# ──────────────────────────────────────────────────────────────────────────────
# MAIN: Generate Synthetic Fallback Dataset
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_migraine_data(n_patients: int = 400) -> pd.DataFrame:
    
    print("\n" + "=" * 60)
    print("⚠  SYNTHETIC FALLBACK: Generating synthetic migraine data")
    print("   The Kaggle migraine dataset was not found at:")
    print(f"   {INPUT_FILE}")
    print("   Using literature-based clinical prevalence rates instead.")
    print("=" * 60 + "\n")

    # Set seed for reproducibility — same seed always produces same dataset
    rng = np.random.default_rng(42)

    # Each column is generated by sampling from a Bernoulli distribution
    # with probability equal to the literature-based prevalence rate.
    # Bernoulli(p) produces 1 with probability p, 0 with probability (1-p).
    data = {
        "photophobia":          rng.choice([0.0, 1.0], size=n_patients,
                                           p=[1 - FALLBACK_PHOTOPHOBIA,
                                              FALLBACK_PHOTOPHOBIA]),
        "phonophobia":          rng.choice([0.0, 1.0], size=n_patients,
                                           p=[1 - FALLBACK_PHONOPHOBIA,
                                              FALLBACK_PHONOPHOBIA]),
        "pressure_sensitivity": rng.choice([0.0, 1.0], size=n_patients,
                                           p=[1 - FALLBACK_PRESSURE,
                                              FALLBACK_PRESSURE]),
        "voc_sensitivity":      rng.choice([0.0, 1.0], size=n_patients,
                                           p=[1 - FALLBACK_VOC,
                                              FALLBACK_VOC]),
        # Attack frequency drawn from a Poisson distribution (count data)
        # λ = 4.0 attacks/month is a typical chronic migraine rate
        "attack_frequency":     rng.poisson(lam=FALLBACK_ATTACK_FREQ,
                                            size=n_patients).astype(float),
        # Pain intensity from a truncated normal distribution (0-10 NRS scale)
        "pain_intensity":       np.clip(
                                    rng.normal(loc=FALLBACK_PAIN_MEAN,
                                              scale=FALLBACK_PAIN_STD,
                                              size=n_patients),
                                    0.0, 10.0
                                ),
    }

    df = pd.DataFrame(data)
    print(f"   Generated {n_patients} synthetic patient records.\n")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN: Extract Population Priors
# ──────────────────────────────────────────────────────────────────────────────
def extract_priors() -> dict:
    
    print("=" * 60)
    print("PHASE 1: Population Prior Extraction")
    print("=" * 60)

    # ── Step 1: Load Dataset ──────────────────────────────────────────────
    # Try to load the real Kaggle migraine dataset first.
    # If the file is missing or unreadable, fall back to synthetic data.
    try:
        if os.path.exists(INPUT_FILE):
            print(f"\n📂 Loading real dataset: {INPUT_FILE}")
            df = pd.read_csv(INPUT_FILE)
            print(f"   Loaded {len(df)} records with {len(df.columns)} columns.")
            print(f"   Columns: {list(df.columns)}")
        else:
            # File doesn't exist — generate synthetic fallback
            df = generate_synthetic_migraine_data()
    except Exception as e:
        # File exists but can't be read (encoding issue, corruption, etc.)
        print(f"\n❌ Error reading dataset: {e}")
        print("   Falling back to synthetic data generation.")
        df = generate_synthetic_migraine_data()

    # ── Step 2: Normalise Column Names ────────────────────────────────────
    # Map variant column names to canonical names used by the pipeline
    print("\n── Normalising column names ──")
    df = normalise_columns(df)
    print(f"   Canonical columns: {list(df.columns)}")

    # ── Step 3: Convert Yes/No Strings to Binary Float ────────────────────
    # Binary columns that may be encoded as "yes"/"no" strings
    binary_cols = ["photophobia", "phonophobia", "pressure_sensitivity",
                   "voc_sensitivity"]
    df = convert_yes_no_to_binary(df, binary_cols)

    # ── Step 4: Compute Prior Probabilities ───────────────────────────────
    # For each trigger type, compute the fraction of migraine patients
    # who report sensitivity.  This fraction IS the population prior.
    #
    #   prior_photophobia = count(photophobia==1) / count(all_patients)
    #                     ≈ P(light-sensitive | has migraine diagnosis)
    #
    # This value is used as the initial belief for new users.
    print("\n── Computing Population Priors ──")

    priors = {
        # Trigger sensitivity priors — fraction of patients with each sensitivity
        "prior_photophobia":            safe_mean(df, "photophobia",
                                                  FALLBACK_PHOTOPHOBIA,
                                                  "P(photophobia|migraine)"),

        "prior_phonophobia":            safe_mean(df, "phonophobia",
                                                  FALLBACK_PHONOPHOBIA,
                                                  "P(phonophobia|migraine)"),

        "prior_pressure_sensitivity":   safe_mean(df, "pressure_sensitivity",
                                                  FALLBACK_PRESSURE,
                                                  "P(pressure_trigger|migraine)"),

        "prior_voc_sensitivity":        safe_mean(df, "voc_sensitivity",
                                                  FALLBACK_VOC,
                                                  "P(voc_trigger|migraine)"),

        # Descriptive statistics — used for synthetic data generation in Phase 3
        "attack_freq_mean_per_month":   safe_mean(df, "attack_frequency",
                                                  FALLBACK_ATTACK_FREQ,
                                                  "mean_attacks_per_month"),

        "pain_intensity_mean":          safe_mean(df, "pain_intensity",
                                                  FALLBACK_PAIN_MEAN,
                                                  "mean_pain_intensity"),

        "pain_intensity_std":           safe_std(df, "pain_intensity",
                                                 FALLBACK_PAIN_STD,
                                                 "std_pain_intensity"),
    }

    # ── Step 5: Compute Logit-Transformed Biases ──────────────────────────
    # The GRU output layer uses sigmoid activation: output = sigmoid(wx + b)
    # To make the initial output ≈ prior probability, we set the bias b such
    # that sigmoid(b) = prior.  Solving: b = log(p / (1 - p))  [logit function]
    #
    # Example: prior_photophobia = 0.80
    #   b = log(0.80 / 0.20) = log(4.0) = 1.386
    #   sigmoid(1.386) ≈ 0.80  ✓  The initial prediction matches our prior.
    print("\n── Computing Logit Biases for GRU Initialisation ──")

    prior_keys = ["prior_photophobia", "prior_phonophobia",
                  "prior_pressure_sensitivity", "prior_voc_sensitivity"]
    logit_biases = {}
    for key in prior_keys:
        p = priors[key]
        # Clip to avoid log(0) or log(inf) — probabilities must be in (0, 1)
        p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        # Logit transform: converts probability to log-odds
        logit = float(np.log(p_clipped / (1.0 - p_clipped)))
        logit_biases[f"{key}_logit_bias"] = round(logit, 6)
        print(f"  ✓  {key}: P={p:.4f} → logit_bias={logit:.4f}")

    # Add logit biases to the output dictionary
    priors.update(logit_biases)

    # Add weighted average prior (used as combined initial bias)
    # Weights reflect relative importance of each trigger modality
    weights = [0.25, 0.20, 0.30, 0.25]  # light, sound, pressure, VOC
    weighted_prior = sum(
        priors[k] * w for k, w in zip(prior_keys, weights)
    )
    priors["combined_weighted_prior"] = round(float(weighted_prior), 6)
    combined_logit = float(np.log(
        np.clip(weighted_prior, 1e-6, 1 - 1e-6) /
        (1.0 - np.clip(weighted_prior, 1e-6, 1 - 1e-6))
    ))
    priors["combined_logit_bias"] = round(combined_logit, 6)
    print(f"\n  ✓  Combined weighted prior: {weighted_prior:.4f} "
          f"→ logit_bias={combined_logit:.4f}")

    # Round all float values for clean JSON output
    priors = {k: round(v, 6) if isinstance(v, float) else v
              for k, v in priors.items()}

    return priors


# ──────────────────────────────────────────────────────────────────────────────
# MAIN: Save and Report
# ──────────────────────────────────────────────────────────────────────────────
def save_priors(priors: dict) -> None:
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Write JSON with indentation for human readability
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Saved population priors to: {OUTPUT_FILE}")

    # ── Print Formatted Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POPULATION PRIORS — SUMMARY")
    print("=" * 60)
    print(f"{'Parameter':<40} {'Value':>10}")
    print("-" * 52)
    for key, value in priors.items():
        if isinstance(value, float):
            print(f"  {key:<38} {value:>10.4f}")
        else:
            print(f"  {key:<38} {str(value):>10}")
    print("-" * 52)
    print("\nThese priors will be used to initialise the GRU model's")
    print("output bias weights, enabling meaningful predictions from Day 1.")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # Set global random seed for reproducibility
    np.random.seed(42)

    # Extract priors from dataset (or synthetic fallback)
    priors = extract_priors()

    # Save to file and print summary report
    save_priors(priors)

    print("\n✅ Phase 1 complete: Population priors extracted successfully.")
