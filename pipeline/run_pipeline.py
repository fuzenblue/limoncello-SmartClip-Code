

import os
import sys
import time
import argparse
import subprocess
import json

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Directory containing all pipeline scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered list of pipeline phases
# Each entry: (phase_id, script_filename, description)
PHASES = [
    ("1",        "prior_from_migraine_dataset.py",  "Population priors extraction"),
    ("2a",       "flicker_fft_validator.py",         "Flicker FFT validation"),
    ("2b",       "pressure_features.py",             "Pressure feature engineering"),
    ("2c",       "voc_features.py",                  "VOC feature engineering"),
    ("2d",       "audio_features.py",                "Audio feature extraction"),
    ("3",        "build_master_dataset.py",           "Master dataset assembly"),
    ("train",    "train_gru_model.py",               "GRU model training"),
    ("quantise", "quantise_and_validate.py",         "INT8 quantisation"),
]

# Aliases for convenience
PHASE_ALIASES = {
    "2": ["2a", "2b", "2c", "2d"],       # "2" runs all Phase 2 scripts
    "all": [p[0] for p in PHASES],         # "all" runs everything
    "features": ["2a", "2b", "2c", "2d"],  # Feature extraction only
}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_phase(phase_id: str, script: str, description: str) -> dict:
    
    script_path = os.path.join(SCRIPT_DIR, script)

    if not os.path.exists(script_path):
        return {
            "phase": phase_id,
            "status": "SKIPPED",
            "error": f"Script not found: {script_path}",
            "runtime_sec": 0,
        }

    print("\n" + "═" * 70)
    print(f"  PHASE {phase_id.upper()}: {description}")
    print(f"  Script: {script}")
    print("═" * 70)

    start = time.time()

    try:
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            capture_output=False,  # Let output flow to terminal
            text=True,
            timeout=600,           # 10-minute timeout per phase
        )

        elapsed = time.time() - start
        status = "PASS" if result.returncode == 0 else "FAIL"

        return {
            "phase": phase_id,
            "status": status,
            "returncode": result.returncode,
            "runtime_sec": round(elapsed, 1),
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "phase": phase_id,
            "status": "TIMEOUT",
            "error": "Phase exceeded 10-minute timeout",
            "runtime_sec": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "phase": phase_id,
            "status": "ERROR",
            "error": str(e),
            "runtime_sec": round(elapsed, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def print_final_report(results: list, total_time: float):
    
    print("\n" + "═" * 70)
    print("  SMART CLIP PIPELINE — FINAL REPORT")
    print("═" * 70)

    print(f"\n  {'Phase':<12} {'Description':<35} {'Status':>8} {'Time':>8}")
    print("  " + "-" * 67)

    n_pass = 0
    n_fail = 0
    n_skip = 0

    for r in results:
        phase_id = r["phase"]
        # Find description from PHASES list
        desc = next((p[2] for p in PHASES if p[0] == phase_id), "Unknown")
        status = r["status"]
        runtime = r["runtime_sec"]

        # Status emoji
        if status == "PASS":
            icon = "✅"
            n_pass += 1
        elif status == "FAIL":
            icon = "❌"
            n_fail += 1
        elif status == "SKIPPED":
            icon = "⏭️"
            n_skip += 1
        elif status == "TIMEOUT":
            icon = "⏱️"
            n_fail += 1
        else:
            icon = "⚠️"
            n_fail += 1

        print(f"  {phase_id:<12} {desc:<35} {icon} {status:>5} {runtime:>6.1f}s")

        if "error" in r:
            print(f"  {'':>12} ⚠ {r['error']}")

    print("  " + "-" * 67)
    print(f"  {'TOTAL':<12} {'':>35} {'':>8} {total_time:>6.1f}s")
    print(f"\n  Passed: {n_pass}  Failed: {n_fail}  Skipped: {n_skip}")

    # Check for key output files
    print("\n  ── Output Files ──")
    output_files = [
        ("data/population_priors.json",                  "Phase 1 priors"),
        ("data/flicker/flicker_validation_report.json",  "Flicker validation"),
        ("data/pressure/pressure_features.csv",          "Pressure features"),
        ("data/voc/voc_features.csv",                    "VOC features"),
        ("data/audio/audio_features.csv",                "Audio features"),
        ("data/master/smartclip_master_dataset.csv",     "Master dataset"),
        ("models/training_results.json",                 "Training results"),
        ("models/quantisation_report.json",              "Quantisation report"),
        ("firmware/include/model_data.h",                "C header (model)"),
    ]

    for rel_path, desc in output_files:
        full_path = os.path.join(SCRIPT_DIR, "..", rel_path)
        exists = os.path.exists(full_path)
        icon = "✅" if exists else "❌"
        size = ""
        if exists:
            size_bytes = os.path.getsize(full_path)
            if size_bytes > 1024 * 1024:
                size = f"({size_bytes / 1024 / 1024:.1f} MB)"
            else:
                size = f"({size_bytes / 1024:.1f} KB)"
        print(f"  {icon} {rel_path:<50} {size}")

    print("\n" + "═" * 70)

    if n_fail == 0:
        print("  🎉 All pipeline phases completed successfully!")
    else:
        print(f"  ⚠  {n_fail} phase(s) had issues. Review output above.")

    print("═" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Clip Data & Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        help="Phase to run (1, 2, 2a, 2b, 2c, 2d, 3, train, quantise, all)"
    )

    args = parser.parse_args()

    print("═" * 70)
    print("  🧠 SMART CLIP — Data & Training Pipeline")
    print("  Hyper-local Migraine Sensitivity Companion")
    print("═" * 70)

    # Resolve phase aliases
    phase_input = args.phase.lower()
    if phase_input in PHASE_ALIASES:
        phases_to_run = PHASE_ALIASES[phase_input]
    else:
        phases_to_run = [phase_input]

    # Validate phase IDs
    valid_phase_ids = {p[0] for p in PHASES}
    for pid in phases_to_run:
        if pid not in valid_phase_ids:
            print(f"❌ Unknown phase: '{pid}'")
            print(f"   Valid phases: {sorted(valid_phase_ids)}")
            sys.exit(1)

    print(f"\n  Phases to run: {', '.join(phases_to_run)}")
    print(f"  Working directory: {SCRIPT_DIR}")

    # Ensure data directories exist
    for subdir in ["raw", "flicker", "pressure", "voc", "audio", "master"]:
        os.makedirs(os.path.join(SCRIPT_DIR, "..", "data", subdir), exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "..", "models"), exist_ok=True)

    # Run phases
    total_start = time.time()
    results = []

    for phase_id, script, description in PHASES:
        if phase_id in phases_to_run:
            result = run_phase(phase_id, script, description)
            results.append(result)

            # If a phase fails, continue with remaining phases
            # (don't abort — later phases may still produce useful output)
            if result["status"] != "PASS":
                print(f"\n⚠  Phase {phase_id} did not pass cleanly. "
                      f"Continuing with remaining phases...")

    total_time = time.time() - total_start

    # Print final report
    print_final_report(results, total_time)
