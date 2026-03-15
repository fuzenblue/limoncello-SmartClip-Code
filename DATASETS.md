# Dataset Management — Smart Clip

This document details the datasets used for training, validating, and simulating the Smart Clip environment across different pipeline phases.

---

## Datasets Used

### Phase 1 — Population Priors

| Dataset | Source | Records | Usage |
|---|---|---|---|
| Kaggle Migraine Dataset | [ranzeet013/migraine-dataset](https://www.kaggle.com/datasets/ranzeet013/migraine-dataset) | 400 patients, 24 columns | Extract P(photophobia), P(phonophobia), attack frequency distribution → initialise GRU bias weights on Day 1 |

**Columns extracted:**

| Prior | Value from dataset | Fallback (literature) |
|---|---|---|
| P(photophobia \| migraine) | **0.9800** (400 records) | 0.80 — Goadsby et al. 2017 |
| P(phonophobia \| migraine) | **0.9775** (400 records) | 0.75 — Goadsby et al. 2017 |
| P(pressure trigger \| migraine) | — column not in dataset | **0.28** — Katsuki et al. 2023 |
| P(VOC trigger \| migraine) | — column not in dataset | **0.45** — Zanchin et al. 2007 |

These four probabilities are converted to logit biases and loaded into the GRU output layer at first boot, enabling calibrated Day-1 predictions before any personal data is collected (**cold-start solution**).

---

### Phase 2a — Flicker Detection Validation

| Dataset | Source | Samples | Usage |
|---|---|---|---|
| Physics-based synthetic signals | Generated via NumPy (no public photodiode dataset exists) | 350 samples (7 types × 50) | Validate IEEE 1789-2015 compliant FFT flicker detector |
| BurstDeflicker methodology | [arXiv:2510.09996](https://arxiv.org/abs/2510.09996) | Reference only | Establishes synthetic flicker datasets as a validated evaluation methodology |

**Signal types generated:**

| Signal type | Label | Detection | Mean Flicker Index |
|---|---|---|---|
| Thai mains LED (100 Hz) | Flicker | 50/50 ✓ | 0.311 |
| US mains LED (120 Hz) | Flicker | 50/50 ✓ | 0.304 |
| PWM LED 150 Hz | Flicker | 50/50 ✓ | 0.516 |
| PWM LED 200 Hz | Flicker | 50/50 ✓ | 0.528 |
| Battery-powered LED | Clean | 0/50 ✓ | 0.001 |
| Sunlight | Clean | 0/50 ✓ | 0.002 |
| Phone flashlight | Clean | 0/50 ✓ | 0.001 |

**DSP configuration:** 800 Hz sample rate · 512-point FFT · Hamming window · 90–210 Hz detection band · Flicker Index threshold 0.08 (IEEE 1789-2015)

---

### Phase 2b — Pressure Feature Engineering

| Dataset | Source | Records | Usage |
|---|---|---|---|
| Max Planck Jena Climate Dataset | [arashnic/max-planck-weather-dataset](https://www.kaggle.com/datasets/arashnic/max-planck-weather-dataset) | 8,640 rows (60 days at 10-min intervals) | Compute dP/dt features; validate pressure-drop trigger logic |

**Features engineered:** `pressure_hpa` · `pressure_ddt_1h` (hPa/hr) · `pressure_ddt_6h` · `pressure_zscore` (30-day rolling) · `pressure_drop_alert` (threshold: −1.5 hPa/hr)

Clinical basis: Katsuki et al. (2023) — pressure drop over the preceding 6 hours has predictive gain 11.7 across 40,617 migraine patients.

---

### Phase 2c — VOC Feature Engineering

| Dataset | Source | Records | Usage |
|---|---|---|---|
| UCI Air Quality Dataset | [UCI ML Repository ID 360](https://archive.ics.uci.edu/dataset/360/air+quality) | 4,320 rows (resampled to 10-min) | Validate personalised Z-score normalisation for VOC gas resistance |

**Key design:** Raw gas resistance (Ω) is not comparable across users or environments. VOC Z-score = (gas_resistance − 7-day personal mean) / 7-day std. Spike flagged when Z > 2.0 σ above personal baseline. Spike rate on real UCI data: **4.17%**.

---

### Phase 2d — Audio Classification

| Dataset | Source | Samples | Usage |
|---|---|---|---|
| UrbanSound8K | [urbansounddataset.weebly.com](https://urbansounddataset.weebly.com/urbansound8k.html) | 8,732 clips (10 classes) | Train 1D-CNN audio classifier (when available) |
| Synthetic MFCC | Generated via NumPy class profiles | 498 samples (3 classes × 166) | Fallback when UrbanSound8K not downloaded |

**Class mapping (UrbanSound8K → Smart Clip):**

| UrbanSound8K class | Smart Clip class |
|---|---|
| siren, jackhammer, gun_shot | 2 — TRIGGER |
| car_horn, engine_idling, drilling | 1 — TRAFFIC |
| air_conditioner, children_playing, dog_bark, street_music | 0 — QUIET |

**MFCC pipeline:** Pre-emphasis → 25ms frames + Hamming window → FFT → 26-band Mel filterbank (300–8,000 Hz) → log energy → DCT → 13 coefficients → mean across frames

---

### Phase 3 — Master Dataset Assembly

**Schema:** 27 columns · one row per 10-minute observation window · prediction target: `migraine_within_6h` (bool)

| Column group | Columns | Source |
|---|---|---|
| Metadata | `user_id`, `window_start`, `motion_active` | Device / IMU |
| Light | `flicker_index`, `flicker_freq_hz`, `flicker_alert` | Photodiode → FFT |
| Pressure | `pressure_hpa`, `pressure_ddt_1h`, `pressure_ddt_6h`, `pressure_zscore`, `pressure_drop_alert` | BME680 |
| VOC/Gas | `voc_raw`, `voc_zscore`, `voc_spike`, `humidity_pct`, `temp_celsius` | BME680 gas |
| Audio | `audio_class`, `audio_confidence`, `audio_db_mean` | INMP441 → 1D-CNN |
| Labels | `pain_score_t0`, `migraine_within_6h`, `attack_confirmed` | App (user) |
| Priors | `prior_photophobia`, `prior_phonophobia`, `prior_pressure_sensitivity`, `prior_voc_sensitivity` | Phase 1 output |
| Composite | `risk_score` | Derived |

**Simulated for validation:** 3 users · 30 days each · 4 attacks/month · pre-attack sensor escalation injected in 6-hour window before each attack

| Dataset metric | Value |
|---|---|
| Total rows | 12,960 |
| Positive windows (migraine_within_6h=1) | 432 (3.33%) |
| Negative windows | 12,528 (96.67%) |
| Flicker alert rate | 7.25% |
| Pressure drop rate | 1.93% |
| VOC spike rate | 0.56% |
