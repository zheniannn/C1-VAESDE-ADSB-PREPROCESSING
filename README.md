# VAESDE_preprocess — GA Trajectory Preprocessing

Preprocessing pipeline for fixed-wing general aviation (GA) aircraft trajectories from raw ADS-B state vectors (OpenSky Network). Produces normalised sequence arrays for use with the **VAESDE_model** repository.

Four study dates: **2022-06-06, 2022-06-13, 2022-06-20, 2022-06-27**.

---

## Repository structure

```
VAESDE_preprocess/
├── preprocess_01_concat_daily.py              # Merge 24 hourly CSVs → 1 daily CSV
├── preprocess_02_filter_ga_aircraft.py        # Filter aircraft database to GA types
├── preprocess_03_filter_sort_daily.py         # Filter daily states to GA icao24s
├── preprocess_04_generate_trajectories.py     # Build trajectory segments
├── preprocess_05_convert_segments_to_enu.py   # Convert lat/lon/alt → ENU + velocity
├── preprocess_06_clean_enu_velocity_outliers.py   # Drop segments with GPS artefacts
├── preprocess_07_split_train_test_by_segment.py   # 90/10 train/test split by segment_id
├── preprocess_08_make_sequence_dataset.py         # Sliding-window sequences → X_train/X_test .npy
│
└── data/
    ├── archive/                               # Raw source files (not tracked in git)
    │   ├── aircraftDatabase-2022-06.csv       # OpenSky aircraft database (82 MB)
    │   └── states_YYYY-MM-DD-HH.csv          # Raw hourly ADS-B state files (96 files, gitignored)
    ├── preprocess_01_output/                  # Daily merged state files (gitignored, ~10 GB each)
    ├── preprocess_02_output/
    │   └── aircraftDatabase-2022-06-FixedWingGA.csv   # GA aircraft registry (119 666 aircraft)
    ├── preprocess_03_output/                  # GA-filtered daily state files (gitignored, ~600 MB each)
    ├── preprocess_04_output/
    │   ├── trajectory_segments.csv            # gitignored, 1.4 GB
    │   ├── trajectory_summary.csv             # One row per segment (114 826 segments)
    │   └── Results.csv                        # Pipeline summary statistics
    ├── preprocess_05_output/
    │   ├── trajectory_segments_enu_with_velocity.csv  # gitignored, 2.5 GB
    │   └── enu_conversion_summary.csv
    ├── preprocess_06_output/
    │   ├── trajectory_segments_enu_clean.csv  # gitignored, 3.4 GB
    │   ├── enu_cleaning_summary.csv
    │   └── dropped_segments_velocity_outliers.csv
    ├── train_segments_enu_clean.csv           # gitignored, ~684 MB
    ├── test_segments_enu_clean.csv            # gitignored, ~73 MB
    ├── split_summary.csv
    ├── X_train.npy                            # gitignored, 1.36 GB  ← handoff to VAESDE_model
    ├── X_test.npy                             # gitignored, 155 MB   ← handoff to VAESDE_model
    ├── normalisation_mean.csv                 # ← handoff to VAESDE_model
    ├── normalisation_std.csv                  # ← handoff to VAESDE_model
    ├── train_sequence_metadata.csv            # gitignored, 80 MB    ← handoff to VAESDE_model
    ├── test_sequence_metadata.csv             # gitignored, 9 MB     ← handoff to VAESDE_model
    └── sequence_dataset_summary.csv
```

---

## Dependencies

```bash
pip install pandas numpy
```

Python 3.10+ required.

---

## Reproducing from scratch

```bash
# 1. Place raw hourly OpenSky CSV files and the aircraft database in data/archive/

# 2. For each of the four dates, edit `date` at the top of preprocess_01 and run:
python3 preprocess_01_concat_daily.py   # repeat for 2022-06-06, -06-13, -06-20, -06-27

# 3. Build the GA aircraft registry:
python3 preprocess_02_filter_ga_aircraft.py \
    data/archive/aircraftDatabase-2022-06.csv \
    data/preprocess_02_output/aircraftDatabase-2022-06-FixedWingGA.csv

# 4. Filter and sort daily state files:
python3 preprocess_03_filter_sort_daily.py

# 5. Generate trajectory segments:
python3 preprocess_04_generate_trajectories.py

# 6. Convert to ENU coordinates:
python3 preprocess_05_convert_segments_to_enu.py

# 7. Drop outlier segments:
python3 preprocess_06_clean_enu_velocity_outliers.py

# 8. Train/test split:
python3 preprocess_07_split_train_test_by_segment.py

# 9. Build sequence dataset (produces the handoff files for VAESDE_model):
python3 preprocess_08_make_sequence_dataset.py
```

Approximate runtimes on a standard laptop:

| Step | Runtime |
|------|---------|
| preprocess_01 (×4 dates) | ~5 min/date |
| preprocess_02 | < 1 min |
| preprocess_03 | ~5 min/date |
| preprocess_04 | ~15–20 min total |
| preprocess_05 | ~10 min |
| preprocess_06 | ~5 min |
| preprocess_07 | ~15 min |
| preprocess_08 | ~5 min |

---

## Pipeline decisions (frozen)

> Do not change these parameters without deliberate justification.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Altitude source | geoaltitude (WGS-84) | Geometric altitude is physically meaningful and consistent |
| Altitude range | 152.4 m – 9144 m (500 – 30 000 ft) | Excludes on-ground taxi and high-altitude airspace above GA ceiling |
| Gap threshold | 60 s | Gaps > 60 s indicate an ADS-B coverage break |
| Speed-jump split | 250 kt (haversine implied speed) | Impossible for GA aircraft; flags GPS position jumps |
| Min segment duration | 5 minutes (300 s) | Removes takeoff/landing fragments too short for motion modelling |
| Sequence length | 30 steps (300 s) | Fixed-length VAE input window |
| Stride | 5 steps (50 s) | Sliding window step size |
| Features | E_m, N_m, vE_mps, vN_mps | Horizontal motion only |

---

## Final dataset statistics

| Metric | Value |
|--------|-------|
| Aircraft in GA registry | 119 666 |
| Aircraft observed flying (across 4 days) | 14 651 |
| Trajectory segments (after cleaning) | 114 493 |
| ADS-B pings (after cleaning) | 10 955 117 |
| Train sequences | 1 412 436 |
| Test sequences | 160 946 |

**Train / test split** (90 / 10 by segment_id, seed = 42):

| Split | Segments | Rows |
|-------|----------|------|
| Train | 103 043 (90.0%) | 9 841 364 |
| Test  | 11 450 (10.0%)  | 1 113 753 |

---

## Handoff to VAESDE_model

After running all 8 steps, copy these files into `VAESDE_model/data/`:

| File | Size | Description |
|---|---|---|
| `data/X_train.npy` | 1.36 GB | Train sequences, shape `(1 412 436, 30, 4)` |
| `data/X_test.npy` | 155 MB | Test sequences, shape `(160 946, 30, 4)` |
| `data/normalisation_mean.csv` | <1 KB | Per-feature mean (computed on train split only) |
| `data/normalisation_std.csv` | <1 KB | Per-feature std (computed on train split only) |
| `data/train_sequence_metadata.csv` | 80 MB | segment_id / start / end time per train sequence |
| `data/test_sequence_metadata.csv` | 9 MB | segment_id / start / end time per test sequence |

```bash
MODELLING_REPO=/path/to/VAESDE_model

cp data/X_train.npy                  $MODELLING_REPO/data/
cp data/X_test.npy                   $MODELLING_REPO/data/
cp data/normalisation_mean.csv       $MODELLING_REPO/data/
cp data/normalisation_std.csv        $MODELLING_REPO/data/
cp data/train_sequence_metadata.csv  $MODELLING_REPO/data/
cp data/test_sequence_metadata.csv   $MODELLING_REPO/data/
```
