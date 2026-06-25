# VAESDE_preprocess

A Python pipeline for preprocessing ADS-B trajectories from the OpenSky Network into normalised sequence arrays for VAE-based trajectory modelling.

---

## Why this exists

The **VAESDE_model** repository expects fixed-length, normalised ENU motion sequences as input. This pipeline takes raw OpenSky state vectors for GA fixed-wing aircraft and produces those sequences through eight cleaning and transformation stages.

Four study dates: **2022-06-06, 2022-06-13, 2022-06-20, 2022-06-27**.

---

## Pipeline

```
Raw hourly ADS-B CSVs
    → [01] concat daily files
    → [02] filter aircraft DB to GA fixed-wing (Cessna, Piper, Cirrus, Diamond)
    → [03] filter daily states to GA icao24s, sort by (icao24, time)
    → [04] build trajectory segments (gap/speed splits, min-duration filter)
    → [05] convert lat/lon/alt → ENU coordinates + per-axis velocities
    → [06] drop segments with speed > 150 m/s or accel > 10 m/s²
    → [07] 90/10 train/test split by segment_id (seed = 42)
    → [08] sliding-window sequences → X_train.npy, X_test.npy
```

---

## Key result

| Metric | Value |
|---|---|
| Aircraft in GA registry | 119 666 |
| Aircraft observed flying (4 days) | 14 651 |
| Trajectory segments after cleaning | 114 493 |
| ADS-B pings after cleaning | 10 955 117 |
| Train sequences `(1 412 436, 30, 4)` | 1 412 436 |
| Test sequences `(160 946, 30, 4)` | 160 946 |

**Train / test split** — 90/10 by segment_id, no segment crosses splits:

| Split | Segments | Rows |
|---|---|---|
| Train | 103 043 (90.0%) | 9 841 364 |
| Test | 11 450 (10.0%) | 1 113 753 |

---

## Quick start

```bash
pip install -e .                   # install package and dependencies
pytest                             # run smoke tests (5 tests)

python3 scripts/run_ingest.py      # stages 01–03: ~25 min total
python3 scripts/run_pipeline.py    # stages 04–06: ~30 min total
python3 scripts/run_dataset.py     # stages 07–08: ~25 min total
```

> Requires Python 3.10+. All paths, dates, and thresholds are in `configs/pipeline.yaml`.
> Raw data files must be placed in `data/archive/` before running.

---

## Pipeline stages

| Script | Stages | What it does |
|---|---|---|
| `scripts/run_ingest.py` | 01–03 | Concat hourly files, filter aircraft DB, filter daily states |
| `scripts/run_pipeline.py` | 04–06 | Generate segments, convert to ENU, drop outliers |
| `scripts/run_dataset.py` | 07–08 | Train/test split, build normalised sequence arrays |

Approximate runtimes on a standard laptop:

| Script | Runtime |
|---|---|
| `run_ingest.py` | ~40 min (4 dates × ~5 min concat + ~5 min filter) |
| `run_pipeline.py` | ~30 min |
| `run_dataset.py` | ~25 min |

---

## Pipeline decisions

> These parameters are frozen. Change them only with deliberate justification.

| Parameter | Value | Rationale |
|---|---|---|
| Altitude source | geoaltitude (WGS-84) | Geometric altitude is physically consistent |
| Altitude range | 152.4 m – 9 144 m (500 – 30 000 ft) | Excludes ground taxi and high-altitude airspace above GA ceiling |
| Gap threshold | 60 s | Gaps > 60 s indicate an ADS-B coverage break |
| Speed-jump split | 250 kt (haversine implied) | Impossible for GA aircraft; flags GPS position jumps |
| Min segment duration | 300 s (5 min) | Removes takeoff/landing fragments too short for motion modelling |
| Sequence length | 30 steps (300 s at 10 s sampling) | Fixed-length VAE input window |
| Stride | 5 steps (50 s) | Sliding window overlap |
| Features | E_m, N_m, vE_mps, vN_mps | Horizontal motion only |

---

## Repository layout

```
VAESDE_preprocess/
├── configs/
│   └── pipeline.yaml          # dates, thresholds, paths, split settings
├── scripts/
│   ├── run_ingest.py          # stages 01–03
│   ├── run_pipeline.py        # stages 04–06
│   └── run_dataset.py         # stages 07–08
├── src/
│   └── adsb_preprocess/       # installable package (pip install -e .)
│       ├── geo.py             # WGS-84 conversions, haversine
│       ├── filters.py         # GA aircraft brand/model filter
│       ├── trajectories.py    # segment generation
│       ├── enu.py             # ENU coordinate conversion
│       ├── cleaning.py        # outlier removal
│       ├── splitting.py       # train/test split helpers
│       ├── sequences.py       # sliding-window sequence builder
│       ├── ingest.py          # stage 01–03 orchestration
│       ├── pipeline.py        # stage 04–06 orchestration
│       ├── dataset.py         # stage 07–08 orchestration
│       └── io_utils.py        # config loader
├── tests/
│   └── test_smoke.py
└── data/                      # gitignored (~92 GB when fully populated)
    ├── archive/               # raw source files (place here before running)
    └── ...                    # all intermediate and final outputs
```

---

## Handoff to VAESDE_model

After all three scripts complete, copy these files into `VAESDE_model/data/`:

| File | Size | Description |
|---|---|---|
| `data/X_train.npy` | 1.36 GB | Train sequences, shape `(1 412 436, 30, 4)` |
| `data/X_test.npy` | 155 MB | Test sequences, shape `(160 946, 30, 4)` |
| `data/normalisation_mean.csv` | < 1 KB | Per-feature mean (train split only) |
| `data/normalisation_std.csv` | < 1 KB | Per-feature std (train split only) |
| `data/train_sequence_metadata.csv` | 80 MB | Sequence provenance for train split |
| `data/test_sequence_metadata.csv` | 9 MB | Sequence provenance for test split |

```bash
MODELLING_REPO=/path/to/VAESDE_model

cp data/X_train.npy                  $MODELLING_REPO/data/
cp data/X_test.npy                   $MODELLING_REPO/data/
cp data/normalisation_mean.csv       $MODELLING_REPO/data/
cp data/normalisation_std.csv        $MODELLING_REPO/data/
cp data/train_sequence_metadata.csv  $MODELLING_REPO/data/
cp data/test_sequence_metadata.csv   $MODELLING_REPO/data/
```

---

## Limitations

- Data covers four specific Mondays in June 2022 only. Seasonal or weekday effects are not accounted for.
- The GA aircraft filter uses manufacturer/model regex patterns; edge cases in OpenSky's database may cause false positives or false negatives.
- ENU origin is set per-segment at the first ping; segments from different geographic regions are not aligned to a common frame.
- No real-time or streaming capability — the pipeline is designed for offline batch processing.
