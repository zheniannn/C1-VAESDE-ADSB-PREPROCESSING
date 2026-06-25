"""
Dataset builder — stages 07–08.

  Stage 07: Split cleaned ENU segments into train (90%) and test (10%) by segment_id.
  Stage 08: Slide fixed-length windows over each split to create normalised sequence arrays.

Outputs land in paths.train_csv, paths.test_csv, and paths.dataset_out
as configured in configs/pipeline.yaml.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH  = os.path.join(_PROJECT_ROOT, "configs", "pipeline.yaml")

CHUNK_SIZE      = 100_000
SUBSAMPLE_EVERY = 10

REQUIRED_COLS = [
    "segment_id", "time",
    "E_m", "N_m", "U_m",
    "vE_mps", "vN_mps", "vU_mps",
    "speed_mps", "accel_mps2",
]

from adsb_preprocess.io_utils  import load_config
from adsb_preprocess.sequences import load_and_validate, make_sequences


# ---------------------------------------------------------------------------
# Stage 07 helpers
# ---------------------------------------------------------------------------

def _get_split(path: str, train_frac: float, seed: int) -> tuple[set, set, int]:
    seen = set()
    for chunk in pd.read_csv(path, usecols=["segment_id"], chunksize=CHUNK_SIZE):
        seen.update(chunk["segment_id"].tolist())

    seg_ids = np.array(sorted(seen), dtype=np.int64)
    n_total = len(seg_ids)
    rng     = np.random.default_rng(seed=seed)
    rng.shuffle(seg_ids)
    n_train   = int(n_total * train_frac)
    train_ids = set(seg_ids[:n_train].tolist())
    test_ids  = set(seg_ids[n_train:].tolist())
    return train_ids, test_ids, n_total


def _route(path: str, train_ids: set, test_ids: set, train_csv: str, test_csv: str) -> tuple[int, int]:
    n_train = n_test = 0
    first_train = first_test = True
    n_chunks = 0

    with open(train_csv, "w") as f_tr, open(test_csv, "w") as f_te:
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            n_chunks += 1
            if n_chunks % 20 == 0:
                print(f"  ... chunk {n_chunks:>4}  routed: {n_train+n_test:>10,}")

            tr = chunk[chunk["segment_id"].isin(train_ids)]
            te = chunk[chunk["segment_id"].isin(test_ids)]

            if not tr.empty:
                tr.to_csv(f_tr, index=False, header=first_train)
                first_train = False
                n_train += len(tr)
            if not te.empty:
                te.to_csv(f_te, index=False, header=first_test)
                first_test = False
                n_test += len(te)

    return n_train, n_test


def _compute_stats(path: str) -> dict:
    STAT_COLS = ["segment_id", "time", "speed_mps", "accel_mps2"]
    n_rows   = 0
    sp_samps = []
    ac_samps = []
    seg_min  = {}
    seg_max  = {}

    for chunk in pd.read_csv(path, usecols=STAT_COLS, chunksize=CHUNK_SIZE):
        n_rows += len(chunk)
        sub = chunk.iloc[::SUBSAMPLE_EVERY]
        sp_samps.append(sub["speed_mps"].values.astype(np.float32))
        ac_samps.append(sub["accel_mps2"].values.astype(np.float32))

        grp = chunk.groupby("segment_id", sort=False)["time"].agg(t_min="min", t_max="max")
        for seg_id, t_min_val, t_max_val in zip(grp.index, grp["t_min"].values, grp["t_max"].values):
            if seg_id in seg_min:
                if t_min_val < seg_min[seg_id]:
                    seg_min[seg_id] = float(t_min_val)
                if t_max_val > seg_max[seg_id]:
                    seg_max[seg_id] = float(t_max_val)
            else:
                seg_min[seg_id] = float(t_min_val)
                seg_max[seg_id] = float(t_max_val)

    sp = np.concatenate(sp_samps)
    ac = np.concatenate(ac_samps)
    durations = np.array([seg_max[s] - seg_min[s] for s in seg_min], dtype=np.float32)
    return {"n_rows": n_rows, "n_segments": len(seg_min), "sp": sp, "ac": ac, "durations": durations}


# ---------------------------------------------------------------------------
# Stage entry points
# ---------------------------------------------------------------------------

def run_split():
    """Split the cleaned ENU dataset into train/test by segment_id."""
    cfg        = load_config(_CONFIG_PATH)
    sp_cfg     = cfg["split"]
    in_path    = os.path.join(cfg["paths"]["stage_06_out"], "trajectory_segments_enu_clean.csv")
    train_csv  = cfg["paths"]["train_csv"]
    test_csv   = cfg["paths"]["test_csv"]
    summary_csv = os.path.join(cfg["paths"]["dataset_out"], "split_summary.csv")

    header = pd.read_csv(in_path, nrows=0)
    missing = [c for c in REQUIRED_COLS if c not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\nPass 1 — determining split ...")
    train_ids, test_ids, n_segs_total = _get_split(in_path, sp_cfg["train_frac"], sp_cfg["seed"])
    print(f"  Total: {n_segs_total:,}  Train: {len(train_ids):,}  Test: {len(test_ids):,}")

    print("\nPass 2 — routing rows ...")
    n_rows_train, n_rows_test = _route(in_path, train_ids, test_ids, train_csv, test_csv)
    n_rows_total = n_rows_train + n_rows_test
    print(f"  Train rows: {n_rows_train:,}  Test rows: {n_rows_test:,}")

    print("\nPass 3 — computing stats ...")
    tr_stats = _compute_stats(train_csv)
    te_stats = _compute_stats(test_csv)

    rows = []
    for label, st, n_rows in (("train", tr_stats, n_rows_train), ("test", te_stats, n_rows_test)):
        sp  = st["sp"];  ac = st["ac"];  dur = st["durations"]
        rows.append({
            "split":            label,
            "n_rows":           st["n_rows"],
            "n_segments":       st["n_segments"],
            "pct_rows":         round(100 * st["n_rows"] / n_rows_total, 2),
            "pct_segments":     round(100 * st["n_segments"] / n_segs_total, 2),
            "mean_speed_mps":   round(float(sp.mean()), 4),
            "p95_speed_mps":    round(float(np.percentile(sp, 95)), 4),
            "max_speed_mps":    round(float(sp.max()), 4),
            "mean_accel_mps2":  round(float(ac.mean()), 4),
            "p95_accel_mps2":   round(float(np.percentile(ac, 95)), 4),
            "max_accel_mps2":   round(float(ac.max()), 4),
            "mean_duration_s":  round(float(dur.mean()), 1),
            "p95_duration_s":   round(float(np.percentile(dur, 95)), 1),
        })
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    overlap = train_ids & test_ids
    print(f"\n  Overlap segment_ids: {len(overlap)}  {'OK' if not overlap else 'FAIL'}")
    print(f"[07] Saved → {train_csv}, {test_csv}")


def run_make_sequences():
    """Create normalised fixed-length sequence arrays for VAE training."""
    cfg      = load_config(_CONFIG_PATH)
    sq_cfg   = cfg["sequences"]
    out_dir  = Path(cfg["paths"]["dataset_out"])

    features = sq_cfg["features"]
    seq_len  = sq_cfg["sequence_length"]
    stride   = sq_cfg["stride"]
    required = ["segment_id", "time"] + features

    train_csv = Path(cfg["paths"]["train_csv"])
    test_csv  = Path(cfg["paths"]["test_csv"])

    train_df = load_and_validate(train_csv, "train", required)
    test_df  = load_and_validate(test_csv,  "test",  required)

    mean = train_df[features].mean()
    std  = train_df[features].std()
    std  = std.where(std != 0.0, 1.0)

    mean.to_frame("mean").to_csv(out_dir / "normalisation_mean.csv")
    std.to_frame("std").to_csv(out_dir  / "normalisation_std.csv")

    print("\nCreating train sequences ...")
    X_train, meta_train, train_used, train_skipped = make_sequences(
        train_df, "train", features, mean, std, seq_len, stride
    )
    print("Creating test sequences ...")
    X_test, meta_test, test_used, test_skipped = make_sequences(
        test_df, "test", features, mean, std, seq_len, stride
    )

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy",  X_test)
    meta_train.to_csv(out_dir / "train_sequence_metadata.csv", index=False)
    meta_test.to_csv(out_dir  / "test_sequence_metadata.csv",  index=False)

    n_feat    = len(features)
    norm_mean = X_train.reshape(-1, n_feat).mean(axis=0)
    norm_std  = X_train.reshape(-1, n_feat).std(axis=0)

    print(f"\n  X_train shape: {X_train.shape}   X_test shape: {X_test.shape}")
    print(f"  Normalised train mean (→~0): {np.round(norm_mean, 4)}")
    print(f"  Normalised train std  (→~1): {np.round(norm_std,  4)}")
    print(f"  NaN in X_train: {bool(np.isnan(X_train).any())}")
    print(f"  NaN in X_test:  {bool(np.isnan(X_test).any())}")

    # Sanity checks
    checks = [
        ("X_train.ndim == 3",                X_train.ndim == 3),
        ("X_test.ndim == 3",                 X_test.ndim == 3),
        (f"X_train.shape[1:] == ({seq_len}, {n_feat})", X_train.shape[1:] == (seq_len, n_feat)),
        ("No NaN in X_train",                not bool(np.isnan(X_train).any())),
        ("No NaN in X_test",                 not bool(np.isnan(X_test).any())),
        ("No Inf in X_train",                not bool(np.isinf(X_train).any())),
        ("Train/test segs do not overlap",
         len(set(meta_train["segment_id"]) & set(meta_test["segment_id"])) == 0),
        ("Metadata rows == sequences (train)", len(meta_train) == len(X_train)),
        ("Metadata rows == sequences (test)",  len(meta_test)  == len(X_test)),
    ]
    all_passed = True
    print("\nSanity checks:")
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if not result:
            all_passed = False
    print("All checks passed." if all_passed else "WARNING: some checks FAILED.")

    summary_csv = out_dir / "sequence_dataset_summary.csv"
    pd.DataFrame([
        {"split": "train", "sequences_created": len(X_train), "segments_used": train_used,
         "segments_skipped": train_skipped, "sequence_length": seq_len, "stride": stride,
         "num_features": n_feat, "feature_names": "|".join(features)},
        {"split": "test",  "sequences_created": len(X_test),  "segments_used": test_used,
         "segments_skipped": test_skipped, "sequence_length": seq_len, "stride": stride,
         "num_features": n_feat, "feature_names": "|".join(features)},
    ]).to_csv(summary_csv, index=False)

    print(f"[08] Saved X_train.npy, X_test.npy, metadata CSVs → {out_dir}")


def main():
    print("=== Stage 07: Train/test split ===", flush=True)
    run_split()
    print("\n=== Stage 08: Make sequence dataset ===", flush=True)
    run_make_sequences()
    print("\nDataset complete.", flush=True)


if __name__ == "__main__":
    main()
