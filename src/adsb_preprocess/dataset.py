"""Orchestration for dataset stages 07–08."""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from adsb_preprocess.splitting import get_split, route_rows, compute_split_stats
from adsb_preprocess.sequences import load_and_validate, make_sequences

REQUIRED_COLS = [
    "segment_id", "time",
    "E_m", "N_m", "U_m",
    "vE_mps", "vN_mps", "vU_mps",
    "speed_mps", "accel_mps2",
]


def run_stage_07(cfg: dict) -> None:
    """Split the cleaned ENU dataset into train/test by segment_id."""
    sp_cfg      = cfg["split"]
    in_path     = os.path.join(cfg["paths"]["stage_06_out"], "trajectory_segments_enu_clean.csv")
    train_csv   = cfg["paths"]["train_csv"]
    test_csv    = cfg["paths"]["test_csv"]
    summary_csv = os.path.join(cfg["paths"]["dataset_out"], "split_summary.csv")

    missing = [c for c in REQUIRED_COLS if c not in pd.read_csv(in_path, nrows=0).columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\nPass 1 — determining split ...")
    train_ids, test_ids, n_segs_total = get_split(in_path, sp_cfg["train_frac"], sp_cfg["seed"])
    print(f"  Total: {n_segs_total:,}  Train: {len(train_ids):,}  Test: {len(test_ids):,}")

    print("\nPass 2 — routing rows ...")
    n_rows_train, n_rows_test = route_rows(in_path, train_ids, test_ids, train_csv, test_csv)
    n_rows_total = n_rows_train + n_rows_test
    print(f"  Train rows: {n_rows_train:,}  Test rows: {n_rows_test:,}")

    print("\nPass 3 — computing stats ...")
    tr_stats = compute_split_stats(train_csv)
    te_stats = compute_split_stats(test_csv)

    rows = []
    for label, st, n_rows in (("train", tr_stats, n_rows_train), ("test", te_stats, n_rows_test)):
        sp  = st["sp"];  ac = st["ac"];  dur = st["durations"]
        rows.append({
            "split":           label,
            "n_rows":          st["n_rows"],
            "n_segments":      st["n_segments"],
            "pct_rows":        round(100 * st["n_rows"] / n_rows_total, 2),
            "pct_segments":    round(100 * st["n_segments"] / n_segs_total, 2),
            "mean_speed_mps":  round(float(sp.mean()), 4),
            "p95_speed_mps":   round(float(np.percentile(sp, 95)), 4),
            "max_speed_mps":   round(float(sp.max()), 4),
            "mean_accel_mps2": round(float(ac.mean()), 4),
            "p95_accel_mps2":  round(float(np.percentile(ac, 95)), 4),
            "max_accel_mps2":  round(float(ac.max()), 4),
            "mean_duration_s": round(float(dur.mean()), 1),
            "p95_duration_s":  round(float(np.percentile(dur, 95)), 1),
        })
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    overlap = train_ids & test_ids
    print(f"\n  Overlap segment_ids: {len(overlap)}  {'OK' if not overlap else 'FAIL'}")
    print(f"[07] Saved → {train_csv}, {test_csv}")


def run_stage_08(cfg: dict) -> None:
    """Create normalised fixed-length sequence arrays for VAE training."""
    sq_cfg  = cfg["sequences"]
    out_dir = Path(cfg["paths"]["dataset_out"])

    features = sq_cfg["features"]
    seq_len  = sq_cfg["sequence_length"]
    stride   = sq_cfg["stride"]
    required = ["segment_id", "time"] + features

    train_df = load_and_validate(Path(cfg["paths"]["train_csv"]), "train", required)
    test_df  = load_and_validate(Path(cfg["paths"]["test_csv"]),  "test",  required)

    mean = train_df[features].mean()
    std  = train_df[features].std().where(lambda s: s != 0.0, other=1.0)

    mean.to_frame("mean").to_csv(out_dir / "normalisation_mean.csv")
    std.to_frame("std").to_csv(out_dir   / "normalisation_std.csv")

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
    print(f"\n  X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"  Normalised mean (→~0): {np.round(norm_mean, 4)}")
    print(f"  Normalised std  (→~1): {np.round(norm_std,  4)}")

    checks = [
        ("X_train.ndim == 3",               X_train.ndim == 3),
        ("X_test.ndim == 3",                X_test.ndim == 3),
        (f"shape[1:] == ({seq_len}, {n_feat})", X_train.shape[1:] == (seq_len, n_feat)),
        ("No NaN in X_train",               not bool(np.isnan(X_train).any())),
        ("No NaN in X_test",                not bool(np.isnan(X_test).any())),
        ("No Inf in X_train",               not bool(np.isinf(X_train).any())),
        ("Train/test segs do not overlap",
         len(set(meta_train["segment_id"]) & set(meta_test["segment_id"])) == 0),
        ("Metadata rows == sequences (train)", len(meta_train) == len(X_train)),
        ("Metadata rows == sequences (test)",  len(meta_test)  == len(X_test)),
    ]
    all_passed = all(r for _, r in checks)
    print("\nSanity checks:")
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    print("All checks passed." if all_passed else "WARNING: some checks FAILED.")

    pd.DataFrame([
        {"split": "train", "sequences_created": len(X_train), "segments_used": train_used,
         "segments_skipped": train_skipped, "sequence_length": seq_len, "stride": stride,
         "num_features": n_feat, "feature_names": "|".join(features)},
        {"split": "test",  "sequences_created": len(X_test),  "segments_used": test_used,
         "segments_skipped": test_skipped, "sequence_length": seq_len, "stride": stride,
         "num_features": n_feat, "feature_names": "|".join(features)},
    ]).to_csv(out_dir / "sequence_dataset_summary.csv", index=False)

    print(f"[08] Saved arrays and metadata → {out_dir}")
