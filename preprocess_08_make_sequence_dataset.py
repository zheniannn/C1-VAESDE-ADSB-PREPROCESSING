"""
preprocess_08_make_sequence_dataset.py

Converts cleaned train/test ENU trajectory CSVs into fixed-length sequence
arrays for VAE training.

Features: E_m, N_m, vE_mps, vN_mps  (horizontal motion only)
Sequence length: 30 timesteps (300 s at 10 s sampling)
Stride: 5 timesteps (new window every 50 s)

Normalisation statistics are computed on TRAIN only, then applied to both splits.
Sequences are created within each segment_id; no sequence crosses segment boundaries.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- Settings ---
FEATURES        = ["E_m", "N_m", "vE_mps", "vN_mps"]
SEQUENCE_LENGTH = 30
STRIDE          = 5
REQUIRED_COLS   = ["segment_id", "time"] + FEATURES

TRAIN_CSV = Path("data/train_segments_enu_clean.csv")
TEST_CSV  = Path("data/test_segments_enu_clean.csv")
OUT_DIR   = Path("data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_validate(path: Path, split: str) -> pd.DataFrame:
    print(f"Loading {split}: {path} ...")
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{split}] Missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    before = len(df)
    df.dropna(subset=REQUIRED_COLS, inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  [{split}] Dropped {dropped:,} rows with NaN in required columns")

    df.sort_values(["segment_id", "time"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  [{split}] {len(df):,} rows, {df['segment_id'].nunique():,} segments")
    return df


def make_sequences(
    df: pd.DataFrame,
    split: str,
    mean: pd.Series,
    std: pd.Series,
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
) -> tuple[np.ndarray, pd.DataFrame, int, int]:
    """
    Returns (X, metadata_df, segments_used, segments_skipped).
    X shape: [num_sequences, seq_len, num_features]
    """
    sequence_chunks: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    segments_used    = 0
    segments_skipped = 0
    seq_id           = 0

    for seg_id, group in df.groupby("segment_id", sort=True):
        group = group.sort_values("time")
        n = len(group)

        if n < seq_len:
            segments_skipped += 1
            continue

        segments_used += 1

        feat_norm = (group[FEATURES].values - mean.values) / std.values  # [n, 4]
        times     = group["time"].values                                   # [n]

        starts = np.arange(0, n - seq_len + 1, stride)     # [num_windows]
        idx    = starts[:, None] + np.arange(seq_len)       # [num_windows, seq_len]
        windows = feat_norm[idx]                             # [num_windows, seq_len, 4]
        sequence_chunks.append(windows)

        for i, start in enumerate(starts):
            end = start + seq_len
            metadata_rows.append({
                "sequence_id":                    seq_id,
                "split":                          split,
                "segment_id":                     seg_id,
                "start_time":                     int(times[start]),
                "end_time":                       int(times[end - 1]),
                "start_row_index_within_segment": int(start),
                "end_row_index_within_segment":   int(end - 1),
                "sequence_length":                seq_len,
                "stride":                         stride,
                "num_points_in_segment":          n,
            })
            seq_id += 1

    if sequence_chunks:
        X = np.concatenate(sequence_chunks, axis=0)
    else:
        X = np.empty((0, seq_len, len(FEATURES)), dtype=np.float64)

    meta = pd.DataFrame(metadata_rows)
    return X, meta, segments_used, segments_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load and validate
    train_df = load_and_validate(TRAIN_CSV, "train")
    test_df  = load_and_validate(TEST_CSV,  "test")

    # 5–6. Compute train-only normalisation statistics; guard against zero std
    mean = train_df[FEATURES].mean()
    std  = train_df[FEATURES].std()
    std  = std.where(std != 0.0, 1.0)

    # 7. Save normalisation statistics
    mean.to_frame("mean").to_csv(OUT_DIR / "normalisation_mean.csv")
    std.to_frame("std").to_csv(OUT_DIR  / "normalisation_std.csv")

    # 9. Create sequences
    print("\nCreating train sequences ...")
    X_train, meta_train, train_used, train_skipped = make_sequences(
        train_df, "train", mean, std
    )
    print("Creating test sequences ...")
    X_test,  meta_test,  test_used,  test_skipped  = make_sequences(
        test_df, "test", mean, std
    )

    # 10. Save arrays
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "X_test.npy",  X_test)

    # 11. Save metadata
    meta_train.to_csv(OUT_DIR / "train_sequence_metadata.csv", index=False)
    meta_test.to_csv(OUT_DIR  / "test_sequence_metadata.csv",  index=False)

    # 12. Create summary CSV
    def duration_stats(meta: pd.DataFrame) -> tuple[float, int, int]:
        if meta.empty:
            return 0.0, 0, 0
        d = (meta["end_time"] - meta["start_time"]).values
        return float(d.mean()), int(d.min()), int(d.max())

    tr_mean_d, tr_min_d, tr_max_d = duration_stats(meta_train)
    te_mean_d, te_min_d, te_max_d = duration_stats(meta_test)

    summary = pd.DataFrame([
        {
            "split":                      "train",
            "rows_loaded":                len(train_df),
            "segments_loaded":            train_df["segment_id"].nunique(),
            "segments_used":              train_used,
            "segments_skipped_too_short": train_skipped,
            "sequences_created":          len(X_train),
            "sequence_length":            SEQUENCE_LENGTH,
            "stride":                     STRIDE,
            "num_features":               len(FEATURES),
            "feature_names":              "|".join(FEATURES),
            "mean_sequence_duration_s":   tr_mean_d,
            "min_sequence_duration_s":    tr_min_d,
            "max_sequence_duration_s":    tr_max_d,
        },
        {
            "split":                      "test",
            "rows_loaded":                len(test_df),
            "segments_loaded":            test_df["segment_id"].nunique(),
            "segments_used":              test_used,
            "segments_skipped_too_short": test_skipped,
            "sequences_created":          len(X_test),
            "sequence_length":            SEQUENCE_LENGTH,
            "stride":                     STRIDE,
            "num_features":               len(FEATURES),
            "feature_names":              "|".join(FEATURES),
            "mean_sequence_duration_s":   te_mean_d,
            "min_sequence_duration_s":    te_min_d,
            "max_sequence_duration_s":    te_max_d,
        },
    ])
    summary.to_csv(OUT_DIR / "sequence_dataset_summary.csv", index=False)

    # 13. Print report
    n_feat = len(FEATURES)
    norm_mean = X_train.reshape(-1, n_feat).mean(axis=0)
    norm_std  = X_train.reshape(-1, n_feat).std(axis=0)

    print()
    print("=" * 62)
    print("SEQUENCE DATASET REPORT")
    print("=" * 62)
    print(f"  Train rows loaded               : {len(train_df):>12,}")
    print(f"  Test rows loaded                : {len(test_df):>12,}")
    print(f"  Train segments loaded           : {train_df['segment_id'].nunique():>12,}")
    print(f"  Test segments loaded            : {test_df['segment_id'].nunique():>12,}")
    print(f"  Train sequences created         : {len(X_train):>12,}")
    print(f"  Test sequences created          : {len(X_test):>12,}")
    print(f"  Train segments skipped (short)  : {train_skipped:>12,}")
    print(f"  Test segments skipped (short)   : {test_skipped:>12,}")
    print(f"  X_train shape                   : {str(X_train.shape):>12}")
    print(f"  X_test shape                    : {str(X_test.shape):>12}")
    print(f"  NaN in X_train                  : {str(bool(np.isnan(X_train).any())):>12}")
    print(f"  NaN in X_test                   : {str(bool(np.isnan(X_test).any())):>12}")
    print()
    print(f"  Normalised train mean (→ ~0)    : {np.round(norm_mean, 4)}")
    print(f"  Normalised train std  (→ ~1)    : {np.round(norm_std,  4)}")
    print()
    print("  First 5 rows — train_sequence_metadata:")
    print(meta_train.head(5).to_string(index=False))
    print()
    print("  First 5 rows — test_sequence_metadata:")
    print(meta_test.head(5).to_string(index=False))

    # Sanity checks
    print()
    print("=" * 62)
    print("SANITY CHECKS")
    print("=" * 62)

    train_segs = set(meta_train["segment_id"]) if not meta_train.empty else set()
    test_segs  = set(meta_test["segment_id"])  if not meta_test.empty  else set()

    checks = [
        ("X_train.ndim == 3",
            X_train.ndim == 3),
        ("X_test.ndim == 3",
            X_test.ndim == 3),
        (f"X_train.shape[1:] == ({SEQUENCE_LENGTH}, {n_feat})",
            X_train.shape[1:] == (SEQUENCE_LENGTH, n_feat)),
        (f"X_test.shape[1:]  == ({SEQUENCE_LENGTH}, {n_feat})",
            X_test.shape[1:]  == (SEQUENCE_LENGTH, n_feat)),
        ("No NaN in X_train",
            not bool(np.isnan(X_train).any())),
        ("No NaN in X_test",
            not bool(np.isnan(X_test).any())),
        ("No Inf in X_train",
            not bool(np.isinf(X_train).any())),
        ("No Inf in X_test",
            not bool(np.isinf(X_test).any())),
        ("Train/test segment_ids do not overlap",
            len(train_segs & test_segs) == 0),
        ("Metadata rows == sequences (train)",
            len(meta_train) == len(X_train)),
        ("Metadata rows == sequences (test)",
            len(meta_test)  == len(X_test)),
    ]

    all_passed = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if not result:
            all_passed = False

    print()
    print("All sanity checks passed." if all_passed else "WARNING: some checks FAILED.")
    print()
    print("Output files written to data/:")
    for fname in [
        "X_train.npy", "X_test.npy",
        "train_sequence_metadata.csv", "test_sequence_metadata.csv",
        "normalisation_mean.csv", "normalisation_std.csv",
        "sequence_dataset_summary.csv",
    ]:
        p = OUT_DIR / fname
        size = p.stat().st_size / 1e6
        print(f"  {fname:<40} {size:6.1f} MB")


if __name__ == "__main__":
    main()
