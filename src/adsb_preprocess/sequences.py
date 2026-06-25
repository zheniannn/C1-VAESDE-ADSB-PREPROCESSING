"""Fixed-length sequence dataset builder for VAE training."""

import numpy as np
import pandas as pd
from pathlib import Path


def load_and_validate(path: Path, split: str, required_cols: list[str]) -> pd.DataFrame:
    """Load a split CSV, validate required columns, and sort by (segment_id, time)."""
    print(f"Loading {split}: {path} ...")
    df = pd.read_csv(path)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{split}] Missing required columns: {missing}")

    df = df[required_cols].copy()
    before = len(df)
    df.dropna(subset=required_cols, inplace=True)
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
    features: list[str],
    mean: pd.Series,
    std: pd.Series,
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, pd.DataFrame, int, int]:
    """
    Slide fixed-length windows over each segment and normalise features.

    Returns (X, metadata_df, segments_used, segments_skipped).
    X shape: [num_sequences, seq_len, num_features].
    """
    sequence_chunks: list[np.ndarray] = []
    metadata_rows:   list[dict]       = []
    segments_used    = 0
    segments_skipped = 0
    seq_id           = 0

    for seg_id, group in df.groupby("segment_id", sort=True):
        group = group.sort_values("time")
        n     = len(group)

        if n < seq_len:
            segments_skipped += 1
            continue

        segments_used += 1

        feat_norm = (group[features].values - mean.values) / std.values
        times     = group["time"].values

        starts  = np.arange(0, n - seq_len + 1, stride)
        idx     = starts[:, None] + np.arange(seq_len)
        windows = feat_norm[idx]
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

    X = (
        np.concatenate(sequence_chunks, axis=0)
        if sequence_chunks
        else np.empty((0, seq_len, len(features)), dtype=np.float64)
    )
    return X, pd.DataFrame(metadata_rows), segments_used, segments_skipped
