"""Train/test splitting helpers for ENU trajectory segment datasets."""

import numpy as np
import pandas as pd


def get_split(path: str, train_frac: float, seed: int, chunk_size: int = 100_000) -> tuple[set, set, int]:
    """
    Stream path to collect unique segment_ids, then split them reproducibly.

    Returns (train_ids, test_ids, n_total).
    """
    seen = set()
    for chunk in pd.read_csv(path, usecols=["segment_id"], chunksize=chunk_size):
        seen.update(chunk["segment_id"].tolist())

    seg_ids = np.array(sorted(seen), dtype=np.int64)
    n_total = len(seg_ids)
    rng     = np.random.default_rng(seed=seed)
    rng.shuffle(seg_ids)
    n_train   = int(n_total * train_frac)
    train_ids = set(seg_ids[:n_train].tolist())
    test_ids  = set(seg_ids[n_train:].tolist())
    return train_ids, test_ids, n_total


def route_rows(
    path:      str,
    train_ids: set,
    test_ids:  set,
    train_csv: str,
    test_csv:  str,
    chunk_size: int = 100_000,
) -> tuple[int, int]:
    """
    Stream path and write each row to train_csv or test_csv based on segment_id.

    Returns (n_train_rows, n_test_rows).
    """
    n_train = n_test = 0
    first_train = first_test = True
    n_chunks = 0

    with open(train_csv, "w") as f_tr, open(test_csv, "w") as f_te:
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
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


def compute_split_stats(path: str, chunk_size: int = 100_000, subsample_every: int = 10) -> dict:
    """
    Stream path reading only speed/accel/time columns; return compact stats dict.

    Percentiles are estimated from a 1-in-subsample_every subsample.
    """
    STAT_COLS = ["segment_id", "time", "speed_mps", "accel_mps2"]
    n_rows   = 0
    sp_samps = []
    ac_samps = []
    seg_min: dict[int, float] = {}
    seg_max: dict[int, float] = {}

    for chunk in pd.read_csv(path, usecols=STAT_COLS, chunksize=chunk_size):
        n_rows += len(chunk)
        sub = chunk.iloc[::subsample_every]
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

    sp        = np.concatenate(sp_samps)
    ac        = np.concatenate(ac_samps)
    durations = np.array([seg_max[s] - seg_min[s] for s in seg_min], dtype=np.float32)
    return {"n_rows": n_rows, "n_segments": len(seg_min), "sp": sp, "ac": ac, "durations": durations}
