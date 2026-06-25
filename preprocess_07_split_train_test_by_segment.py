"""
preprocess_07_split_train_test_by_segment.py

Splits the cleaned ENU trajectory dataset into train and test sets using a
90/10 split by unique segment_id.  No segment appears in both splits.

Memory strategy — three lean passes:
  Pass 1 : stream only segment_id column → collect unique IDs → determine split.
  Pass 2 : stream full CSV → route rows to train/test files.
           Zero accumulation: peak RAM = one chunk (~30 MB).
  Pass 3 : stream the (much smaller) output files → compute stats from
           4 slim columns; subsample 1-in-10 for percentile estimation.

Input : data/preprocess_06_output/trajectory_segments_enu_clean.csv
Output: data/train_segments_enu_clean.csv
        data/test_segments_enu_clean.csv
        data/split_summary.csv
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
INPUT_CSV   = "data/preprocess_06_output/trajectory_segments_enu_clean.csv"
TRAIN_CSV   = "data/train_segments_enu_clean.csv"
TEST_CSV    = "data/test_segments_enu_clean.csv"
SUMMARY_CSV = "data/split_summary.csv"

TRAIN_FRAC      = 0.90
SEED            = 42
CHUNK_SIZE      = 100_000
SUBSAMPLE_EVERY = 10

REQUIRED_COLS = [
    "segment_id", "time",
    "E_m", "N_m", "U_m",
    "vE_mps", "vN_mps", "vU_mps",
    "speed_mps", "accel_mps2",
]
MAX_SPEED_MPS  = 150.0
MAX_ACCEL_MPS2 = 10.0


# ---------------------------------------------------------------------------
# Pass 1 — unique segment_ids via streaming
# ---------------------------------------------------------------------------
def get_split(path: str) -> tuple[set, set, int]:
    seen = set()
    for chunk in pd.read_csv(path, usecols=["segment_id"], chunksize=CHUNK_SIZE):
        seen.update(chunk["segment_id"].tolist())

    seg_ids = np.array(sorted(seen), dtype=np.int64)
    n_total = len(seg_ids)

    rng = np.random.default_rng(seed=SEED)
    rng.shuffle(seg_ids)
    n_train   = int(n_total * TRAIN_FRAC)
    train_ids = set(seg_ids[:n_train].tolist())
    test_ids  = set(seg_ids[n_train:].tolist())
    return train_ids, test_ids, n_total


# ---------------------------------------------------------------------------
# Pass 2 — pure routing, zero accumulation
# ---------------------------------------------------------------------------
def route(path: str, train_ids: set, test_ids: set) -> tuple[int, int]:
    n_train = n_test = 0
    first_train = first_test = True
    n_chunks = 0

    with open(TRAIN_CSV, "w") as f_tr, open(TEST_CSV, "w") as f_te:
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


# ---------------------------------------------------------------------------
# Pass 3 — stats from output files (slim columns only)
# ---------------------------------------------------------------------------
def compute_stats(path: str) -> dict:
    """
    Stream a split CSV reading only 4 columns.
    Returns compact stats dict.
    """
    STAT_COLS = ["segment_id", "time", "speed_mps", "accel_mps2"]
    n_rows   = 0
    sp_samps = []       # subsampled speed values (float32)
    ac_samps = []       # subsampled accel values (float32)
    seg_min  = {}       # seg_id -> min_time
    seg_max  = {}       # seg_id -> max_time

    for chunk in pd.read_csv(path, usecols=STAT_COLS, chunksize=CHUNK_SIZE):
        n_rows += len(chunk)

        # Subsampled values for percentile estimation
        sub = chunk.iloc[::SUBSAMPLE_EVERY]
        sp_samps.append(sub["speed_mps"].values.astype(np.float32))
        ac_samps.append(sub["accel_mps2"].values.astype(np.float32))

        # Per-segment time range — iterate over aligned arrays (avoids itertuples unpacking issues)
        grp = chunk.groupby("segment_id", sort=False)["time"].agg(t_min="min", t_max="max")
        for seg_id, t_min_val, t_max_val in zip(
            grp.index, grp["t_min"].values, grp["t_max"].values
        ):
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
    durations = np.array(
        [seg_max[s] - seg_min[s] for s in seg_min], dtype=np.float32
    )

    return {
        "n_rows"      : n_rows,
        "n_segments"  : len(seg_min),
        "sp"          : sp,
        "ac"          : ac,
        "durations"   : durations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  split_train_test_by_segment.py")
    print(f"  Input : {INPUT_CSV}")
    print(SEP)

    # Check required columns
    header = pd.read_csv(INPUT_CSV, nrows=0)
    missing = [c for c in REQUIRED_COLS if c not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print("  Required columns: OK")

    # --- Pass 1 ---
    print("\nPass 1 — determining split ...")
    train_ids, test_ids, n_segs_total = get_split(INPUT_CSV)
    print(f"  Total segments : {n_segs_total:,}")
    print(f"  Train          : {len(train_ids):,}  ({100*len(train_ids)/n_segs_total:.1f}%)")
    print(f"  Test           : {len(test_ids):,}  ({100*len(test_ids)/n_segs_total:.1f}%)")

    # --- Pass 2 ---
    print("\nPass 2 — routing rows ...")
    n_rows_train, n_rows_test = route(INPUT_CSV, train_ids, test_ids)
    n_rows_total = n_rows_train + n_rows_test
    print(f"  Train rows : {n_rows_train:,}")
    print(f"  Test  rows : {n_rows_test:,}")

    # --- Pass 3 ---
    print("\nPass 3 — computing stats from output files ...")
    print(f"  train ...")
    tr_stats = compute_stats(TRAIN_CSV)
    print(f"  test ...")
    te_stats = compute_stats(TEST_CSV)

    # --- Summary CSV ---
    rows = []
    for label, st, n_rows in (
        ("train", tr_stats, n_rows_train),
        ("test",  te_stats, n_rows_test),
    ):
        sp  = st["sp"]
        ac  = st["ac"]
        dur = st["durations"]
        rows.append({
            "split"           : label,
            "n_rows"          : st["n_rows"],
            "n_segments"      : st["n_segments"],
            "pct_rows"        : round(100 * st["n_rows"] / n_rows_total, 2),
            "pct_segments"    : round(100 * st["n_segments"] / n_segs_total, 2),
            "mean_speed_mps"  : round(float(sp.mean()), 4),
            "p95_speed_mps"   : round(float(np.percentile(sp, 95)), 4),
            "max_speed_mps"   : round(float(sp.max()), 4),
            "mean_accel_mps2" : round(float(ac.mean()), 4),
            "p95_accel_mps2"  : round(float(np.percentile(ac, 95)), 4),
            "max_accel_mps2"  : round(float(ac.max()), 4),
            "mean_duration_s" : round(float(dur.mean()), 1),
            "p95_duration_s"  : round(float(np.percentile(dur, 95)), 1),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"\n  Saved: {TRAIN_CSV}   ({n_rows_train:,} rows)")
    print(f"  Saved: {TEST_CSV}    ({n_rows_test:,} rows)")
    print(f"  Saved: {SUMMARY_CSV}")

    # --- Verifications ---
    print(f"\n{SEP}")
    print("  VERIFICATION")
    print(SEP)

    overlap = train_ids & test_ids
    print(f"\n  Overlap segment_ids     : {len(overlap)}"
          f"  ({'OK' if not overlap else 'FAIL'})")

    n_segs_out = tr_stats["n_segments"] + te_stats["n_segments"]
    print(f"  Segment count matches   : {n_segs_out:,} / {n_segs_total:,}"
          f"  ({'OK' if n_segs_out == n_segs_total else 'FAIL'})")

    print(f"  Row count               : {n_rows_train:,} + {n_rows_test:,} = {n_rows_total:,}")

    # NaN check — tiny header sample
    for path, label in ((TRAIN_CSV, "train"), (TEST_CSV, "test")):
        s = pd.read_csv(path, nrows=1_000)
        nans = s[["E_m","N_m","U_m","vE_mps","vN_mps","vU_mps"]].isnull().sum().sum()
        print(f"  NaN ENU/vel ({label}, 1k sample) : {nans}"
              f"  ({'OK' if nans == 0 else 'FAIL'})")

    for label, st in (("train", tr_stats), ("test", te_stats)):
        sp_max = float(st["sp"].max())
        ac_max = float(st["ac"].max())
        print(f"  {label:<5}  max_speed={sp_max:.2f}"
              f" ({'OK' if sp_max <= MAX_SPEED_MPS else 'FAIL'})"
              f"   max_accel={ac_max:.2f}"
              f" ({'OK' if ac_max <= MAX_ACCEL_MPS2 else 'FAIL'})")

    # --- Report ---
    print(f"\n{SEP}")
    print("  SPLIT REPORT")
    print(SEP)

    print(f"\n  {'':20} {'train':>12} {'test':>12}")
    print(f"  {'Segments':20} {tr_stats['n_segments']:>12,} {te_stats['n_segments']:>12,}")
    print(f"  {'Rows':20} {n_rows_train:>12,} {n_rows_test:>12,}")
    print(f"  {'% of segments':20}"
          f" {100*tr_stats['n_segments']/n_segs_total:>11.1f}%"
          f" {100*te_stats['n_segments']/n_segs_total:>11.1f}%")
    print(f"  {'% of rows':20}"
          f" {100*n_rows_train/n_rows_total:>11.1f}%"
          f" {100*n_rows_test/n_rows_total:>11.1f}%")

    print(f"\n  Speed (m/s) [1-in-{SUBSAMPLE_EVERY} sample]:")
    for label, st in (("train", tr_stats), ("test", te_stats)):
        sp = st["sp"]
        print(f"    {label:<5}  mean={sp.mean():.2f}  "
              f"p95={np.percentile(sp,95):.2f}  max={sp.max():.2f}")

    print(f"\n  Accel (m/s²) [1-in-{SUBSAMPLE_EVERY} sample]:")
    for label, st in (("train", tr_stats), ("test", te_stats)):
        ac = st["ac"]
        print(f"    {label:<5}  mean={ac.mean():.3f}  "
              f"p95={np.percentile(ac,95):.3f}  max={ac.max():.3f}")

    print(f"\n  Duration per segment (s):")
    for label, st in (("train", tr_stats), ("test", te_stats)):
        dur = st["durations"]
        print(f"    {label:<5}  mean={dur.mean():.0f}  "
              f"p95={np.percentile(dur,95):.0f}  max={dur.max():.0f}")

    PREV = ["segment_id","time","E_m","N_m","U_m",
            "vE_mps","vN_mps","vU_mps","speed_mps","accel_mps2"]
    print(f"\n  First 3 rows of train:")
    print(pd.read_csv(TRAIN_CSV, nrows=3)[PREV].to_string(index=False))
    print(f"\n  First 3 rows of test:")
    print(pd.read_csv(TEST_CSV, nrows=3)[PREV].to_string(index=False))

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
