"""
preprocess_06_clean_enu_velocity_outliers.py

Drops trajectory segments with unrealistic speed or acceleration.

Drop criteria (entire segment dropped if either fires):
  max_speed_mps  > 150    (~291 kt)
  max_accel_mps2 > 10     (~1 g)

Memory strategy — single streaming pass:
  Read INPUT_CSV in chunks of CHUNK_SIZE rows.
  Because the file is sorted by (segment_id, time), segments are contiguous.
  Carry a one-segment row buffer across chunk boundaries.
  Process each segment the moment it is complete, then immediately discard it.
  Write kept segments to the output file incrementally.
  Peak RAM ≈ one chunk + one segment (well under 100 MB).

Input : data/preprocess_05_output/trajectory_segments_enu_with_velocity.csv
Output: data/preprocess_06_output/trajectory_segments_enu_clean.csv
        data/preprocess_06_output/enu_cleaning_summary.csv
        data/preprocess_06_output/dropped_segments_velocity_outliers.csv
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_CSV    = "data/preprocess_05_output/trajectory_segments_enu_with_velocity.csv"
OUTPUT_CSV   = "data/preprocess_06_output/trajectory_segments_enu_clean.csv"
SUMMARY_CSV  = "data/preprocess_06_output/enu_cleaning_summary.csv"
DROPPED_CSV  = "data/preprocess_06_output/dropped_segments_velocity_outliers.csv"

MAX_SPEED_MPS  = 150.0
MAX_ACCEL_MPS2 = 10.0
CHUNK_SIZE     = 100_000


# ---------------------------------------------------------------------------
# Per-segment processor
# ---------------------------------------------------------------------------
def process_segment(rows: list) -> tuple[pd.DataFrame | None, dict]:
    """
    rows  : list of DataFrame slices belonging to one segment (already sorted)
    Returns (output_df_or_None, summary_dict).
    output_df is None for dropped segments.
    """
    seg = pd.concat(rows, ignore_index=True)
    seg = seg.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    seg_id = int(seg["segment_id"].iloc[0])

    base = {
        "segment_id"    : seg_id,
        "n_points"      : len(seg),
        "start_time"    : int(seg["time"].iloc[0]),
        "end_time"      : int(seg["time"].iloc[-1]),
        "duration_s"    : int(seg["time"].iloc[-1]) - int(seg["time"].iloc[0]),
    }

    if len(seg) < 3:
        base.update({
            "mean_speed_mps" : np.nan, "p95_speed_mps" : np.nan, "max_speed_mps" : np.nan,
            "mean_accel_mps2": np.nan, "p95_accel_mps2": np.nan, "max_accel_mps2": np.nan,
            "drop_reason"    : "too_few_points",
        })
        return None, base

    t  = seg["time"].values.astype(np.float64)
    vE = seg["vE_mps"].values
    vN = seg["vN_mps"].values
    vU = seg["vU_mps"].values

    speed = np.sqrt(vE ** 2 + vN ** 2 + vU ** 2)
    aE    = np.gradient(vE, t)
    aN    = np.gradient(vN, t)
    aU    = np.gradient(vU, t)
    accel = np.sqrt(aE ** 2 + aN ** 2 + aU ** 2)

    max_speed = float(speed.max())
    max_accel = float(accel.max())

    speed_bad = max_speed > MAX_SPEED_MPS
    accel_bad = max_accel > MAX_ACCEL_MPS2

    if speed_bad and accel_bad:
        reason = "speed_and_accel_outlier"
    elif speed_bad:
        reason = "speed_outlier"
    elif accel_bad:
        reason = "accel_outlier"
    else:
        reason = "keep"

    summary = {
        **base,
        "mean_speed_mps" : float(speed.mean()),
        "p95_speed_mps"  : float(np.percentile(speed, 95)),
        "max_speed_mps"  : max_speed,
        "mean_accel_mps2": float(accel.mean()),
        "p95_accel_mps2" : float(np.percentile(accel, 95)),
        "max_accel_mps2" : max_accel,
        "drop_reason"    : reason,
    }

    if reason != "keep":
        return None, summary

    seg["speed_mps"]  = speed
    seg["aE_mps2"]    = aE
    seg["aN_mps2"]    = aN
    seg["aU_mps2"]    = aU
    seg["accel_mps2"] = accel

    return seg, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  preprocess_06_clean_enu_velocity_outliers.py")
    print(f"  Input : {INPUT_CSV}")
    print(SEP)

    os.makedirs("data/preprocess_06_output", exist_ok=True)

    # Accumulators
    summary_rows  : list[dict]      = []
    buffer_rows   : list[pd.DataFrame] = []
    current_seg_id: int | None       = None

    n_rows_in    = 0
    n_rows_out   = 0
    n_segs_total = 0
    n_segs_kept  = 0
    header_done  = False
    n_chunks     = 0

    with open(OUTPUT_CSV, "w") as out_f:

        def flush(rows, seg_id):
            """Process buffered rows for one segment; write if kept."""
            nonlocal n_segs_total, n_segs_kept, n_rows_out, header_done
            if not rows:
                return
            result, summary = process_segment(rows)
            summary_rows.append(summary)
            n_segs_total += 1
            if result is not None:
                result.to_csv(out_f, index=False, header=not header_done)
                header_done = True
                n_rows_out += len(result)
                n_segs_kept += 1

        for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False):
            n_rows_in += len(chunk)
            n_chunks  += 1
            if n_chunks % 20 == 0:
                print(f"  ... chunk {n_chunks:>4}  rows read: {n_rows_in:>10,}  "
                      f"segs done: {n_segs_total:>7,}")

            # Iterate through each group in this chunk in sorted order
            for seg_id, grp in chunk.groupby("segment_id", sort=True):
                if seg_id != current_seg_id:
                    flush(buffer_rows, current_seg_id)
                    buffer_rows   = [grp]
                    current_seg_id = seg_id
                else:
                    buffer_rows.append(grp)

        flush(buffer_rows, current_seg_id)   # final segment

    # -----------------------------------------------------------------------
    # Write summary files
    # -----------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    dropped_df = summary_df[summary_df["drop_reason"] != "keep"]
    dropped_df.to_csv(DROPPED_CSV, index=False)

    n_dropped    = n_segs_total - n_segs_kept
    n_rows_drop  = n_rows_in - n_rows_out

    print(f"\n  Saved: {OUTPUT_CSV}   ({n_rows_out:,} rows)")
    print(f"  Saved: {SUMMARY_CSV}   ({len(summary_df):,} rows)")
    print(f"  Saved: {DROPPED_CSV}   ({n_dropped:,} rows)")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  CLEANING REPORT")
    print(SEP)

    print(f"\n  Rows")
    print(f"    Before : {n_rows_in:>10,}")
    print(f"    After  : {n_rows_out:>10,}")
    print(f"    Dropped: {n_rows_drop:>10,}  ({100*n_rows_drop/n_rows_in:.2f}%)")

    print(f"\n  Segments")
    print(f"    Before : {n_segs_total:>10,}")
    print(f"    After  : {n_segs_kept:>10,}")
    print(f"    Dropped: {n_dropped:>10,}  ({100*n_dropped/n_segs_total:.2f}%)")

    print(f"\n  Drop reasons")
    for reason, cnt in summary_df["drop_reason"].value_counts().items():
        print(f"    {reason:<28}: {cnt:>6,}  ({100*cnt/n_segs_total:.2f}%)")

    kept  = summary_df[summary_df["drop_reason"] == "keep"]

    def dist(label, col, df):
        s = df[col].dropna()
        print(f"    {label:<14}  mean={s.mean():.3f}  p50={s.median():.3f}"
              f"  p95={s.quantile(.95):.3f}  p99={s.quantile(.99):.3f}  max={s.max():.3f}")

    print(f"\n  Per-segment MEAN speed (m/s)  — before/after")
    dist("all segments",  "mean_speed_mps", summary_df)
    dist("kept segments", "mean_speed_mps", kept)

    print(f"\n  Per-segment MAX speed (m/s)   — before/after")
    dist("all segments",  "max_speed_mps", summary_df)
    dist("kept segments", "max_speed_mps", kept)

    print(f"\n  Per-segment MEAN accel (m/s²) — before/after")
    dist("all segments",  "mean_accel_mps2", summary_df)
    dist("kept segments", "mean_accel_mps2", kept)

    print(f"\n  Per-segment MAX accel (m/s²)  — before/after")
    dist("all segments",  "max_accel_mps2", summary_df)
    dist("kept segments", "max_accel_mps2", kept)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
