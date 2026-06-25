"""Velocity and acceleration outlier removal for ENU trajectory segments."""

import numpy as np
import pandas as pd


def process_segment(rows: list, max_speed_mps: float, max_accel_mps2: float) -> tuple[pd.DataFrame | None, dict]:
    """
    Evaluate one segment for speed/acceleration outliers.

    Returns (output_df_or_None, summary_dict). output_df is None for dropped segments.
    """
    seg    = pd.concat(rows, ignore_index=True)
    seg    = seg.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    seg_id = int(seg["segment_id"].iloc[0])

    base = {
        "segment_id": seg_id,
        "n_points":   len(seg),
        "start_time": int(seg["time"].iloc[0]),
        "end_time":   int(seg["time"].iloc[-1]),
        "duration_s": int(seg["time"].iloc[-1]) - int(seg["time"].iloc[0]),
    }

    if len(seg) < 3:
        base.update({
            "mean_speed_mps":  np.nan, "p95_speed_mps":  np.nan, "max_speed_mps":  np.nan,
            "mean_accel_mps2": np.nan, "p95_accel_mps2": np.nan, "max_accel_mps2": np.nan,
            "drop_reason":     "too_few_points",
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

    speed_bad = max_speed > max_speed_mps
    accel_bad = max_accel > max_accel_mps2

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
        "mean_speed_mps":  float(speed.mean()),
        "p95_speed_mps":   float(np.percentile(speed, 95)),
        "max_speed_mps":   max_speed,
        "mean_accel_mps2": float(accel.mean()),
        "p95_accel_mps2":  float(np.percentile(accel, 95)),
        "max_accel_mps2":  max_accel,
        "drop_reason":     reason,
    }

    if reason != "keep":
        return None, summary

    seg["speed_mps"]  = speed
    seg["aE_mps2"]    = aE
    seg["aN_mps2"]    = aN
    seg["aU_mps2"]    = aU
    seg["accel_mps2"] = accel
    return seg, summary


def stream_clean_segments(
    input_csv:      str,
    output_csv:     str,
    max_speed_mps:  float,
    max_accel_mps2: float,
    chunk_size:     int = 100_000,
) -> tuple[pd.DataFrame, int, int, int, int]:
    """
    Stream input_csv, drop outlier segments, write clean rows to output_csv.

    Returns (summary_df, n_rows_in, n_rows_out, n_segs_total, n_segs_kept).
    """
    summary_rows:   list[dict]         = []
    buffer_rows:    list[pd.DataFrame] = []
    current_seg_id: int | None         = None

    n_rows_in = n_rows_out = n_segs_total = n_segs_kept = 0
    header_done = False
    n_chunks    = 0

    with open(output_csv, "w") as out_f:
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
            n_rows_in += len(chunk)
            n_chunks  += 1
            if n_chunks % 20 == 0:
                print(f"  ... chunk {n_chunks:>4}  rows: {n_rows_in:>10,}  segs: {n_segs_total:>7,}")

            for seg_id, grp in chunk.groupby("segment_id", sort=True):
                if seg_id != current_seg_id:
                    if buffer_rows:
                        result, summary = process_segment(buffer_rows, max_speed_mps, max_accel_mps2)
                        summary_rows.append(summary)
                        n_segs_total += 1
                        if result is not None:
                            result.to_csv(out_f, index=False, header=not header_done)
                            header_done = True
                            n_rows_out += len(result)
                            n_segs_kept += 1
                    buffer_rows    = [grp]
                    current_seg_id = seg_id
                else:
                    buffer_rows.append(grp)

        if buffer_rows:
            result, summary = process_segment(buffer_rows, max_speed_mps, max_accel_mps2)
            summary_rows.append(summary)
            n_segs_total += 1
            if result is not None:
                result.to_csv(out_f, index=False, header=not header_done)
                n_rows_out += len(result)
                n_segs_kept += 1

    return pd.DataFrame(summary_rows), n_rows_in, n_rows_out, n_segs_total, n_segs_kept
