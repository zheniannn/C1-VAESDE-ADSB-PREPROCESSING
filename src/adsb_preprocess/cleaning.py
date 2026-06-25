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
