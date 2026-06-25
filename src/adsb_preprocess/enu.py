"""ENU coordinate conversion for trajectory segments."""

import numpy as np
import pandas as pd

from adsb_preprocess.geo import geodetic_to_enu


def process_one_segment(
    seg_df: pd.DataFrame,
    alt_col: str,
) -> tuple[pd.DataFrame | None, dict | None, str | None]:
    """
    Convert one segment to ENU coordinates and compute per-axis velocities.

    Returns (result_df, summary_dict, skip_reason).
    skip_reason is None when the segment is valid.
    """
    seg_id = seg_df["segment_id"].iloc[0]

    seg = seg_df.dropna(subset=["time", "lat", "lon", alt_col]).copy()
    seg = seg.sort_values("time").drop_duplicates(subset="time", keep="first").reset_index(drop=True)

    if len(seg) < 3:
        return None, None, f"too_few_points ({len(seg)})"

    lat0 = float(seg["lat"].iloc[0])
    lon0 = float(seg["lon"].iloc[0])
    alt0 = float(seg[alt_col].iloc[0])

    E_m, N_m, U_m = geodetic_to_enu(
        seg["lat"].values, seg["lon"].values, seg[alt_col].values,
        lat0, lon0, alt0,
    )

    origin_err = max(abs(E_m[0]), abs(N_m[0]), abs(U_m[0]))
    if origin_err > 1e-3:
        return None, None, f"origin_error ({origin_err:.2e} m)"

    seg["E_m"] = E_m
    seg["N_m"] = N_m
    seg["U_m"] = U_m

    t = seg["time"].values.astype(np.float64)
    seg["vE_mps"] = np.gradient(E_m, t)
    seg["vN_mps"] = np.gradient(N_m, t)
    seg["vU_mps"] = np.gradient(U_m, t)

    out_cols = ["E_m", "N_m", "U_m", "vE_mps", "vN_mps", "vU_mps"]
    if seg[out_cols].isnull().any().any():
        return None, None, "nan_in_output"

    speed = np.sqrt(seg["vE_mps"] ** 2 + seg["vN_mps"] ** 2 + seg["vU_mps"] ** 2)
    summary = {
        "segment_id":    seg_id,
        "n_points":      len(seg),
        "start_time":    int(seg["time"].iloc[0]),
        "end_time":      int(seg["time"].iloc[-1]),
        "duration_s":    int(seg["time"].iloc[-1]) - int(seg["time"].iloc[0]),
        "max_abs_E_m":   float(np.abs(E_m).max()),
        "max_abs_N_m":   float(np.abs(N_m).max()),
        "max_abs_U_m":   float(np.abs(U_m).max()),
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps":  float(speed.max()),
    }
    return seg, summary, None


def convert_all_segments(
    df: pd.DataFrame,
    alt_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Iterate over all unique segment_ids and convert each to ENU.

    Returns (result_df, summary_df, skip_report).
    """
    segments     = df.groupby("segment_id", sort=True)
    n_total      = df["segment_id"].nunique()
    result_parts = []
    summary_rows = []
    skip_report  = {}
    n_valid      = 0

    print(f"  Processing {n_total:,} segments ...")

    for idx, (seg_id, seg_df) in enumerate(segments):
        if idx % 10_000 == 0 and idx > 0:
            print(f"    {idx:>7,} / {n_total:,}   valid so far: {n_valid:,}")

        result, summary, reason = process_one_segment(seg_df, alt_col)
        if reason is not None:
            skip_report[reason] = skip_report.get(reason, 0) + 1
            continue

        result_parts.append(result)
        summary_rows.append(summary)
        n_valid += 1

    print(f"  Done — valid: {n_valid:,}  skipped: {n_total - n_valid:,}")

    result_df  = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return result_df, summary_df, skip_report
