"""
preprocess_05_convert_segments_to_enu.py

Converts lat / lon / geoaltitude to ENU (East, North, Up) coordinates
for every trajectory segment, using a proper WGS-84 geodetic → ECEF → ENU
pipeline.  The first point of each segment is the local ENU origin (0, 0, 0).
Velocities are computed with np.gradient inside each segment only.

NOTE: The 'altitude' column in trajectory_segments.csv equals geoaltitude
      (set during preprocess_04: df['altitude'] = df['geoaltitude']).
      The script reads that column as the altitude input.

Input : data/preprocess_04_output/trajectory_segments.csv
Output: data/preprocess_05_output/trajectory_segments_enu_with_velocity.csv
        data/preprocess_05_output/enu_conversion_summary.csv

Run:
    python3 preprocess_05_convert_segments_to_enu.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. WGS-84 constants
# ---------------------------------------------------------------------------
WGS84_A   = 6_378_137.0                        # semi-major axis (m)
WGS84_F   = 1.0 / 298.257_223_563              # flattening
WGS84_E2  = 2 * WGS84_F - WGS84_F ** 2        # first eccentricity squared


# ---------------------------------------------------------------------------
# 2. Geodetic → ECEF
# ---------------------------------------------------------------------------
def geodetic_to_ecef(lat_deg: np.ndarray,
                     lon_deg: np.ndarray,
                     alt_m:   np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geodetic (lat, lon, alt) to ECEF (X, Y, Z).

    Parameters
    ----------
    lat_deg : geodetic latitude  (degrees)
    lon_deg : geodetic longitude (degrees)
    alt_m   : ellipsoidal height (metres, i.e. geoaltitude)

    Returns
    -------
    X, Y, Z : ECEF coordinates (metres)
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    h   = alt_m

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    # Prime-vertical radius of curvature
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)

    X = (N + h) * cos_lat * np.cos(lon)
    Y = (N + h) * cos_lat * np.sin(lon)
    Z = (N * (1.0 - WGS84_E2) + h) * sin_lat

    return X, Y, Z


# ---------------------------------------------------------------------------
# 3. Geodetic origin + ECEF point → ENU
# ---------------------------------------------------------------------------
def geodetic_to_enu(lat_deg:  np.ndarray,
                    lon_deg:  np.ndarray,
                    alt_m:    np.ndarray,
                    lat0_deg: float,
                    lon0_deg: float,
                    alt0_m:   float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert arrays of geodetic positions to local ENU relative to a
    reference point (lat0, lon0, alt0).

    The rotation matrix is:
        | E |   | -sin_lon0               cos_lon0              0        | | dX |
        | N | = | -sin_lat0*cos_lon0  -sin_lat0*sin_lon0  cos_lat0 | | dY |
        | U |   |  cos_lat0*cos_lon0   cos_lat0*sin_lon0  sin_lat0 | | dZ |

    Parameters
    ----------
    lat_deg, lon_deg, alt_m  : arrays of target positions
    lat0_deg, lon0_deg, alt0_m : scalar reference / origin

    Returns
    -------
    E_m, N_m, U_m : East, North, Up displacements in metres
    """
    # Reference point in ECEF
    X0, Y0, Z0 = geodetic_to_ecef(
        np.array([lat0_deg]), np.array([lon0_deg]), np.array([alt0_m])
    )
    X0, Y0, Z0 = float(X0[0]), float(Y0[0]), float(Z0[0])

    # All points in ECEF
    X, Y, Z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)

    # ECEF offset from reference
    dX = X - X0
    dY = Y - Y0
    dZ = Z - Z0

    # Rotation matrix elements
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    sin_lat0 = np.sin(lat0);  cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0);  cos_lon0 = np.cos(lon0)

    E_m =  -sin_lon0              * dX + cos_lon0              * dY
    N_m =  -sin_lat0 * cos_lon0   * dX - sin_lat0 * sin_lon0   * dY + cos_lat0 * dZ
    U_m =   cos_lat0 * cos_lon0   * dX + cos_lat0 * sin_lon0   * dY + sin_lat0 * dZ

    return E_m, N_m, U_m


# ---------------------------------------------------------------------------
# 4. Process one segment
# ---------------------------------------------------------------------------
def process_one_segment(seg_df: pd.DataFrame,
                        alt_col: str) -> tuple[pd.DataFrame | None, dict | None, str | None]:
    """
    Convert one segment DataFrame to ENU + compute velocities.

    Parameters
    ----------
    seg_df  : rows for a single segment_id (any order)
    alt_col : name of the altitude column ('altitude' or 'geoaltitude')

    Returns
    -------
    (result_df, summary_dict, skip_reason)
    skip_reason is None when the segment is valid.
    """
    seg_id = seg_df["segment_id"].iloc[0]

    # Drop rows with missing required values
    required = ["time", "lat", "lon", alt_col]
    seg = seg_df.dropna(subset=required).copy()

    # Sort by time, remove duplicate timestamps
    seg = seg.sort_values("time").drop_duplicates(subset="time", keep="first")
    seg = seg.reset_index(drop=True)

    # Skip segments with fewer than 3 points
    if len(seg) < 3:
        return None, None, f"too_few_points ({len(seg)})"

    # Local ENU origin = first point of this segment
    lat0 = float(seg["lat"].iloc[0])
    lon0 = float(seg["lon"].iloc[0])
    alt0 = float(seg[alt_col].iloc[0])

    # Convert all points
    E_m, N_m, U_m = geodetic_to_enu(
        seg["lat"].values,
        seg["lon"].values,
        seg[alt_col].values,
        lat0, lon0, alt0,
    )

    # Sanity: first point should be at (≈0, ≈0, ≈0)
    origin_err = max(abs(E_m[0]), abs(N_m[0]), abs(U_m[0]))
    if origin_err > 1e-3:   # tolerance: 1 mm
        return None, None, f"origin_error ({origin_err:.2e} m)"

    seg["E_m"] = E_m
    seg["N_m"] = N_m
    seg["U_m"] = U_m

    # Compute velocities using np.gradient (central differences, m/s)
    t = seg["time"].values.astype(np.float64)
    seg["vE_mps"] = np.gradient(E_m, t)
    seg["vN_mps"] = np.gradient(N_m, t)
    seg["vU_mps"] = np.gradient(U_m, t)

    # Sanity: no NaNs in output columns
    out_cols = ["E_m", "N_m", "U_m", "vE_mps", "vN_mps", "vU_mps"]
    if seg[out_cols].isnull().any().any():
        return None, None, "nan_in_output"

    # Per-segment summary
    speed = np.sqrt(seg["vE_mps"] ** 2 + seg["vN_mps"] ** 2 + seg["vU_mps"] ** 2)
    summary = {
        "segment_id"   : seg_id,
        "n_points"     : len(seg),
        "start_time"   : int(seg["time"].iloc[0]),
        "end_time"     : int(seg["time"].iloc[-1]),
        "duration_s"   : int(seg["time"].iloc[-1]) - int(seg["time"].iloc[0]),
        "max_abs_E_m"  : float(np.abs(E_m).max()),
        "max_abs_N_m"  : float(np.abs(N_m).max()),
        "max_abs_U_m"  : float(np.abs(U_m).max()),
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps" : float(speed.max()),
    }

    return seg, summary, None


# ---------------------------------------------------------------------------
# 5. Convert all segments
# ---------------------------------------------------------------------------
def convert_all_segments(df: pd.DataFrame,
                         alt_col: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Iterate over all unique segment_ids and process each one.

    Returns
    -------
    result_df   : concatenated ENU + velocity DataFrame
    summary_df  : per-segment summary DataFrame
    skip_report : dict of {reason: count}
    """
    segments        = df.groupby("segment_id", sort=True)
    n_total         = df["segment_id"].nunique()
    result_parts    = []
    summary_rows    = []
    skip_report     = {}
    n_valid         = 0

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


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    INPUT_CSV   = "data/preprocess_04_output/trajectory_segments.csv"
    OUTPUT_CSV  = "data/preprocess_05_output/trajectory_segments_enu_with_velocity.csv"
    SUMMARY_CSV = "data/preprocess_05_output/enu_conversion_summary.csv"

    # The 'altitude' column in trajectory_segments.csv holds geoaltitude
    # (set during preprocess_04: df['altitude'] = df['geoaltitude'])
    ALT_COL = "geoaltitude" if "geoaltitude" in pd.read_csv(INPUT_CSV, nrows=1).columns \
              else "altitude"

    print(f"\n{'='*60}")
    print(f"  preprocess_05_convert_segments_to_enu.py")
    print(f"{'='*60}")
    print(f"  Input  : {INPUT_CSV}")
    print(f"  Output : {OUTPUT_CSV}")
    print(f"  Summary: {SUMMARY_CSV}")

    os.makedirs("data/preprocess_05_output", exist_ok=True)

    # --- Load ---
    print(f"\nLoading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Rows loaded : {len(df):,}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Altitude col: '{ALT_COL}'")

    # Drop rows missing any critical field before groupby
    critical = ["segment_id", "time", "lat", "lon", ALT_COL]
    before   = len(df)
    df       = df.dropna(subset=critical)
    dropped  = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing critical values")

    n_segs = df["segment_id"].nunique()
    print(f"  Segments    : {n_segs:,}\n")

    # --- Convert ---
    result_df, summary_df, skip_report = convert_all_segments(df, ALT_COL)

    # --- Save ---
    result_df.to_csv(OUTPUT_CSV,  index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n  Saved: {OUTPUT_CSV}  ({len(result_df):,} rows)")
    print(f"  Saved: {SUMMARY_CSV}  ({len(summary_df):,} rows)")

    # --- Sanity checks ---
    print(f"\n{'='*60}")
    print("  SANITY CHECKS")
    print(f"{'='*60}")

    # 1. Output file exists
    print(f"\n  1. Output file exists       : {os.path.exists(OUTPUT_CSV)}")

    # 2. Every segment starts near (0, 0, 0)
    first_pts = result_df.groupby("segment_id").first()[["E_m", "N_m", "U_m"]]
    max_origin = first_pts.abs().max().max()
    print(f"  2. Max |origin| across segs : {max_origin:.2e} m  "
          f"({'OK' if max_origin < 1e-3 else 'WARNING'})")

    # 3. No NaNs in ENU / velocity columns
    enu_cols   = ["E_m", "N_m", "U_m", "vE_mps", "vN_mps", "vU_mps"]
    nan_counts = result_df[enu_cols].isnull().sum()
    total_nans = nan_counts.sum()
    print(f"  3. NaN values in ENU/vel    : {total_nans}  "
          f"({'OK' if total_nans == 0 else 'WARNING — ' + str(nan_counts.to_dict())})")

    # 4. First few rows of output
    print(f"\n  4. First 3 rows of output:")
    preview_cols = ["segment_id", "time", "lat", "lon", ALT_COL, "E_m", "N_m", "U_m",
                    "vE_mps", "vN_mps", "vU_mps"]
    print(result_df[preview_cols].head(3).to_string(index=False))

    # 5. First few rows of summary
    print(f"\n  5. First 3 rows of summary:")
    print(summary_df.head(3).to_string(index=False))

    # 6. Skipped segments
    total_skipped = sum(skip_report.values())
    print(f"\n  6. Skipped segments         : {total_skipped:,}")
    for reason, cnt in sorted(skip_report.items(), key=lambda x: -x[1]):
        print(f"       {reason:<35}: {cnt:,}")

    # Global speed stats
    speed_all = np.sqrt(
        result_df["vE_mps"] ** 2 +
        result_df["vN_mps"] ** 2 +
        result_df["vU_mps"] ** 2
    )
    print(f"\n  Speed (3D) across all points:")
    print(f"    mean : {speed_all.mean():.2f} m/s")
    print(f"    p95  : {np.percentile(speed_all, 95):.2f} m/s")
    print(f"    max  : {speed_all.max():.2f} m/s")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
