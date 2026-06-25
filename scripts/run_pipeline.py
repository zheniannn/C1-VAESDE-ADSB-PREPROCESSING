"""
Trajectory pipeline — stages 04–06.

  Stage 04: Build cleaned trajectory segments from daily GA state files.
  Stage 05: Convert lat/lon/alt to ENU coordinates; compute velocities.
  Stage 06: Drop segments with unrealistic speed (>150 m/s) or accel (>10 m/s²).

Outputs land in paths.stage_04_out, paths.stage_05_out, paths.stage_06_out
as configured in configs/pipeline.yaml.
"""

import os
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH  = os.path.join(_PROJECT_ROOT, "configs", "pipeline.yaml")

from adsb_preprocess.io_utils     import load_config
from adsb_preprocess.trajectories import clean_day, split_into_segments, KNOTS_TO_MS, SEG_COLS
from adsb_preprocess.enu          import convert_all_segments
from adsb_preprocess.cleaning     import stream_clean_segments


def run_generate_trajectories():
    """Build trajectory segments from per-day GA state files."""
    cfg     = load_config(_CONFIG_PATH)
    tr_cfg  = cfg["trajectory"]
    out_dir = cfg["paths"]["stage_04_out"]

    SEG_PATH = os.path.join(out_dir, "trajectory_segments.csv")
    SUM_PATH = os.path.join(out_dir, "trajectory_summary.csv")
    RES_PATH = os.path.join(out_dir, "Results.csv")

    max_implied_ms = tr_cfg["max_implied_kt"] * KNOTS_TO_MS
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(columns=SEG_COLS).to_csv(SEG_PATH, index=False)

    all_summary_rows  = []
    global_seg_id     = 0
    observed_aircraft = set()
    seg_pings_discarded = 0

    for date in cfg["dates"]:
        path = os.path.join(cfg["paths"]["stage_03_out"], f"states_{date}-FixedWingGA.csv")
        print(f"\n{'='*55}\n  {date}\n{'='*55}")

        df = pd.read_csv(path, low_memory=False)
        df, drops = clean_day(df, date, tr_cfg["min_alt_m"], tr_cfg["max_alt_m"])
        print(f"  After row filters: {drops['after_filter']:,} rows remaining")

        seg_frames, summary_rows, global_seg_id, time_splits, speed_splits, pings_disc = \
            split_into_segments(
                df, global_seg_id,
                tr_cfg["max_time_gap_s"], max_implied_ms, tr_cfg["min_duration_s"],
            )

        seg_pings_discarded += pings_disc
        all_summary_rows.extend(summary_rows)
        observed_aircraft.update(df['icao24'].unique())

        if seg_frames:
            pd.concat(seg_frames, ignore_index=True).to_csv(SEG_PATH, mode='a', header=False, index=False)

        print(f"  Time-gap splits: {time_splits:,}  Speed splits: {speed_splits:,}")
        print(f"  Short-segment pings dropped: {pings_disc:,}")
        print(f"  Running total segments: {global_seg_id:,}")

    df_sum  = pd.DataFrame(all_summary_rows)
    df_sum.to_csv(SUM_PATH, index=False)

    ga_registry = len(pd.read_csv(cfg["aircraft_db"], usecols=['icao24']))
    dur_min     = df_sum['duration_seconds'] / 60
    pd.DataFrame({
        'Metric': ['Aircraft in GA registry', 'Aircraft observed flying (across 4 days)',
                   'Trajectory segments', 'ADS-B pings in output',
                   'Mean segment duration', 'Median segment duration', 'Max segment duration'],
        'Value':  [ga_registry, len(observed_aircraft), global_seg_id,
                   int(df_sum['n_points'].sum()),
                   f"{dur_min.mean():.1f} min", f"{dur_min.median():.1f} min", f"{dur_min.max():.1f} min"],
    }).to_csv(RES_PATH, index=False)

    print(f"\n[04] {global_seg_id:,} segments → {SEG_PATH}")


def run_convert_enu():
    """Convert trajectory segments to ENU coordinates with per-axis velocities."""
    cfg     = load_config(_CONFIG_PATH)
    in_dir  = cfg["paths"]["stage_04_out"]
    out_dir = cfg["paths"]["stage_05_out"]

    INPUT_CSV   = os.path.join(in_dir,  "trajectory_segments.csv")
    OUTPUT_CSV  = os.path.join(out_dir, "trajectory_segments_enu_with_velocity.csv")
    SUMMARY_CSV = os.path.join(out_dir, "enu_conversion_summary.csv")

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  Stage 05: Convert to ENU\n  Input : {INPUT_CSV}\n{'='*60}")

    ALT_COL = "geoaltitude" if "geoaltitude" in pd.read_csv(INPUT_CSV, nrows=1).columns else "altitude"

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Rows: {len(df):,}  Segments: {df['segment_id'].nunique():,}")

    before = len(df)
    df     = df.dropna(subset=["segment_id", "time", "lat", "lon", ALT_COL])
    if before - len(df):
        print(f"  Dropped {before - len(df):,} rows with missing critical values")

    result_df, summary_df, skip_report = convert_all_segments(df, ALT_COL)

    result_df.to_csv(OUTPUT_CSV,   index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    first_pts  = result_df.groupby("segment_id").first()[["E_m", "N_m", "U_m"]]
    max_origin = first_pts.abs().max().max()
    total_nans = result_df[["E_m", "N_m", "U_m", "vE_mps", "vN_mps", "vU_mps"]].isnull().sum().sum()
    print(f"[05] {len(result_df):,} rows → {OUTPUT_CSV}")
    print(f"  Skipped: {sum(skip_report.values()):,}  Max |origin|: {max_origin:.2e} m  NaNs: {total_nans}")


def run_clean_outliers():
    """Drop segments with speed > 150 m/s or accel > 10 m/s²."""
    cfg     = load_config(_CONFIG_PATH)
    cl_cfg  = cfg["cleaning"]
    in_dir  = cfg["paths"]["stage_05_out"]
    out_dir = cfg["paths"]["stage_06_out"]

    INPUT_CSV   = os.path.join(in_dir,  "trajectory_segments_enu_with_velocity.csv")
    OUTPUT_CSV  = os.path.join(out_dir, "trajectory_segments_enu_clean.csv")
    SUMMARY_CSV = os.path.join(out_dir, "enu_cleaning_summary.csv")
    DROPPED_CSV = os.path.join(out_dir, "dropped_segments_velocity_outliers.csv")

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  Stage 06: Clean outliers\n  Input : {INPUT_CSV}\n{'='*60}")

    summary_df, n_rows_in, n_rows_out, n_segs_total, n_segs_kept = stream_clean_segments(
        INPUT_CSV, OUTPUT_CSV, cl_cfg["max_speed_mps"], cl_cfg["max_accel_mps2"]
    )

    summary_df.to_csv(SUMMARY_CSV, index=False)
    summary_df[summary_df["drop_reason"] != "keep"].to_csv(DROPPED_CSV, index=False)

    n_dropped = n_segs_total - n_segs_kept
    print(f"[06] Rows:     {n_rows_in:,} → {n_rows_out:,} (dropped {n_rows_in - n_rows_out:,})")
    print(f"[06] Segments: {n_segs_total:,} → {n_segs_kept:,} (dropped {n_dropped:,})")
    for reason, cnt in summary_df["drop_reason"].value_counts().items():
        print(f"  {reason:<28}: {cnt:>6,}")
    print(f"[06] Saved → {OUTPUT_CSV}")


def main():
    print("=== Stage 04: Generate trajectories ===", flush=True)
    run_generate_trajectories()
    print("\n=== Stage 05: Convert to ENU ===", flush=True)
    run_convert_enu()
    print("\n=== Stage 06: Clean outliers ===", flush=True)
    run_clean_outliers()
    print("\nPipeline complete.", flush=True)


if __name__ == "__main__":
    main()
