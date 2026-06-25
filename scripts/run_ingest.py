"""
Ingest pipeline — stages 01–03.

  Stage 01: Merge 24 hourly ADS-B CSVs into one daily CSV per date.
  Stage 02: Filter OpenSky aircraft database to GA fixed-wing aircraft.
  Stage 03: Filter each daily states CSV to GA-only rows; sort by (icao24, time).

Outputs land in paths.stage_01_out, paths.stage_02_out, paths.stage_03_out
as configured in configs/pipeline.yaml.
"""

import os
import glob
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH  = os.path.join(_PROJECT_ROOT, "configs", "pipeline.yaml")

CHUNK_SIZE_01 = 100_000
CHUNK_SIZE_03 = 500_000

from adsb_preprocess.io_utils import load_config
from adsb_preprocess.filters  import filter_ga_aircraft


def run_concat_daily():
    """Merge hourly state CSVs into one daily CSV for each configured date."""
    cfg   = load_config(_CONFIG_PATH)
    dates = cfg["dates"]
    out_dir = cfg["paths"]["stage_01_out"]
    os.makedirs(out_dir, exist_ok=True)

    for date in dates:
        pattern = os.path.join(cfg["paths"]["archive_dir"], f"states_{date}-*.csv")
        files   = sorted(glob.glob(pattern))
        print(f"[01] {date}: found {len(files)} hourly files")

        output     = os.path.join(out_dir, f"states_{date}.csv")
        total_rows = 0

        with open(output, "w") as out_f:
            for i, filepath in enumerate(files):
                print(f"  [{i+1}/{len(files)}] {filepath}")
                for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE_01):
                    chunk.to_csv(out_f, index=False, header=(out_f.tell() == 0))
                    total_rows += len(chunk)

        print(f"[01] Saved {total_rows:,} rows → {output}\n")


def run_filter_aircraft():
    """Filter the raw OpenSky aircraft database to GA fixed-wing aircraft."""
    cfg       = load_config(_CONFIG_PATH)
    input_path  = cfg["aircraft_db_raw"]
    output_path = cfg["aircraft_db"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path, low_memory=False)
    print(f"[02] Loaded {len(df):,} rows from {input_path}")

    df = filter_ga_aircraft(df)
    print(f"[02] After GA filter: {len(df):,} rows")

    df.to_csv(output_path, index=False)
    print(f"[02] Saved → {output_path}\n")


def run_filter_sort_daily():
    """Filter each daily states CSV to GA aircraft only and sort by (icao24, time)."""
    cfg     = load_config(_CONFIG_PATH)
    ga_db   = cfg["aircraft_db"]
    dates   = cfg["dates"]
    in_dir  = cfg["paths"]["stage_01_out"]
    out_dir = cfg["paths"]["stage_03_out"]
    os.makedirs(out_dir, exist_ok=True)

    ga_icao24 = set(pd.read_csv(ga_db, usecols=['icao24'], low_memory=False)['icao24'])
    print(f"[03] Loaded {len(ga_icao24):,} GA icao24 codes\n")

    for date in dates:
        input_path  = os.path.join(in_dir,  f"states_{date}.csv")
        output_path = os.path.join(out_dir, f"states_{date}-FixedWingGA.csv")
        print(f"[03] Processing {input_path} ...")

        chunks = []
        for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE_03):
            ga_rows = chunk[chunk['icao24'].isin(ga_icao24)]
            if not ga_rows.empty:
                chunks.append(ga_rows)

        df = pd.concat(chunks, ignore_index=True)
        df.sort_values(['icao24', 'time'], inplace=True)
        df.to_csv(output_path, index=False)
        print(f"[03] Saved {len(df):,} rows → {output_path}\n")


def main():
    print("=== Stage 01: Concat daily ===", flush=True)
    run_concat_daily()
    print("=== Stage 02: Filter aircraft DB ===", flush=True)
    run_filter_aircraft()
    print("=== Stage 03: Filter and sort daily ===", flush=True)
    run_filter_sort_daily()
    print("Ingest complete.", flush=True)


if __name__ == "__main__":
    main()
