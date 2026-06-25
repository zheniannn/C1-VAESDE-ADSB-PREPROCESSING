"""Orchestration for ingest stages 01–03."""

import os
import glob
import pandas as pd

from adsb_preprocess.filters import filter_ga_aircraft

CHUNK_SIZE_01 = 100_000
CHUNK_SIZE_03 = 500_000


def run_stage_01(cfg: dict) -> None:
    """Merge hourly state CSVs into one daily CSV for each configured date."""
    out_dir = cfg["paths"]["stage_01_out"]
    os.makedirs(out_dir, exist_ok=True)

    for date in cfg["dates"]:
        pattern    = os.path.join(cfg["paths"]["archive_dir"], f"states_{date}-*.csv")
        files      = sorted(glob.glob(pattern))
        output     = os.path.join(out_dir, f"states_{date}.csv")
        total_rows = 0
        print(f"[01] {date}: {len(files)} hourly files")

        with open(output, "w") as out_f:
            for i, filepath in enumerate(files):
                print(f"  [{i+1}/{len(files)}] {filepath}")
                for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE_01):
                    chunk.to_csv(out_f, index=False, header=(out_f.tell() == 0))
                    total_rows += len(chunk)

        print(f"[01] Saved {total_rows:,} rows → {output}\n")


def run_stage_02(cfg: dict) -> None:
    """Filter the raw OpenSky aircraft database to GA fixed-wing aircraft."""
    input_path  = cfg["aircraft_db_raw"]
    output_path = cfg["aircraft_db"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    print(f"[02] Loaded {len(df):,} rows from {input_path}")
    df = filter_ga_aircraft(df)
    df.to_csv(output_path, index=False)
    print(f"[02] After GA filter: {len(df):,} rows → {output_path}\n")


def run_stage_03(cfg: dict) -> None:
    """Filter each daily states CSV to GA-only rows and sort by (icao24, time)."""
    ga_icao24 = set(pd.read_csv(cfg["aircraft_db"], usecols=["icao24"], low_memory=False)["icao24"])
    print(f"[03] Loaded {len(ga_icao24):,} GA icao24 codes\n")

    in_dir  = cfg["paths"]["stage_01_out"]
    out_dir = cfg["paths"]["stage_03_out"]
    os.makedirs(out_dir, exist_ok=True)

    for date in cfg["dates"]:
        input_path  = os.path.join(in_dir,  f"states_{date}.csv")
        output_path = os.path.join(out_dir, f"states_{date}-FixedWingGA.csv")
        print(f"[03] Processing {input_path} ...")

        chunks = [
            ga_rows
            for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE_03)
            if not (ga_rows := chunk[chunk["icao24"].isin(ga_icao24)]).empty
        ]
        df = pd.concat(chunks, ignore_index=True)
        df.sort_values(["icao24", "time"], inplace=True)
        df.to_csv(output_path, index=False)
        print(f"[03] Saved {len(df):,} rows → {output_path}\n")
