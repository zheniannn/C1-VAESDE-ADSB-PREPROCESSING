# preprocess_03_filter_sort_daily.py
#
# Filters each daily states CSV to keep only GA aircraft (those whose icao24
# appears in the FixedWingGA aircraft database), then sorts by icao24 then time.
#
# Reads the daily file in chunks to avoid memory issues, collects matching rows,
# then sorts the (much smaller) filtered result in memory before saving.
#
# Usage:
#   python3 preprocess_03_filter_sort_daily.py
#
# Input:  data/preprocess_01_output/states_YYYY-MM-DD.csv  (daily flight states)
#         data/preprocess_02_output/aircraftDatabase-2022-06-FixedWingGA.csv
# Output: data/preprocess_03_output/states_YYYY-MM-DD-FixedWingGA.csv

import os
import pandas as pd

# --- Settings ---
GA_DB      = "data/preprocess_02_output/aircraftDatabase-2022-06-FixedWingGA.csv"
DATES      = ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"]
CHUNK_SIZE = 500_000   # rows per read chunk; tune down if memory is tight

# Load the set of valid GA icao24 codes once
ga_icao24 = set(pd.read_csv(GA_DB, usecols=['icao24'], low_memory=False)['icao24'])
print(f"Loaded {len(ga_icao24):,} GA icao24 codes from {GA_DB}\n")

os.makedirs("data/preprocess_03_output", exist_ok=True)

for date in DATES:
    input_path  = f"data/preprocess_01_output/states_{date}.csv"
    output_path = f"data/preprocess_03_output/states_{date}-FixedWingGA.csv"

    print(f"Processing {input_path} ...")

    # Read chunks, keep only GA rows, accumulate
    chunks = []
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
        ga_rows = chunk[chunk['icao24'].isin(ga_icao24)]
        if not ga_rows.empty:
            chunks.append(ga_rows)

    # Combine, sort, save
    df = pd.concat(chunks, ignore_index=True)
    df.sort_values(['icao24', 'time'], inplace=True)
    df.to_csv(output_path, index=False)

    print(f"  Saved {len(df):,} rows to {output_path}\n")

print("Done.")
