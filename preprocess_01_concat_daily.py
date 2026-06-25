# preprocess_01_concat_daily.py
#
# Merges 24 hourly state CSV files into a single daily CSV.
# Reads one chunk at a time to avoid loading ~10 GB into memory at once.
#
# To use: change `date` to the target date, then run:
#   python3 preprocess_01_concat_daily.py
#
# Input:  data/archive/states_YYYY-MM-DD-HH.csv  (24 hourly files)
# Output: data/preprocess_01_output/states_YYYY-MM-DD.csv

import glob
import os
import pandas as pd

# --- Settings ---
date       = "2022-06-06"           # date to process (YYYY-MM-DD)
CHUNK_SIZE = 100_000                # rows per chunk; keeps memory usage low

# Find all hourly files for this date, sorted by hour
pattern = f"data/archive/states_{date}-*.csv"
files   = sorted(glob.glob(pattern))
print(f"Found {len(files)} hourly files for {date}")

os.makedirs("data/preprocess_01_output", exist_ok=True)
output     = f"data/preprocess_01_output/states_{date}.csv"
total_rows = 0

with open(output, "w") as out_f:
    for i, filepath in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {filepath}")
        for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE):
            # Write header only for the very first chunk
            chunk.to_csv(out_f, index=False, header=(out_f.tell() == 0))
            total_rows += len(chunk)

print(f"Saved {total_rows} rows to {output}")
