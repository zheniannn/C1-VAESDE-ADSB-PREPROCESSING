# preprocess_04_generate_trajectories.py
#
# Builds cleaned trajectory segments from the 4 daily GA state files.
#
# Cleaning steps (applied in order):
#   1. Drop rows with missing lat or lon
#   2. Drop duplicate (icao24, time) rows — keep first occurrence
#   3. Drop rows where onground == True
#   4. Drop rows where geoaltitude is missing
#   5. Drop rows where geoaltitude < 152.4 m (500 ft) or > 9144 m (30 000 ft)
#   6. Sort each aircraft's pings by time
#   7. Split track on time gap > 60 s (ADS-B coverage gap)
#   8. Split track on implied speed > 250 kt (haversine position jump ÷ time delta)
#   9. Discard segments shorter than 5 minutes (300 s)
#
# Altitude in output: geoaltitude (WGS-84)
#
# Input:  data/preprocess_03_output/states_YYYY-MM-DD-FixedWingGA.csv  (4 dates)
#         data/preprocess_02_output/aircraftDatabase-2022-06-FixedWingGA.csv  (for Results.csv)
# Outputs:
#   data/preprocess_04_output/trajectory_segments.csv  — one row per ADS-B ping in a valid segment
#   data/preprocess_04_output/trajectory_summary.csv   — one row per segment
#   data/preprocess_04_output/Results.csv              — summary statistics

import os
import numpy as np
import pandas as pd

# --- Constants ---
KNOTS_TO_MS    = 0.514444
MAX_IMPLIED_MS = 250 * KNOTS_TO_MS   # 250 kt → ~128.6 m/s
MAX_TIME_GAP_S = 60                  # seconds
MIN_DURATION_S = 300                 # 5 minutes
MIN_ALT_M      = 152.4              # 500 ft  — WGS-84 lower bound
MAX_ALT_M      = 9_144.0            # 30 000 ft — WGS-84 upper bound
EARTH_RADIUS_M = 6_371_000

DATES    = ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"]
SEG_COLS = ['time', 'icao24', 'lat', 'lon', 'velocity', 'heading',
            'vertrate', 'callsign', 'squawk', 'altitude', 'date', 'segment_id']

SEG_PATH = "data/preprocess_04_output/trajectory_segments.csv"
SUM_PATH = "data/preprocess_04_output/trajectory_summary.csv"
RES_PATH = "data/preprocess_04_output/Results.csv"


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance (metres) between arrays of consecutive points."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# Accumulators for drop counts across all days
drops = {
    'raw':          0,
    'lat_lon':      0,
    'duplicates':   0,
    'onground':     0,
    'alt_missing':  0,
    'alt_range':    0,
    'after_filter': 0,
}
seg_pings_discarded = 0   # pings in segments < 5 min

# Initialise output files
os.makedirs("data/preprocess_04_output", exist_ok=True)
pd.DataFrame(columns=SEG_COLS).to_csv(SEG_PATH, index=False)

summary_rows      = []
global_seg_id     = 0
observed_aircraft = set()

# ===========================================================================
# Main loop — process one day at a time to limit peak memory use
# ===========================================================================
for date in DATES:
    path = f"data/preprocess_03_output/states_{date}-FixedWingGA.csv"
    print(f"\n{'='*55}")
    print(f"  {date}")
    print(f"{'='*55}")

    df = pd.read_csv(path, low_memory=False)
    n0 = len(df)
    drops['raw'] += n0

    # Step 1 — drop rows with missing lat or lon
    df = df.dropna(subset=['lat', 'lon'])
    n1 = len(df)
    print(f"  Step 1  missing lat/lon          dropped: {n0-n1:>10,}   remaining: {n1:,}")

    # Step 2 — drop duplicate (icao24, time) rows
    df = df.drop_duplicates(subset=['icao24', 'time'], keep='first')
    n2 = len(df)
    print(f"  Step 2  duplicate (icao24,time)  dropped: {n1-n2:>10,}   remaining: {n2:,}")

    # Step 3 — drop on-ground pings
    df = df[~df['onground']]
    n3 = len(df)
    print(f"  Step 3  onground == True         dropped: {n2-n3:>10,}   remaining: {n3:,}")

    # Step 4 — drop rows where geoaltitude is missing
    df = df.dropna(subset=['geoaltitude'])
    n4 = len(df)
    print(f"  Step 4  missing geoaltitude      dropped: {n3-n4:>10,}   remaining: {n4:,}")

    # Step 5 — drop rows outside geoaltitude window [152.4 m, 9144 m]
    df = df[(df['geoaltitude'] >= MIN_ALT_M) & (df['geoaltitude'] <= MAX_ALT_M)]
    n5 = len(df)
    print(f"  Step 5  geoalt out of range      dropped: {n4-n5:>10,}   remaining: {n5:,}")

    # Altitude output column = geoaltitude (already validated above)
    df['altitude'] = df['geoaltitude']
    df['callsign'] = df['callsign'].astype(str).str.strip()
    df['date']     = date

    drops['lat_lon']     += n0 - n1
    drops['duplicates']  += n1 - n2
    drops['onground']    += n2 - n3
    drops['alt_missing'] += n3 - n4
    drops['alt_range']   += n4 - n5
    drops['after_filter'] += n5

    # --- Trajectory generation (steps 6–9) ---
    day_segs      = []
    day_seg_count = 0
    time_splits   = 0
    speed_splits  = 0
    day_pings_kept      = 0
    day_pings_discarded = 0

    for icao24, grp in df.groupby('icao24', sort=False):
        # Step 6 — sort pings by time
        grp = grp.sort_values('time').reset_index(drop=True)
        t   = grp['time'].values

        # Step 7 — time gap splits
        dt       = np.diff(t, prepend=t[0])
        time_gap = dt > MAX_TIME_GAP_S

        # Step 8 — implied speed splits (haversine distance ÷ time delta)
        lat  = grp['lat'].values
        lon  = grp['lon'].values
        dist = np.empty(len(grp))
        dist[0] = 0
        if len(grp) > 1:
            dist[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])

        with np.errstate(invalid='ignore', divide='ignore'):
            implied_speed = np.where(dt > 0, dist / dt, 0)
        speed_jump = implied_speed > MAX_IMPLIED_MS

        split    = time_gap | speed_jump
        split[0] = False
        seg_nums = split.cumsum()

        time_splits  += int(time_gap[1:].sum())
        speed_splits += int(speed_jump[1:].sum())

        # Step 9 — discard segments shorter than 5 minutes
        for _, seg in grp.groupby(seg_nums, sort=False):
            duration = int(seg['time'].iloc[-1]) - int(seg['time'].iloc[0])
            if duration < MIN_DURATION_S:
                day_pings_discarded += len(seg)
                continue

            seg = seg.copy()
            seg['segment_id'] = global_seg_id

            summary_rows.append({
                'segment_id'      : global_seg_id,
                'icao24'          : icao24,
                'callsign'        : seg['callsign'].iloc[0],
                'start_time'      : int(seg['time'].iloc[0]),
                'end_time'        : int(seg['time'].iloc[-1]),
                'duration_seconds': duration,
                'n_points'        : len(seg),
                'mean_alt_m'      : round(seg['altitude'].mean(), 2),
                'median_alt_m'    : round(seg['altitude'].median(), 2),
                'date'            : date,
            })

            day_segs.append(seg[SEG_COLS])
            day_pings_kept += len(seg)
            global_seg_id  += 1
            day_seg_count  += 1
            observed_aircraft.add(icao24)

    seg_pings_discarded += day_pings_discarded

    if day_segs:
        pd.concat(day_segs, ignore_index=True).to_csv(
            SEG_PATH, mode='a', header=False, index=False
        )

    print(f"  Step 7  time gap splits:         {time_splits:>10,}")
    print(f"  Step 8  speed splits:            {speed_splits:>10,}")
    print(f"  Step 9  pings in short segments  dropped: {day_pings_discarded:>10,}")
    print(f"  Segments kept: {day_seg_count:,}  |  pings kept: {day_pings_kept:,}"
          f"  |  running total segments: {global_seg_id:,}")

# ===========================================================================
# Summary across all days
# ===========================================================================
print(f"\n{'='*55}")
print("  TOTALS ACROSS ALL DAYS")
print(f"{'='*55}")
print(f"  Raw rows loaded:                 {drops['raw']:>12,}")
print(f"  Step 1  missing lat/lon:         {drops['lat_lon']:>12,}  dropped")
print(f"  Step 2  duplicate (icao24,time): {drops['duplicates']:>12,}  dropped")
print(f"  Step 3  onground:                {drops['onground']:>12,}  dropped")
print(f"  Step 4  missing geoaltitude:     {drops['alt_missing']:>12,}  dropped")
print(f"  Step 5  geoalt out of range:     {drops['alt_range']:>12,}  dropped")
print(f"  After row filters:               {drops['after_filter']:>12,}  remaining")
print(f"  Step 9  short-segment pings:     {seg_pings_discarded:>12,}  dropped")
print(f"  Final ADS-B pings in output:     {drops['after_filter']-seg_pings_discarded:>12,}")

# ===========================================================================
# Write outputs
# ===========================================================================
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(SUM_PATH, index=False)
print(f"\nSaved {len(df_sum):,} rows to {SUM_PATH}")

ga_registry = len(
    pd.read_csv("data/preprocess_02_output/aircraftDatabase-2022-06-FixedWingGA.csv", usecols=['icao24'])
)
dur_min = df_sum['duration_seconds'] / 60

results = pd.DataFrame({
    'Metric': [
        'Aircraft in GA registry',
        'Aircraft observed flying (across 4 days)',
        'Trajectory segments',
        'ADS-B pings in output',
        'Mean segment duration',
        'Median segment duration',
        'Max segment duration',
    ],
    'Value': [
        ga_registry,
        len(observed_aircraft),
        global_seg_id,
        int(df_sum['n_points'].sum()),
        f"{dur_min.mean():.1f} min",
        f"{dur_min.median():.1f} min",
        f"{dur_min.max():.1f} min",
    ]
})
results.to_csv(RES_PATH, index=False)

print(f"\n=== Results ===")
for _, row in results.iterrows():
    print(f"  {row['Metric']}: {row['Value']}")
