"""Trajectory segment generation from raw GA ADS-B state data."""

import numpy as np
import pandas as pd

from adsb_preprocess.geo import haversine_m

KNOTS_TO_MS = 0.514444

SEG_COLS = [
    'time', 'icao24', 'lat', 'lon', 'velocity', 'heading',
    'vertrate', 'callsign', 'squawk', 'altitude', 'date', 'segment_id',
]


def clean_day(df: pd.DataFrame, date: str, min_alt_m: float, max_alt_m: float) -> tuple[pd.DataFrame, dict]:
    """
    Apply row-level quality filters to one day's GA state data.

    Returns the filtered DataFrame and a dict of drop counts per step.
    """
    drops = {}
    n0 = len(df)

    df = df.dropna(subset=['lat', 'lon'])
    n1 = len(df)

    df = df.drop_duplicates(subset=['icao24', 'time'], keep='first')
    n2 = len(df)

    df = df[~df['onground']]
    n3 = len(df)

    df = df.dropna(subset=['geoaltitude'])
    n4 = len(df)

    df = df[(df['geoaltitude'] >= min_alt_m) & (df['geoaltitude'] <= max_alt_m)]
    n5 = len(df)

    df['altitude'] = df['geoaltitude']
    df['callsign'] = df['callsign'].astype(str).str.strip()
    df['date']     = date

    drops = {
        'raw':         n0,
        'lat_lon':     n0 - n1,
        'duplicates':  n1 - n2,
        'onground':    n2 - n3,
        'alt_missing': n3 - n4,
        'alt_range':   n4 - n5,
        'after_filter': n5,
    }
    return df, drops


def split_into_segments(
    df: pd.DataFrame,
    global_seg_id: int,
    max_time_gap_s: float,
    max_implied_ms: float,
    min_duration_s: float,
) -> tuple[list[pd.DataFrame], list[dict], int, int, int, int]:
    """
    Split a filtered day DataFrame into trajectory segments per aircraft.

    Returns (seg_frames, summary_rows, global_seg_id, time_splits, speed_splits, pings_discarded).
    """
    seg_frames      = []
    summary_rows    = []
    time_splits     = 0
    speed_splits    = 0
    pings_discarded = 0

    for icao24, grp in df.groupby('icao24', sort=False):
        grp = grp.sort_values('time').reset_index(drop=True)
        t   = grp['time'].values
        dt  = np.diff(t, prepend=t[0])

        time_gap = dt > max_time_gap_s

        lat  = grp['lat'].values
        lon  = grp['lon'].values
        dist = np.empty(len(grp))
        dist[0] = 0
        if len(grp) > 1:
            dist[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])

        with np.errstate(invalid='ignore', divide='ignore'):
            implied_speed = np.where(dt > 0, dist / dt, 0)

        speed_jump = implied_speed > max_implied_ms

        split    = time_gap | speed_jump
        split[0] = False
        seg_nums = split.cumsum()

        time_splits  += int(time_gap[1:].sum())
        speed_splits += int(speed_jump[1:].sum())

        for _, seg in grp.groupby(seg_nums, sort=False):
            duration = int(seg['time'].iloc[-1]) - int(seg['time'].iloc[0])
            if duration < min_duration_s:
                pings_discarded += len(seg)
                continue

            seg = seg.copy()
            seg['segment_id'] = global_seg_id

            summary_rows.append({
                'segment_id':       global_seg_id,
                'icao24':           icao24,
                'callsign':         seg['callsign'].iloc[0],
                'start_time':       int(seg['time'].iloc[0]),
                'end_time':         int(seg['time'].iloc[-1]),
                'duration_seconds': duration,
                'n_points':         len(seg),
                'mean_alt_m':       round(seg['altitude'].mean(), 2),
                'median_alt_m':     round(seg['altitude'].median(), 2),
                'date':             seg['date'].iloc[0],
            })

            seg_frames.append(seg[SEG_COLS])
            global_seg_id += 1

    return seg_frames, summary_rows, global_seg_id, time_splits, speed_splits, pings_discarded
