"""Smoke tests — verify imports and core logic functions are callable."""

import numpy as np
import pandas as pd


def test_imports():
    from adsb_preprocess.geo         import geodetic_to_ecef, geodetic_to_enu, haversine_m
    from adsb_preprocess.filters     import filter_ga_aircraft
    from adsb_preprocess.trajectories import clean_day, split_into_segments
    from adsb_preprocess.enu         import process_one_segment, convert_all_segments
    from adsb_preprocess.cleaning    import process_segment
    from adsb_preprocess.sequences   import load_and_validate, make_sequences
    from adsb_preprocess.io_utils    import load_config


def test_haversine_zero():
    from adsb_preprocess.geo import haversine_m
    d = haversine_m(
        np.array([51.5]), np.array([-0.1]),
        np.array([51.5]), np.array([-0.1]),
    )
    assert float(d[0]) == 0.0


def test_geodetic_to_enu_origin():
    from adsb_preprocess.geo import geodetic_to_enu
    E, N, U = geodetic_to_enu(
        np.array([51.5]), np.array([-0.1]), np.array([100.0]),
        51.5, -0.1, 100.0,
    )
    assert abs(float(E[0])) < 1e-3
    assert abs(float(N[0])) < 1e-3
    assert abs(float(U[0])) < 1e-3


def test_filter_ga_aircraft_keeps_cessna():
    from adsb_preprocess.filters import filter_ga_aircraft
    df = pd.DataFrame([
        {"manufacturername": "CESSNA", "model": "172"},
        {"manufacturername": "CESSNA", "model": "CitationX"},
        {"manufacturername": "BOEING", "model": "737"},
    ])
    result = filter_ga_aircraft(df)
    assert len(result) == 1
    assert result.iloc[0]["model"] == "172"


def test_make_sequences_shape():
    from adsb_preprocess.sequences import make_sequences
    rng = np.random.default_rng(0)
    features = ["E_m", "N_m", "vE_mps", "vN_mps"]
    n = 50
    df = pd.DataFrame({
        "segment_id": np.zeros(n, dtype=int),
        "time":       np.arange(n, dtype=float),
        "E_m":        rng.normal(size=n),
        "N_m":        rng.normal(size=n),
        "vE_mps":     rng.normal(size=n),
        "vN_mps":     rng.normal(size=n),
    })
    mean = df[features].mean()
    std  = df[features].std().replace(0, 1)
    X, meta, used, skipped = make_sequences(df, "train", features, mean, std, seq_len=30, stride=5)
    assert X.ndim == 3
    assert X.shape[1] == 30
    assert X.shape[2] == 4
    assert not np.isnan(X).any()
