"""Geodetic utility functions: WGS-84 conversions and haversine distance."""

import numpy as np

# WGS-84 ellipsoid constants
WGS84_A  = 6_378_137.0
WGS84_F  = 1.0 / 298.257_223_563
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2

EARTH_RADIUS_M = 6_371_000


def geodetic_to_ecef(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m:   np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic (lat, lon, ellipsoidal height) to ECEF (X, Y, Z) in metres."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    X = (N + alt_m) * cos_lat * np.cos(lon)
    Y = (N + alt_m) * cos_lat * np.sin(lon)
    Z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return X, Y, Z


def geodetic_to_enu(
    lat_deg:  np.ndarray,
    lon_deg:  np.ndarray,
    alt_m:    np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m:   float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert arrays of geodetic positions to local ENU relative to a reference point.

    The first point of a segment is typically passed as the reference origin.
    Returns East, North, Up displacements in metres.
    """
    X0, Y0, Z0 = geodetic_to_ecef(
        np.array([lat0_deg]), np.array([lon0_deg]), np.array([alt0_m])
    )
    X0, Y0, Z0 = float(X0[0]), float(Y0[0]), float(Z0[0])

    X, Y, Z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    dX, dY, dZ = X - X0, Y - Y0, Z - Z0

    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    sin_lat0 = np.sin(lat0);  cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0);  cos_lon0 = np.cos(lon0)

    E_m = -sin_lon0 * dX + cos_lon0 * dY
    N_m = -sin_lat0 * cos_lon0 * dX - sin_lat0 * sin_lon0 * dY + cos_lat0 * dZ
    U_m =  cos_lat0 * cos_lon0 * dX + cos_lat0 * sin_lon0 * dY + sin_lat0 * dZ

    return E_m, N_m, U_m


def haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine distance (metres) between arrays of consecutive points."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
