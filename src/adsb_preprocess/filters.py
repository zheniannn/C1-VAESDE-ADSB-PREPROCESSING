"""GA aircraft brand/model filter for the OpenSky aircraft database."""

import re
import pandas as pd

_BRAND_PATTERN = r'cessna|reims|fuji.*cessna|piper|diamond|hoac|cirrus'

# Cessna: piston singles and light twins; excludes Citations, Caravans
_CESSNA_ALLOWED = r'150|152|162|172|177|180|182|185|206|207|210|310|336|337|340|402|414|421'
_CESSNA_MODEL   = rf'(?<!\d)({_CESSNA_ALLOWED})(?!\d)'

# Piper: J3/J5 Cubs and PA-XX families; excludes PA-31 Navajo, PA-46 Malibu etc.
_PIPER_ALLOWED = r'11|12|16|18|20|22|23|24|28|30|32|34|38|44'
_PIPER_MODEL   = rf'(\bJ[ -]?[35](?!\d)|PA[ -]?({_PIPER_ALLOWED})(?!\d))'

# Cirrus: SR20 and SR22 family; excludes SF50 Vision Jet
_CIRRUS_MODEL = r'\bSR[ -]?2[02]'

# Diamond: DA/DV series, HK36 Super Dimona, Katana, Twin/Diamond Star
_DIAMOND_MODEL = (
    r'(\bDA[ -]?(20|40|42|62)(?!\d)'
    r'|\bDV[ -]?2[02]\b'
    r'|\bHK[ -]?36(?!\d)'
    r'|\bKatana\b'
    r'|\bTwin[ \-]?Star\b'
    r'|\bDiamond[ \-]?Star\b)'
)


def _match_row(row) -> bool:
    """Return True if this aircraft belongs to an allowed GA model family."""
    mfr   = str(row['manufacturername']).lower()
    model = str(row['model'])
    if re.search(r'cessna|reims|fuji.*cessna', mfr):
        return bool(re.search(_CESSNA_MODEL,  model, re.IGNORECASE))
    if re.search(r'piper', mfr):
        return bool(re.search(_PIPER_MODEL,   model, re.IGNORECASE))
    if re.search(r'cirrus', mfr):
        return bool(re.search(_CIRRUS_MODEL,  model, re.IGNORECASE))
    if re.search(r'diamond|hoac', mfr):
        return bool(re.search(_DIAMOND_MODEL, model, re.IGNORECASE))
    return False


def filter_ga_aircraft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter an OpenSky aircraft database DataFrame to GA fixed-wing aircraft.

    Applies a two-stage filter: brand (Cessna, Piper, Cirrus, Diamond),
    then model family within each brand.
    """
    df = df[df['manufacturername'].str.contains(_BRAND_PATTERN, case=False, na=False)].copy()
    return df[df.apply(_match_row, axis=1)]
