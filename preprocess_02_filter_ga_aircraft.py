# preprocess_02_filter_ga_aircraft.py
#
# Filters the OpenSky aircraft database down to GA fixed-wing aircraft
# from four manufacturer families: Cessna, Piper, Cirrus, Diamond.
#
# Two-stage filter:
#   1a. Brand  — keep rows whose manufacturername matches a target brand
#   1b. Model  — within each brand, keep only allowed GA model families
#       (removes jets, turboprops, helicopters that share the same manufacturer)
#
# Usage:
#   python3 preprocess_02_filter_ga_aircraft.py <input.csv> <output.csv>
#
# Example:
#   python3 preprocess_02_filter_ga_aircraft.py \
#       data/archive/aircraftDatabase-2022-06.csv \
#       data/preprocess_02_output/aircraftDatabase-2022-06-FixedWingGA.csv

import sys
import re
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python3 preprocess_02_filter_ga_aircraft.py <input.csv> <output.csv>")
    sys.exit(1)

input_path, output_path = sys.argv[1], sys.argv[2]

df = pd.read_csv(input_path, low_memory=False)
print(f"Loaded {len(df)} rows from {input_path}")


# ===========================================================================
# Step 1a — Brand filter
# Keep rows whose manufacturername contains one of the target brands.
# Reims/Cessna and Fuji/Cessna are licensed Cessna producers; HOAC is
# the original company name for Diamond Aircraft.
# ===========================================================================

brand_pattern = r'cessna|reims|fuji.*cessna|piper|diamond|hoac|cirrus'
df = df[df['manufacturername'].str.contains(brand_pattern, case=False, na=False)].copy()
print(f"After brand filter: {len(df)} rows")


# ===========================================================================
# Step 1b — Model filter
# Each brand has its own regex targeting allowed GA model families.
# (?<!\d) / (?!\d) prevent partial number matches (e.g. 172 matching 1720).
# \b is used where a word boundary is sufficient.
# ===========================================================================

# Cessna: piston singles and light twins; excludes Citations, Caravans, 140/170 etc.
cessna_allowed = r'150|152|162|172|177|180|182|185|206|207|210|310|336|337|340|402|414|421'
cessna_model   = rf'(?<!\d)({cessna_allowed})(?!\d)'

# Piper: J3/J5 Cubs and PA-XX families; excludes PA-31 Navajo, PA-46 Malibu etc.
# J[ -]?[35] handles "J3", "J-3", "J3C-65", "J3L-65", "J5A" etc.
# PA[ -]?XX handles "PA-28", "PA28", "PA-28R", "PA-32RT" etc.
piper_allowed  = r'11|12|16|18|20|22|23|24|28|30|32|34|38|44'
piper_model    = rf'(\bJ[ -]?[35](?!\d)|PA[ -]?({piper_allowed})(?!\d))'

# Cirrus: SR20 and SR22 family (SR22T included); excludes SF50 Vision Jet.
# SR[ -]?2[02] handles "SR20", "SR22", "SR-22", "SR22T" etc.
cirrus_model   = r'\bSR[ -]?2[02]'

# Diamond: DA/DV series, HK36 Super Dimona, Katana brand name, Twin/Diamond Star.
# (?!\d) after model number captures variants like DA42MNG, DA40D without false positives.
diamond_model  = (
    r'(\bDA[ -]?(20|40|42|62)(?!\d)'   # DA20, DA40, DA42, DA62 and variants (DA42MNG etc.)
    r'|\bDV[ -]?2[02]\b'               # DV20, DV22 (Katana)
    r'|\bHK[ -]?36(?!\d)'              # HK36 Super Dimona variants
    r'|\bKatana\b'                     # "Katana" brand name in model string
    r'|\bTwin[ \-]?Star\b'             # "Twin Star" DA42 marketing name
    r'|\bDiamond[ \-]?Star\b)'         # "Diamond Star" DA40 marketing name
)


def match(row):
    """Return True if this aircraft row belongs to an allowed GA model family."""
    mfr   = str(row['manufacturername']).lower()
    model = str(row['model'])

    if re.search(r'cessna|reims|fuji.*cessna', mfr):
        return bool(re.search(cessna_model,  model, re.IGNORECASE))
    if re.search(r'piper', mfr):
        return bool(re.search(piper_model,   model, re.IGNORECASE))
    if re.search(r'cirrus', mfr):
        return bool(re.search(cirrus_model,  model, re.IGNORECASE))
    if re.search(r'diamond|hoac', mfr):
        return bool(re.search(diamond_model, model, re.IGNORECASE))
    return False


df = df[df.apply(match, axis=1)]
print(f"After model filter: {len(df)} rows")

df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
