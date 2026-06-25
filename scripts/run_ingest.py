"""Ingest pipeline — stages 01–03: concat, filter aircraft DB, filter and sort daily."""

import os
from adsb_preprocess.io_utils import load_config
from adsb_preprocess.ingest   import run_stage_01, run_stage_02, run_stage_03

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "pipeline.yaml")


def main():
    cfg = load_config(_CONFIG_PATH)
    print("=== Stage 01: Concat daily ===", flush=True)
    run_stage_01(cfg)
    print("=== Stage 02: Filter aircraft DB ===", flush=True)
    run_stage_02(cfg)
    print("=== Stage 03: Filter and sort daily ===", flush=True)
    run_stage_03(cfg)
    print("Ingest complete.", flush=True)


if __name__ == "__main__":
    main()
