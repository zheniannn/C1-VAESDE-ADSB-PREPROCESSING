"""Trajectory pipeline — stages 04–06: generate segments, ENU conversion, outlier cleaning."""

import os
from adsb_preprocess.io_utils import load_config
from adsb_preprocess.pipeline import run_stage_04, run_stage_05, run_stage_06

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "pipeline.yaml")


def main():
    cfg = load_config(_CONFIG_PATH)
    print("=== Stage 04: Generate trajectories ===", flush=True)
    run_stage_04(cfg)
    print("\n=== Stage 05: Convert to ENU ===", flush=True)
    run_stage_05(cfg)
    print("\n=== Stage 06: Clean outliers ===", flush=True)
    run_stage_06(cfg)
    print("\nPipeline complete.", flush=True)


if __name__ == "__main__":
    main()
