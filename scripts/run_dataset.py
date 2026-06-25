"""Dataset builder — stages 07–08: train/test split, sequence array generation."""

import os
from adsb_preprocess.io_utils import load_config
from adsb_preprocess.dataset  import run_stage_07, run_stage_08

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "pipeline.yaml")


def main():
    cfg = load_config(_CONFIG_PATH)
    print("=== Stage 07: Train/test split ===", flush=True)
    run_stage_07(cfg)
    print("\n=== Stage 08: Make sequence dataset ===", flush=True)
    run_stage_08(cfg)
    print("\nDataset complete.", flush=True)


if __name__ == "__main__":
    main()
