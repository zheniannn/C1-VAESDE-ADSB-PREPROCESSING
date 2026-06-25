"""I/O helpers: config loading and shared file utilities."""

import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)
