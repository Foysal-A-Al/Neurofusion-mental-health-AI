from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path | None = None) -> dict:
    # Application assets live in the checkout rather than inside the installed wheel.
    cfg_path = Path(path) if path else Path("configs/config.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
