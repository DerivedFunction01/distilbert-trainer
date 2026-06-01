from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.paths import ROOT_DIR, resolve_repo_path


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str | Path) -> Path:
    return resolve_repo_path(value)
