from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT_DIR / ".cache"
ARTIFACT_ROOT = ROOT_DIR / "artifacts"
HF_TOKEN_PATH = ROOT_DIR / "hf_token"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


for path in (ROOT_DIR, CACHE_ROOT, ARTIFACT_ROOT):
    path.mkdir(parents=True, exist_ok=True)
