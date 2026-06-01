from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT_DIR / ".cache"
ARTIFACT_ROOT = ROOT_DIR / "artifacts"
HF_TOKEN_PATH = ROOT_DIR / "hf_token"

DEFAULT_MODEL_NAME = "distilbert/distilbert-base-uncased"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "distilbert-classifier"
DEFAULT_TOKENIZED_CACHE_DIR = CACHE_ROOT / "test" / "tokenized"
DEFAULT_TOKENIZED_CACHE_META = DEFAULT_TOKENIZED_CACHE_DIR / "dataset.meta.json"

PATHS = {
    "root_dir": ROOT_DIR,
    "cache_root": CACHE_ROOT,
    "artifact_root": ARTIFACT_ROOT,
    "hf_token_path": HF_TOKEN_PATH,
    "default_output_dir": DEFAULT_OUTPUT_DIR,
    "default_tokenized_cache_dir": DEFAULT_TOKENIZED_CACHE_DIR,
    "default_tokenized_cache_meta": DEFAULT_TOKENIZED_CACHE_META,
}

for value in PATHS.values():
    if value.suffix:
        continue
    value.mkdir(parents=True, exist_ok=True)
