from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

from shared.building import rows_to_dataset_dict
from shared.fetch import load_hf_dataset, load_hf_dataset_dict
from shared.paths import ROOT_DIR
from shared.tokenization import (
    load_tokenized_dataset_cache,
    save_tokenized_dataset_cache,
    tokenize_dataset_dict,
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_dataset_from_config(config: dict[str, Any]) -> DatasetDict:
    source = config["dataset"]["source"]
    source_type = source["type"]

    if source_type == "hf_dataset_dict":
        dataset = load_hf_dataset_dict(source["dataset_name"], config=source.get("config"))
        if isinstance(dataset, DatasetDict):
            return dataset
        raise TypeError("Expected DatasetDict from hf_dataset_dict source")

    if source_type == "hf_dataset":
        split = source["split"]
        dataset = load_hf_dataset(source["dataset_name"], split=split, config=source.get("config"))
        if not isinstance(dataset, Dataset):
            raise TypeError("Expected Dataset from hf_dataset source")
        rows = dataset.to_list()
        return rows_to_dataset_dict(
            rows,
            val_size=config["dataset"].get("val_size"),
            test_size=config["dataset"].get("test_size"),
            seed=config["dataset"].get("seed", 42),
            columns=dataset.column_names,
        )

    if source_type == "local_parquet":
        loaded = load_from_disk(str(_resolve_path(source["path"])))
        if isinstance(loaded, DatasetDict):
            return loaded
        raise TypeError("Expected DatasetDict from local_parquet source")

    raise ValueError(f"Unknown dataset source type: {source_type!r}")


def build_and_cache_dataset(config: dict[str, Any]) -> DatasetDict:
    dataset_cfg = config["dataset"]
    cache_dir = _resolve_path(dataset_cfg["cache_dir"])
    meta_path = cache_dir / "dataset.meta.json"
    expected_meta = {
        "dataset": dataset_cfg,
        "tokenization": config["tokenization"],
    }
    cached = load_tokenized_dataset_cache(
        str(cache_dir),
        meta_path=str(meta_path),
        expected_meta=expected_meta,
    )
    if cached is not None:
        return cached

    raw_dataset = load_dataset_from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenized = tokenize_dataset_dict(
        raw_dataset,
        tokenizer=tokenizer,
        kind=config["tokenization"]["kind"],
        text_columns=tuple(config["tokenization"]["text_columns"]),
        max_length=config["tokenization"]["max_length"],
        padding=config["tokenization"].get("padding", "max_length"),
    )
    save_tokenized_dataset_cache(
        tokenized,
        str(cache_dir),
        meta_path=str(meta_path),
        meta=expected_meta,
        overwrite=True,
    )
    return tokenized
