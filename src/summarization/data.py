"""Dataset loading and normalization utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ARTICLE_FIELDS = ("article", "text", "document", "content")
SUMMARY_FIELDS = ("highlights", "summary", "target", "reference")


def _first_value(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_record(record: dict[str, Any]) -> dict[str, str]:
    article = _first_value(record, ARTICLE_FIELDS)
    if not article:
        raise ValueError(f"Record is missing a non-empty article field: {ARTICLE_FIELDS}")
    return {"article": article, "reference": _first_value(record, SUMMARY_FIELDS)}


def load_local_records(path: str | Path) -> list[dict[str, str]]:
    """Load JSON, JSONL, or CSV records with article and optional reference fields."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {source}")
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            raw_records = list(csv.DictReader(handle))
    elif source.suffix.lower() == ".jsonl":
        raw_records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        raw_records = payload if isinstance(payload, list) else payload.get("data", [])
    else:
        raise ValueError("Input must be .json, .jsonl, or .csv")
    return [normalize_record(record) for record in raw_records]


def load_records(
    dataset: str = "cnn_dailymail",
    split: str = "test[:20]",
    input_path: str | None = None,
    max_examples: int | None = None,
) -> list[dict[str, str]]:
    """Load normalized records from a local file or Hugging Face Datasets."""
    if input_path:
        records = load_local_records(input_path)
    else:
        from datasets import load_dataset

        dataset_split = load_dataset(dataset, "3.0.0", split=split)
        records = [normalize_record(dict(row)) for row in dataset_split]
    if max_examples is not None:
        records = records[:max_examples]
    if not records:
        raise ValueError("The dataset contains no usable records.")
    return records
