"""ROUGE evaluation and human-inspection artifact writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer


def score_summaries(predictions: list[str], references: list[str]) -> dict[str, dict[str, float]]:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Predictions and references must be non-empty and have equal length.")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals: dict[str, list[float]] = {metric: [] for metric in ("rouge1", "rouge2", "rougeL")}
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for metric in totals:
            totals[metric].append(scores[metric].fmeasure)
    return {metric: {"f1": sum(values) / len(values)} for metric, values in totals.items()}


def write_artifacts(records: list[dict[str, Any]], metrics: dict[str, Any], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (destination / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if records:
        with (destination / "examples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
