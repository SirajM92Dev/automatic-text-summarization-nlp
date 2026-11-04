"""Comparison pipeline for both summarization approaches."""

from __future__ import annotations

from typing import Any

from .abstractive import AbstractiveSummarizer, GenerationConfig
from .evaluation import score_summaries, write_artifacts
from .extractive import ExtractiveSummarizer


def compare_records(
    records: list[dict[str, str]],
    model_name: str = "t5-small",
    max_sentences: int = 3,
    generation: GenerationConfig | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    extractive = ExtractiveSummarizer(max_sentences=max_sentences)
    abstractive = AbstractiveSummarizer(generation or GenerationConfig(model_name=model_name))
    examples: list[dict[str, str]] = []
    for record in records:
        article = record["article"]
        examples.append(
            {
                "article": article,
                "reference": record.get("reference", ""),
                "extractive_summary": extractive.summarize(article),
                "abstractive_summary": abstractive.summarize(article),
            }
        )
    references = [row["reference"] for row in examples]
    metrics: dict[str, Any] = {}
    if all(references):
        metrics = {
            "extractive": score_summaries([row["extractive_summary"] for row in examples], references),
            "abstractive": score_summaries([row["abstractive_summary"] for row in examples], references),
        }
    if output_dir:
        write_artifacts(examples, metrics, output_dir)
    return {"metrics": metrics, "examples": examples}
