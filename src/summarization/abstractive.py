"""Hugging Face Transformer abstractive summarization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "t5-small"
    max_input_tokens: int = 1024
    max_new_tokens: int = 128
    min_new_tokens: int = 24
    num_beams: int = 4
    device: int = -1


class AbstractiveSummarizer:
    """Lazy-loading wrapper around the Hugging Face summarization pipeline."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "summarization",
                model=self.config.model_name,
                tokenizer=self.config.model_name,
                device=self.config.device,
            )
        return self._pipeline

    def summarize(self, text: str) -> str:
        if not text.strip():
            raise ValueError("Cannot summarize an empty document.")
        prompt = text if self.config.model_name.lower().startswith(("facebook/bart", "google/pegasus")) else f"summarize: {text}"
        result = self._get_pipeline()(
            prompt,
            truncation=True,
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=min(self.config.min_new_tokens, self.config.max_new_tokens),
            num_beams=self.config.num_beams,
            do_sample=False,
        )
        return result[0]["summary_text"].strip()
