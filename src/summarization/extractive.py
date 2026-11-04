"""Extractive TF-IDF centrality summarizer."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text.strip()) if sentence.strip()]


class ExtractiveSummarizer:
    """Select the most central sentences while retaining their original order."""

    def __init__(self, max_sentences: int = 3) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        self.max_sentences = max_sentences

    def summarize(self, text: str) -> str:
        sentences = split_sentences(text)
        if len(sentences) <= self.max_sentences:
            return " ".join(sentences)
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
            centroid = matrix.mean(axis=0)
            scores = np.asarray(matrix @ centroid.T).ravel()
        except ValueError:
            scores = np.asarray([len(sentence.split()) for sentence in sentences], dtype=float)
        selected = np.argsort(scores)[-self.max_sentences :]
        return " ".join(sentences[index] for index in sorted(selected))
