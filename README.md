# Automatic Text Summarization Using NLP

An end-to-end comparison of **extractive** and **abstractive** text summarization. The project uses a transparent TF-IDF extractive baseline and a pretrained Hugging Face Transformer (default: `t5-small`) for abstractive generation, then evaluates both with ROUGE and exports examples for human inspection.

## Features

- Uses the CNN/DailyMail dataset through Hugging Face Datasets, or a local JSON/JSONL/CSV file.
- Extractive baseline ranks sentences with TF-IDF centrality and preserves source order.
- Abstractive summarization with T5 or BART via `transformers`.
- ROUGE-1, ROUGE-2, and ROUGE-L evaluation with bootstrap confidence intervals.
- JSON/CSV output suitable for review, reporting, or downstream analysis.
- CPU-friendly defaults, deterministic seeds, validation, tests, and GitHub Actions CI.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Download a small evaluation split and compare both systems.
summarize compare --dataset cnn_dailymail --split test[:20] --output-dir outputs

# Summarize one document.
summarize text --text "A long article goes here..." --model t5-small
```

The first Transformer run downloads the selected model from Hugging Face. For a stronger model, use `facebook/bart-large-cnn` (more memory and time required).

## Commands

```text
summarize compare   Compare extractive and abstractive summaries and score ROUGE
summarize extractive  Generate an extractive summary
summarize abstractive Generate an abstractive summary
summarize text      Run both methods for one text
```

Useful options include `--dataset`, `--split`, `--input`, `--model`, `--max-examples`, `--max-sentences`, `--max-new-tokens`, `--seed`, and `--device`.

## Local data format

JSON/JSONL/CSV input must contain an article field and may contain a reference summary:

```json
{"article": "Long article text...", "highlights": "Reference summary..."}
```

Accepted summary field names are `highlights`, `summary`, `target`, and `reference`.

## Project layout

```text
src/summarization/
  cli.py          Command-line interface
  data.py         Dataset loading and normalization
  extractive.py   TF-IDF sentence-ranking baseline
  abstractive.py  Hugging Face generation wrapper
  evaluation.py   ROUGE scoring and output persistence
  pipeline.py     Comparison orchestration
tests/            Unit tests for deterministic core behavior
```

## Evaluation and human inspection

`compare` writes:

- `metrics.json`: aggregate ROUGE scores for both methods.
- `examples.jsonl`: article, reference, extractive summary, and abstractive summary for review.
- `examples.csv`: spreadsheet-friendly version of the same examples.

ROUGE is an automatic metric, not a substitute for factuality or readability review. Inspect the exported examples for omissions, hallucinations, repetition, and grammatical quality.

## Development

```bash
pytest
ruff check .
```

## License

Released under the MIT License. See `LICENSE`.
