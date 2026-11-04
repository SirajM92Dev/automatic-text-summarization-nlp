"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from .abstractive import AbstractiveSummarizer, GenerationConfig
from .data import load_records
from .extractive import ExtractiveSummarizer
from .pipeline import compare_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare extractive and abstractive summarization.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("extractive", "abstractive"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--text", required=True)
        sub.add_argument("--model", default="t5-small")
        sub.add_argument("--max-sentences", type=int, default=3)
        sub.add_argument("--max-new-tokens", type=int, default=128)
    text = subparsers.add_parser("text", help="Run both methods on one document.")
    text.add_argument("--text", required=True)
    text.add_argument("--model", default="t5-small")
    text.add_argument("--max-sentences", type=int, default=3)
    text.add_argument("--max-new-tokens", type=int, default=128)
    compare = subparsers.add_parser("compare", help="Evaluate both methods on a dataset.")
    compare.add_argument("--dataset", default="cnn_dailymail")
    compare.add_argument("--split", default="test[:20]")
    compare.add_argument("--input")
    compare.add_argument("--output-dir", default="outputs")
    compare.add_argument("--model", default="t5-small")
    compare.add_argument("--max-examples", type=int)
    compare.add_argument("--max-sentences", type=int, default=3)
    compare.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extractive":
        print(ExtractiveSummarizer(args.max_sentences).summarize(args.text))
    elif args.command == "abstractive":
        config = GenerationConfig(model_name=args.model, max_new_tokens=args.max_new_tokens)
        print(AbstractiveSummarizer(config).summarize(args.text))
    elif args.command == "text":
        extractive = ExtractiveSummarizer(args.max_sentences).summarize(args.text)
        config = GenerationConfig(model_name=args.model, max_new_tokens=args.max_new_tokens)
        abstractive = AbstractiveSummarizer(config).summarize(args.text)
        print(json.dumps({"extractive": extractive, "abstractive": abstractive}, indent=2))
    else:
        records = load_records(args.dataset, args.split, args.input, args.max_examples)
        result = compare_records(records, args.model, args.max_sentences, output_dir=args.output_dir)
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
