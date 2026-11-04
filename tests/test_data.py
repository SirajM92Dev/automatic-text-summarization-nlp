import json

from summarization.data import load_local_records


def test_jsonl_normalization(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps({"article": "Article", "highlights": "Summary"}) + "\n",
        encoding="utf-8",
    )
    assert load_local_records(source) == [{"article": "Article", "reference": "Summary"}]
