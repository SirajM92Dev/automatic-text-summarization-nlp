from summarization.evaluation import score_summaries


def test_rouge_perfect_match():
    result = score_summaries(["A short summary."], ["A short summary."])
    assert result["rouge1"]["f1"] == 1.0
    assert result["rougeL"]["f1"] == 1.0
