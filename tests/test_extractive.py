from summarization.extractive import ExtractiveSummarizer, split_sentences


def test_split_sentences_and_sentence_limit():
    text = "Cats sleep. Dogs run. Birds fly."
    assert split_sentences(text) == ["Cats sleep.", "Dogs run.", "Birds fly."]
    assert len(ExtractiveSummarizer(2).summarize(text).split(". ")) == 2


def test_short_text_is_preserved():
    text = "One sentence only."
    assert ExtractiveSummarizer().summarize(text) == text
