from src.word2vec.corpus import (
    clean_text,
    read_leipzig_sentences,
    sample_sentences,
    tokenize,
    tokenize_sentences,
)


def test_read_leipzig_sentences_reads_valid_tab_rows() -> None:
    rows = [
        "1\tDies ist ein Satz.\n",
        "ohne-tab\n",
        "x\tKeine gueltige ID\n",
        "2\tNoch ein Satz.\n",
    ]

    assert list(read_leipzig_sentences(rows)) == ["Dies ist ein Satz.", "Noch ein Satz."]


def test_sample_sentences_is_deterministic() -> None:
    sentences = [f"Satz {index}" for index in range(20)]

    first = sample_sentences(sentences, sample_size=5, seed=42)
    second = sample_sentences(sentences, sample_size=5, seed=42)

    assert first == second
    assert len(first) == 5


def test_clean_text_removes_html_and_collapses_whitespace() -> None:
    assert clean_text("  Text&nbsp; mit <b>HTML</b>   Resten ") == "Text mit HTML Resten"


def test_tokenize_keeps_german_umlauts_and_lowercases() -> None:
    assert tokenize("Wörter, Märkte und GRÖSSERE Unternehmen!") == [
        "wörter",
        "märkte",
        "und",
        "grössere",
        "unternehmen",
    ]


def test_tokenize_sentences_discards_too_short_sentences() -> None:
    tokenized = tokenize_sentences(["Nur", "Zwei Wörter", "Drei kleine Wörter"], min_sentence_tokens=2)

    assert tokenized == [["zwei", "wörter"], ["drei", "kleine", "wörter"]]
