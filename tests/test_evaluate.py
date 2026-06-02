from src.word2vec.evaluate import resolve_target_words


class DummyModel:
    def __init__(self, vocabulary: set[str]) -> None:
        self.wv = vocabulary


def test_resolve_target_words_uses_primary_when_available() -> None:
    model = DummyModel({"bank", "banken"})

    assert resolve_target_words(model, {"bank": ["banken"]}) == {"bank": "bank"}


def test_resolve_target_words_uses_fallback_when_primary_missing() -> None:
    model = DummyModel({"banken"})

    assert resolve_target_words(model, {"bank": ["banken"]}) == {"bank": "banken"}


def test_resolve_target_words_skips_missing_target_group() -> None:
    model = DummyModel({"internet"})

    assert resolve_target_words(model, {"bank": ["banken"]}) == {}
