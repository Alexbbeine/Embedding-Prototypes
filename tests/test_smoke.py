import json
from pathlib import Path

import pytest

pytest.importorskip("gensim")
pytest.importorskip("matplotlib")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("tabulate")

from src.word2vec.evaluate import evaluate_model
from src.word2vec.train import train_model


def write_tiny_corpus(path: Path) -> None:
    sentences = [
        ["regierung", "wahl", "partei", "politik", "bundestag", "minister"],
        ["markt", "unternehmen", "bank", "wirtschaft", "kredit", "handel"],
        ["daten", "software", "modell", "netz", "internet", "system"],
        ["bank", "kredit", "kunden", "geld", "markt", "unternehmen"],
        ["modell", "daten", "software", "system", "technik", "netz"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for _ in range(8):
            for sentence in sentences:
                file.write(json.dumps(sentence, ensure_ascii=False) + "\n")


def test_training_and_evaluation_smoke(tmp_path: Path) -> None:
    sentences_path = tmp_path / "sentences.jsonl"
    model_path = tmp_path / "model.model"
    metadata_path = tmp_path / "metadata.json"
    neighbors_csv = tmp_path / "neighbors.csv"
    neighbors_md = tmp_path / "neighbors.md"
    targets_json = tmp_path / "targets.json"
    figure_path = tmp_path / "pca.png"
    target_figure_dir = tmp_path / "targets"

    write_tiny_corpus(sentences_path)

    train_model(
        sentences_path=sentences_path,
        model_path=model_path,
        metadata_path=metadata_path,
        params={"vector_size": 20, "window": 3, "min_count": 1, "epochs": 5, "workers": 1, "sg": 1, "seed": 42},
    )
    evaluate_model(
        model_path=model_path,
        neighbors_csv=neighbors_csv,
        neighbors_md=neighbors_md,
        targets_json=targets_json,
        pca_figure=figure_path,
        target_figure_dir=target_figure_dir,
        topn=2,
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert neighbors_csv.exists()
    assert neighbors_md.exists()
    assert targets_json.exists()
    assert figure_path.exists()
