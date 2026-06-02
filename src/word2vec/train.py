"""
train.py
========
Trainiert ein Word2Vec-Modell (Gensim) auf den tokenisierten Sätzen und
speichert Modell + Trainingsmetadaten.

``gensim`` wird erst innerhalb von :func:`train_model` importiert, damit das
Paket auch ohne installiertes ``gensim`` importierbar bleibt (z. B. für die
reinen Korpus-Tests). Zum Trainieren muss ``gensim`` installiert sein.

Ausführung (vom Projektverzeichnis):
    python -m src.word2vec.train
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .config import MODEL_METADATA_PATH, MODEL_PATH, SENTENCES_JSONL, WORD2VEC_PARAMS
from .corpus import load_tokenized_sentences


def train_model(
    sentences_path: Path = SENTENCES_JSONL,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    params: dict[str, int] | None = None,
) -> dict[str, float | int | str | dict[str, int]]:
    """Trainiert Gensim Word2Vec und gibt die Trainingsmetadaten zurück."""
    from gensim.models import Word2Vec  # lazy: nur beim Training benötigt

    if not sentences_path.exists():
        raise FileNotFoundError(
            f"Aufbereitete Sätze nicht gefunden unter {sentences_path}. "
            f"Zuerst 'python -m src.word2vec.prepare' ausführen."
        )

    training_params = dict(WORD2VEC_PARAMS)
    if params:
        training_params.update(params)

    sentences = load_tokenized_sentences(sentences_path)
    counts = Counter(token for sentence in sentences for token in sentence)

    start = time.perf_counter()
    model = Word2Vec(sentences=sentences, **training_params)
    runtime_seconds = round(time.perf_counter() - start, 3)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    metadata: dict[str, float | int | str | dict[str, int]] = {
        "sentences_path": str(sentences_path),
        "model_path": str(model_path),
        "runtime_seconds": runtime_seconds,
        "sentence_count": len(sentences),
        "token_count": sum(counts.values()),
        "raw_vocabulary_size": len(counts),
        "final_vocabulary_size": len(model.wv),
        "parameters": training_params,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiert ein Gensim-Word2Vec-Modell.")
    parser.add_argument("--sentences", type=Path, default=SENTENCES_JSONL, help="Tokenisierte JSONL-Eingabe.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Ausgabepfad des Modells.")
    parser.add_argument("--metadata", type=Path, default=MODEL_METADATA_PATH, help="Ausgabepfad der Metadaten.")
    parser.add_argument("--vector-size", type=int, default=WORD2VEC_PARAMS["vector_size"])
    parser.add_argument("--window", type=int, default=WORD2VEC_PARAMS["window"])
    parser.add_argument("--min-count", type=int, default=WORD2VEC_PARAMS["min_count"])
    parser.add_argument("--epochs", type=int, default=WORD2VEC_PARAMS["epochs"])
    parser.add_argument("--seed", type=int, default=WORD2VEC_PARAMS["seed"])
    parser.add_argument("--workers", type=int, default=WORD2VEC_PARAMS["workers"])
    parser.add_argument("--sg", type=int, default=WORD2VEC_PARAMS["sg"], choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "seed": args.seed,
        "workers": args.workers,
        "sg": args.sg,
    }
    train_model(args.sentences, args.model, args.metadata, params)


if __name__ == "__main__":
    main()
