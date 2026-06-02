"""
pipeline.py
===========
Führt den vollständigen Word2Vec-Ablauf aus:

    download → prepare → train → evaluate

Ausführung (vom Projektverzeichnis):
    python -m src.word2vec.pipeline
    python -m src.word2vec.pipeline --skip-download   # vorhandenes Archiv nutzen

Erfordert installiertes ``gensim`` (Training/Auswertung). Im Docker-Container
ist dies bereits enthalten – siehe src/word2vec/Dockerfile.
"""
from __future__ import annotations

import argparse

from .download import download_corpus
from .evaluate import evaluate_model
from .prepare import prepare_corpus
from .train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Führt den vollständigen Word2Vec-Prototyp-Ablauf aus.")
    parser.add_argument("--skip-download", action="store_true", help="Vorhandenes Archiv in data/raw verwenden.")
    args = parser.parse_args()

    if not args.skip_download:
        download_corpus()
    prepare_corpus()
    train_model()
    evaluate_model()


if __name__ == "__main__":
    main()
