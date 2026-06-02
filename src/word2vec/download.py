"""
download.py
===========
Lädt das Leipzig-Korpusarchiv (deu_news_2010_100K) herunter.

Ausführung (vom Projektverzeichnis):
    python -m src.word2vec.download
    python -m src.word2vec.download --force
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .config import ARCHIVE_PATH, CORPUS_URL


def download_corpus(
    url: str = CORPUS_URL,
    destination: Path = ARCHIVE_PATH,
    force: bool = False,
) -> Path:
    """Lädt das Korpusarchiv, sofern es nicht bereits vorliegt."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Archiv existiert bereits: {destination}")
        return destination

    print(f"Lade herunter: {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Archiv gespeichert unter: {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lädt das Leipzig-Korpus für den Word2Vec-Prototyp herunter.")
    parser.add_argument("--url", default=CORPUS_URL, help="URL des Korpusarchivs.")
    parser.add_argument("--output", type=Path, default=ARCHIVE_PATH, help="Zielpfad des Archivs.")
    parser.add_argument("--force", action="store_true", help="Vorhandenes Archiv überschreiben.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_corpus(args.url, args.output, args.force)


if __name__ == "__main__":
    main()
