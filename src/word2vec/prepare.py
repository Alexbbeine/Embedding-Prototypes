"""
prepare.py
==========
Liest das Leipzig-Archiv ein, zieht (optional) eine Stichprobe, tokenisiert die
Sätze und speichert sie als JSON Lines. Schreibt zusätzlich Metadaten über die
verwendete Datenbasis.

Ausführung (vom Projektverzeichnis):
    python -m src.word2vec.prepare
    python -m src.word2vec.prepare --sample-size 20000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    ARCHIVE_PATH,
    CORPUS_ID,
    CORPUS_PAGE,
    DEFAULT_MIN_SENTENCE_TOKENS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    PREPARE_METADATA_PATH,
    SENTENCES_JSONL,
)
from .corpus import (
    corpus_stats,
    iter_sentences_from_archive,
    sample_sentences,
    save_tokenized_sentences,
    tokenize_sentences,
)


def prepare_corpus(
    archive_path: Path = ARCHIVE_PATH,
    output_path: Path = SENTENCES_JSONL,
    metadata_path: Path = PREPARE_METADATA_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    min_sentence_tokens: int = DEFAULT_MIN_SENTENCE_TOKENS,
) -> dict[str, int | str]:
    """Bereitet das Korpus auf und schreibt tokenisierte Sätze + Metadaten."""
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Korpusarchiv nicht gefunden unter {archive_path}. "
            f"Zuerst 'python -m src.word2vec.download' ausführen."
        )

    all_sentences = list(iter_sentences_from_archive(archive_path))
    sampled = sample_sentences(all_sentences, sample_size=sample_size, seed=seed)
    tokenized = tokenize_sentences(sampled, min_sentence_tokens=min_sentence_tokens)
    save_tokenized_sentences(tokenized, output_path)

    stats = corpus_stats(tokenized)
    selection_method = "full_corpus" if sample_size >= len(all_sentences) else "random_sample"
    metadata: dict[str, int | str] = {
        "corpus_id": CORPUS_ID,
        "corpus_page": CORPUS_PAGE,
        "archive_path": str(archive_path),
        "output_path": str(output_path),
        "source_sentences": len(all_sentences),
        "selection_method": selection_method,
        "requested_sample_size": sample_size,
        "selected_sentences": len(sampled),
        "seed": seed,
        "min_sentence_tokens": min_sentence_tokens,
        **stats,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bereitet das Leipzig-Korpus auf und tokenisiert es.")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH, help="Pfad zum Leipzig-Korpusarchiv.")
    parser.add_argument("--output", type=Path, default=SENTENCES_JSONL, help="Ausgabepfad (tokenisierte JSONL).")
    parser.add_argument("--metadata", type=Path, default=PREPARE_METADATA_PATH, help="Ausgabepfad (Metadaten-JSON).")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Anzahl zu verwendender Sätze. Standard nutzt den vollständigen 100K-Korpus.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Zufalls-Seed für kleinere Stichproben.")
    parser.add_argument(
        "--min-sentence-tokens",
        type=int,
        default=DEFAULT_MIN_SENTENCE_TOKENS,
        help="Tokenisierte Sätze unterhalb dieser Länge verwerfen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_corpus(
        archive_path=args.archive,
        output_path=args.output,
        metadata_path=args.metadata,
        sample_size=args.sample_size,
        seed=args.seed,
        min_sentence_tokens=args.min_sentence_tokens,
    )


if __name__ == "__main__":
    main()
