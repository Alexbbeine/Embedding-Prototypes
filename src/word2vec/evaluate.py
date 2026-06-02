"""
evaluate.py
===========
Nachbarschaftsanalyse des trainierten Word2Vec-Modells.

Für jedes Zielwort werden die ``topn`` nächsten Nachbarn (Kosinusähnlichkeit)
bestimmt und als CSV/Markdown gespeichert. Zusätzlich werden PCA-Grafiken über
:mod:`src.word2vec.visualize` erzeugt. Fehlt ein Zielwort im Vokabular, wird ein
konfiguriertes Ersatzwort verwendet (siehe ``target_words`` in der YAML-Konfig).

``gensim`` wird erst innerhalb von :func:`evaluate_model` importiert, damit das
Paket auch ohne installiertes ``gensim`` importierbar bleibt.

Ausführung (vom Projektverzeichnis):
    python -m src.word2vec.evaluate
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_TOPN,
    MODEL_PATH,
    NEIGHBORS_CSV,
    NEIGHBORS_MD,
    PCA_FIGURE,
    TARGET_FIGURE_DIR,
    TARGET_WORDS,
    TARGETS_JSON,
)
from .visualize import plot_pca, plot_target_pcas


def resolve_target_words(model: Any, target_words: dict[str, list[str]] = TARGET_WORDS) -> dict[str, str]:
    """Ordnet jedem Zielwort das erste im Vokabular vorhandene (Ersatz-)Wort zu."""
    resolved: dict[str, str] = {}
    for primary, fallbacks in target_words.items():
        candidates = [primary, *fallbacks]
        match = next((candidate for candidate in candidates if candidate in model.wv), None)
        if match is not None:
            resolved[primary] = match
    return resolved


def classify_neighbor(target: str, neighbor: str) -> str:
    """Platzhalter-Einordnung; wird nach Sichtung manuell fachlich ergänzt."""
    return "manuell einordnen"


def build_neighbor_rows(
    model: Any,
    resolved_targets: dict[str, str],
    topn: int = DEFAULT_TOPN,
) -> list[dict[str, str | float | int]]:
    """Berechnet je Zielwort die ``topn`` nächsten Nachbarn als Tabellenzeilen."""
    rows: list[dict[str, str | float | int]] = []
    for requested_word, target in resolved_targets.items():
        for rank, (neighbor, similarity) in enumerate(model.wv.most_similar(target, topn=topn), start=1):
            rows.append(
                {
                    "angefragtes_zielwort": requested_word,
                    "verwendetes_zielwort": target,
                    "rang": rank,
                    "nachbar": neighbor,
                    "cosine_similarity": round(float(similarity), 4),
                    "einordnung": classify_neighbor(target, neighbor),
                    "kommentar": "",
                }
            )
    return rows


def write_csv(rows: list[dict[str, str | float | int]], path: Path) -> None:
    """Schreibt die Nachbarschaftstabelle als CSV."""
    if not rows:
        raise ValueError("Keine Zeilen zum Schreiben vorhanden.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str | float | int]], path: Path) -> None:
    """Schreibt die Nachbarschaftstabelle als Markdown (benötigt ``tabulate``)."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    path.write_text(dataframe.to_markdown(index=False), encoding="utf-8")


def evaluate_model(
    model_path: Path = MODEL_PATH,
    neighbors_csv: Path = NEIGHBORS_CSV,
    neighbors_md: Path = NEIGHBORS_MD,
    targets_json: Path = TARGETS_JSON,
    pca_figure: Path = PCA_FIGURE,
    target_figure_dir: Path = TARGET_FIGURE_DIR,
    topn: int = DEFAULT_TOPN,
) -> dict[str, str | int]:
    """Lädt das Modell, bestimmt Nachbarn und schreibt Tabellen + PCA-Grafiken."""
    from gensim.models import Word2Vec  # lazy: nur zur Auswertung benötigt

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden unter {model_path}. "
            f"Zuerst 'python -m src.word2vec.train' ausführen."
        )

    model = Word2Vec.load(str(model_path))
    resolved_targets = resolve_target_words(model)
    missing_targets = sorted(set(TARGET_WORDS) - set(resolved_targets))
    if not resolved_targets:
        raise ValueError("Kein konfiguriertes Zielwort (auch kein Ersatzwort) ist im Vokabular vorhanden.")

    rows = build_neighbor_rows(model, resolved_targets, topn=topn)
    write_csv(rows, neighbors_csv)
    write_markdown(rows, neighbors_md)
    plot_pca(model, rows, pca_figure)
    target_figures = plot_target_pcas(model, rows, target_figure_dir)

    target_info = {
        "resolved_targets": resolved_targets,
        "missing_targets": missing_targets,
        "target_figures": target_figures,
        "topn": topn,
    }
    targets_json.parent.mkdir(parents=True, exist_ok=True)
    targets_json.write_text(json.dumps(target_info, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "resolved_target_count": len(resolved_targets),
        "missing_target_count": len(missing_targets),
        "neighbors_csv": str(neighbors_csv),
        "neighbors_md": str(neighbors_md),
        "pca_figure": str(pca_figure),
        "target_figure_dir": str(target_figure_dir),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wertet Word2Vec-Nachbarschaften aus und erzeugt PCA-Grafiken.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Pfad zum trainierten Word2Vec-Modell.")
    parser.add_argument("--neighbors-csv", type=Path, default=NEIGHBORS_CSV)
    parser.add_argument("--neighbors-md", type=Path, default=NEIGHBORS_MD)
    parser.add_argument("--targets-json", type=Path, default=TARGETS_JSON)
    parser.add_argument("--pca-figure", type=Path, default=PCA_FIGURE)
    parser.add_argument("--target-figure-dir", type=Path, default=TARGET_FIGURE_DIR)
    parser.add_argument("--topn", type=int, default=DEFAULT_TOPN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        model_path=args.model,
        neighbors_csv=args.neighbors_csv,
        neighbors_md=args.neighbors_md,
        targets_json=args.targets_json,
        pca_figure=args.pca_figure,
        target_figure_dir=args.target_figure_dir,
        topn=args.topn,
    )


if __name__ == "__main__":
    main()
