"""
visualize.py
============
PCA-Visualisierung der Word2Vec-Nachbarschaften.

Reduziert Zielwörter und ihre nächsten Nachbarn mit PCA auf zwei Dimensionen –
einmal global über alle Zielwörter und einmal je Zielwort. Die Grafiken dienen
ausschließlich der explorativen Veranschaulichung.

``matplotlib`` und ``scikit-learn`` werden erst innerhalb der Funktionen
importiert, damit das Paket ohne diese Pakete importierbar bleibt und das
nicht-interaktive ``Agg``-Backend zuverlässig gesetzt werden kann.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def slugify_word(word: str) -> str:
    """Erzeugt einen dateinamentauglichen Bezeichner aus einem Wort."""
    slug = re.sub(r"[^a-z0-9äöüß]+", "_", word.lower()).strip("_")
    return slug or "zielwort"


def plot_pca(
    model: Any,
    rows: list[dict[str, str | float | int]],
    figure_path: Path,
    title: str = "PCA-Visualisierung ausgewählter Word2Vec-Nachbarschaften",
) -> None:
    """Zeichnet eine PCA-Grafik für die in ``rows`` enthaltenen Wörter."""
    cache_dir = figure_path.parent.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    words: list[str] = []
    for row in rows:
        for column in ("verwendetes_zielwort", "nachbar"):
            word = str(row[column])
            if word not in words and word in model.wv:
                words.append(word)

    if len(words) < 2:
        raise ValueError("Für die PCA-Visualisierung werden mindestens zwei Wörter benötigt.")

    vectors = [model.wv[word] for word in words]
    coordinates = PCA(n_components=2, random_state=42).fit_transform(vectors)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=34, color="#276fbf")
    for word, (x_coordinate, y_coordinate) in zip(words, coordinates):
        plt.annotate(word, (x_coordinate, y_coordinate), fontsize=9, alpha=0.86)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def plot_target_pcas(
    model: Any,
    rows: list[dict[str, str | float | int]],
    target_figure_dir: Path,
) -> dict[str, str]:
    """Erzeugt je Zielwort eine eigene PCA-Grafik (Zielwort + direkte Nachbarn)."""
    target_figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, str] = {}
    target_words = sorted({str(row["angefragtes_zielwort"]) for row in rows})

    for target_word in target_words:
        target_rows = [row for row in rows if row["angefragtes_zielwort"] == target_word]
        used_target = str(target_rows[0]["verwendetes_zielwort"])
        figure_path = target_figure_dir / f"pca_{slugify_word(target_word)}.png"
        plot_pca(
            model,
            target_rows,
            figure_path,
            title=f"PCA-Nachbarschaft für '{used_target}'",
        )
        figure_paths[target_word] = str(figure_path)

    return figure_paths
