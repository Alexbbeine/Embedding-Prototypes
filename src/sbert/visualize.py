"""
visualize.py
============
Matplotlib-Visualisierungen des SBERT-Prototyps.

Erzeugt dieselben fünf Grafiken wie das Notebook ``02_sbert_similarity.ipynb``:

    1. demo_similarity_heatmap.png        – 3-Satz-Demo-Ähnlichkeitsmatrix
    2. gold_score_distribution.png        – Verteilung der Gold-Scores (mit Schwellen)
    3. sbert_scatter_gold_vs_predicted.png– Gold-Score vs. Kosinusähnlichkeit
    4. sbert_cosine_by_category.png       – Kosinusähnlichkeit je Kategorie (Histogramme)
    5. sbert_diagnostic_results.png       – diagnostische Satzpaare (Treffer/Fehler)

``matplotlib`` wird erst innerhalb der Funktionen importiert, damit das Paket auch
ohne installiertes ``matplotlib`` importierbar bleibt und das nicht-interaktive
``Agg``-Backend zuverlässig gesetzt werden kann (analog zu ``src/word2vec/visualize.py``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# Drei Sätze für die einleitende Demo-Heatmap (identisch zum Notebook).
DEMO_SENTENCES: list[str] = [
    "Das Wetter ist heute wunderbar.",
    "Es ist herrlich sonnig draußen.",
    "Er fuhr zum Stadion.",
]

# Farbschema der drei Ähnlichkeitskategorien (identisch zum Notebook).
CATEGORY_COLORS: dict[str, str] = {
    "unähnlich":         "#e74c3c",
    "teilweise ähnlich": "#f39c12",
    "sehr ähnlich":      "#27ae60",
}

CATEGORY_ORDER: list[str] = ["unähnlich", "teilweise ähnlich", "sehr ähnlich"]


def _pyplot(figure_path: Path):
    """Bereitet matplotlib (Agg-Backend, lokaler Cache) vor und gibt ``pyplot`` zurück."""
    cache_dir = figure_path.parent.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    return plt


# ---------------------------------------------------------------------------
# 1 · Demo-Ähnlichkeitsmatrix
# ---------------------------------------------------------------------------

def plot_demo_heatmap(
    model: Any,
    figure_path: Path,
    demo_sentences: list[str] = DEMO_SENTENCES,
) -> None:
    """Zeichnet die Kosinusähnlichkeitsmatrix einiger Demo-Sätze als Heatmap."""
    plt = _pyplot(figure_path)

    embeddings = np.asarray(model.encode(demo_sentences))
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = normed @ normed.T

    n = len(demo_sentences)
    labels = [f"S{i + 1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Kosinusähnlichkeit")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            color = "white" if sim_matrix[i, j] > 0.7 else "black"
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    ax.set_title(f"Demo: Ähnlichkeitsmatrix ({n} Sätze)", fontsize=10)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2 · Verteilung der Gold-Scores
# ---------------------------------------------------------------------------

def plot_gold_score_distribution(
    gold_scores: list[float] | np.ndarray,
    dissimilar_max: float,
    similar_min: float,
    figure_path: Path,
) -> None:
    """Histogramm der Gold-Scores (Skala 0–5) mit eingezeichneten Schwellenwerten."""
    plt = _pyplot(figure_path)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(gold_scores, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(dissimilar_max * 5, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Schwelle unähnlich: {dissimilar_max * 5:.1f}")
    ax.axvline(similar_min * 5, color="#27ae60", linestyle="--", linewidth=1.5,
               label=f"Schwelle sehr ähnlich: {similar_min * 5:.1f}")
    ax.set_xlabel("Gold-Score (0–5)", fontsize=11)
    ax.set_ylabel("Anzahl Satzpaare", fontsize=11)
    ax.set_title("Verteilung der Gold-Scores im STS-B Deutsch (dev)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3 · Streudiagramm Gold-Score vs. Vorhersage
# ---------------------------------------------------------------------------

def plot_scatter_gold_vs_predicted(
    results_df: Any,
    pearson_r: float,
    spearman_r: float,
    mae: float,
    figure_path: Path,
) -> None:
    """Streudiagramm Gold-Score gegen vorhergesagte Kosinusähnlichkeit, nach Kategorie eingefärbt."""
    plt = _pyplot(figure_path)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cat, color in CATEGORY_COLORS.items():
        mask = results_df["category"] == cat
        ax.scatter(results_df.loc[mask, "gold_score_normalized"],
                   results_df.loc[mask, "predicted_cosine_similarity"],
                   s=8, alpha=0.4, color=color, label=f"{cat} (n={int(mask.sum())})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Ideale Vorhersage (y = x)")
    ax.text(0.04, 0.93,
            f"Pearson r  = {pearson_r:.3f}\nSpearman r = {spearman_r:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    ax.set_xlabel("Gold-Score (normalisiert, 0–1)", fontsize=11)
    ax.set_ylabel("Vorhergesagte Kosinusähnlichkeit", fontsize=11)
    ax.set_title("SBERT: Gold-Score vs. Kosinusähnlichkeit\n(STS-B Deutsch, dev-Split)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4 · Kosinusähnlichkeit je Kategorie
# ---------------------------------------------------------------------------

def plot_cosine_by_category(results_df: Any, figure_path: Path) -> None:
    """Drei nebeneinanderliegende Histogramme der Kosinusähnlichkeit je Kategorie."""
    plt = _pyplot(figure_path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, cat in zip(axes, CATEGORY_ORDER):
        vals = results_df.loc[results_df["category"] == cat, "predicted_cosine_similarity"]
        ax.hist(vals, bins=20, color=CATEGORY_COLORS[cat], edgecolor="white", linewidth=0.4)
        if len(vals) > 0:
            ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.2,
                       label=f"MW: {vals.mean():.2f}")
            ax.legend(fontsize=8)
        ax.set_title(f"{cat}  (n = {len(vals)})", fontsize=10)
        ax.set_xlabel("Kosinusähnlichkeit", fontsize=9)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Anzahl Satzpaare", fontsize=9)
    fig.suptitle("Verteilung der Kosinusähnlichkeit nach Kategorie", fontsize=11)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5 · Diagnostische Satzpaare
# ---------------------------------------------------------------------------

def plot_diagnostic_results(
    diag_df: Any,
    dissimilar_max: float,
    similar_min: float,
    figure_path: Path,
) -> None:
    """Balkendiagramm der diagnostischen Satzpaare (grün = korrekt, rot = falsch)."""
    plt = _pyplot(figure_path)
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(diag_df))
    bar_colors = ["#27ae60" if m else "#e74c3c" for m in diag_df["match"]]
    ax.barh(y_pos, diag_df["predicted_cosine_similarity"],
            color=bar_colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax.axvline(dissimilar_max, color="#777", linestyle="--", linewidth=1.3)
    ax.axvline(similar_min, color="navy", linestyle="--", linewidth=1.3)
    for i, val in enumerate(diag_df["predicted_cosine_similarity"]):
        ax.text(min(val + 0.012, 1.01), i, f"{val:.3f}", va="center", fontsize=8)
    y_labels = [f"{row.id}  [{row.linguistic_category}]" for row in diag_df.itertuples()]
    ax.set_yticks(y_pos); ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.set_xlabel("Kosinusähnlichkeit (vorhergesagt)", fontsize=10)
    ax.set_title(
        "Diagnostische Satzpaare – SBERT-Kosinusähnlichkeit\n"
        "(grün = korrekt  |  rot = falsch kategorisiert)", fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.legend(handles=[
        mpatches.Patch(color="#27ae60", label="Korrekt"),
        mpatches.Patch(color="#e74c3c", label="Falsch"),
        plt.Line2D([0], [0], color="#777", linestyle="--", label=f"< {dissimilar_max} = unähnlich"),
        plt.Line2D([0], [0], color="navy", linestyle="--", label=f">= {similar_min} = sehr ähnlich"),
    ], fontsize=8.5, loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
