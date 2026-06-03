"""
src.sbert – Sentence-BERT Prototyp

Module:
    similarity   – Embeddings, Kosinusähnlichkeit, Kategorie-Zuordnung
    evaluation   – Auswertungsmetriken (Pearson, Spearman, MAE)
    diagnostics  – Manuell konstruierte deutsche Satzpaare
    visualize    – Matplotlib-Grafiken (dieselben fünf wie das Notebook)

``visualize`` importiert ``matplotlib`` erst innerhalb der Funktionen, daher bleibt
``import src.sbert`` auch ohne installiertes matplotlib möglich.
"""
from .similarity  import compute_embeddings, compute_cosine_similarities, assign_category
from .evaluation  import compute_metrics
from .diagnostics import DIAGNOSTIC_PAIRS
from .visualize   import (
    plot_demo_heatmap,
    plot_gold_score_distribution,
    plot_scatter_gold_vs_predicted,
    plot_cosine_by_category,
    plot_diagnostic_results,
)

__all__ = [
    "compute_embeddings",
    "compute_cosine_similarities",
    "assign_category",
    "compute_metrics",
    "DIAGNOSTIC_PAIRS",
    "plot_demo_heatmap",
    "plot_gold_score_distribution",
    "plot_scatter_gold_vs_predicted",
    "plot_cosine_by_category",
    "plot_diagnostic_results",
]
