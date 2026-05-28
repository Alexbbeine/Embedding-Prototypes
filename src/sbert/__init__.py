"""
src.sbert – Sentence-BERT Prototyp

Module:
    similarity   – Embeddings, Kosinusähnlichkeit, Kategorie-Zuordnung
    evaluation   – Auswertungsmetriken (Pearson, Spearman, MAE)
    diagnostics  – Manuell konstruierte deutsche Satzpaare
"""
from .similarity  import compute_embeddings, compute_cosine_similarities, assign_category
from .evaluation  import compute_metrics
from .diagnostics import DIAGNOSTIC_PAIRS

__all__ = [
    "compute_embeddings",
    "compute_cosine_similarities",
    "assign_category",
    "compute_metrics",
    "DIAGNOSTIC_PAIRS",
]
