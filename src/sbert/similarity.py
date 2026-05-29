"""
similarity.py
=============
Sentence-Embeddings, paarweise Kosinusähnlichkeit und Kategorie-Zuordnung.

Kosinusähnlichkeit misst den Winkel zwischen zwei Vektoren im Raum:

    cos(θ) = (u · v) / (‖u‖ · ‖v‖)

Ein Wert von  1.0 bedeutet identische Richtung (maximale Ähnlichkeit),
             0.0 Orthogonalität (keine lineare Beziehung) und
            −1.0 entgegengesetzte Richtung.

Bei SBERT-Embeddings liegt der praktische Wertebereich nahezu immer in [0, 1].
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def compute_embeddings(
    model: SentenceTransformer,
    sentences: list[str],
    batch_size: int = 64,
    desc: str = "",
) -> np.ndarray:
    """
    Berechnet Sentence-Embeddings für eine Liste von Sätzen.

    Args:
        model:      Geladenes SentenceTransformer-Modell.
        sentences:  Liste der zu enkodierenden Sätze.
        batch_size: Anzahl Sätze pro Batch (beeinflusst Speicherbedarf).
        desc:       Optionale Beschreibung, die vor dem Fortschrittsbalken auf der Konsole ausgegeben wird.

    Returns:
        Embedding-Matrix der Form (N, D), dtype float32.
    """
    if desc:
        print(f"  Enkodiere {len(sentences)} Sätze ({desc}) ...")
    return model.encode(sentences, batch_size=batch_size, show_progress_bar=True)


# ---------------------------------------------------------------------------
# Kosinusähnlichkeit
# ---------------------------------------------------------------------------

def compute_cosine_similarities(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
) -> np.ndarray:
    """
    Berechnet die paarweise Kosinusähnlichkeit zwischen zwei Embedding-Matrizen.

    Die Embeddings werden vor der Berechnung explizit L2-normalisiert.
    Das stellt korrekte Ergebnisse sicher, unabhängig davon, ob das Modell die Vektoren bereits normalisiert ausgibt. 
    Bei bereits normierten Vektoren gilt: Skalarprodukt = Kosinusähnlichkeit.

    Args:
        embeddings1: Embedding-Matrix, Form (N, D).
        embeddings2: Embedding-Matrix, Form (N, D).

    Returns:
        1-D-Array mit N Ähnlichkeitswerten in [−1.0, 1.0].
    """
    norm1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    # Zeilenweises Skalarprodukt → ein Ähnlichkeitswert je Satzpaar
    return (norm1 * norm2).sum(axis=1)


# ---------------------------------------------------------------------------
# Kategorisierung
# ---------------------------------------------------------------------------

def assign_category(
    score: float,
    dissimilar_max: float,
    similar_min: float,
) -> str:
    """
    Ordnet einen Ähnlichkeitswert einer von drei lesbaren Kategorien zu.

    Die Schwellenwerte sind in ``configs/sbert.yaml`` unter ``thresholds`` konfigurierbar und werden hier als explizite Parameter übergeben, 
    damit die Funktion zustandslos und unabhängig von Konfigurationsdateien bleibt.

    Kategorien::

        score < dissimilar_max                  → 'unähnlich'
        dissimilar_max ≤ score < similar_min    → 'teilweise ähnlich'
        score ≥ similar_min                     → 'sehr ähnlich'

    Args:
        score:          Kosinusähnlichkeit oder normalisierter Gold-Score (0–1).
        dissimilar_max: Obere Grenze (exklusiv) für 'unähnlich'.
        similar_min:    Untere Grenze (inklusiv) für 'sehr ähnlich'.

    Returns:
        Kategorie-String: 'unähnlich', 'teilweise ähnlich' oder 'sehr ähnlich'.
    """
    if score < dissimilar_max:
        return "unähnlich"
    if score >= similar_min:
        return "sehr ähnlich"
    return "teilweise ähnlich"


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------

def compute_absolute_deviation(cosine_sim: float, gold_norm: float) -> float:
    """
    Berechnet die absolute Abweichung zwischen berechnetem und annotiertem Wert.

    Args:
        cosine_sim: Vom Modell berechnete Kosinusähnlichkeit.
        gold_norm:  Normalisierter Gold-Score aus dem Datensatz (0–1).

    Returns:
        |cosine_sim − gold_norm|, gerundet auf 4 Dezimalstellen.
    """
    return round(abs(cosine_sim - gold_norm), 4)
