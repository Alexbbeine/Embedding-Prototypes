#!/usr/bin/env python3
"""
evaluate_similarity.py
======================
Sentence-BERT Prototyp – Einstiegspunkt und Pipeline-Steuerung.

Führt zwei Aufgaben aus:
    1. Datensatz-Evaluation auf dem deutschen STS-B-Benchmark
       (``mteb/stsb_multi_mt``, Sprache: ``de``, Split: ``dev``, 1500 Paare).
    2. Qualitative Analyse auf eigenen diagnostischen Grenzfall-Paaren.

Die eigentliche Berechnungslogik ist auf separate Module aufgeteilt:

    similarity.py   – Embeddings, Kosinusähnlichkeit, Kategorisierung
    evaluation.py   – Auswertungsmetriken (Pearson, Spearman, MAE)
    diagnostics.py  – Manuell konstruierte deutsche Satzpaare

Ausführung (vom Projektverzeichnis):
    python src/sbert/evaluate_similarity.py
    python src/sbert/evaluate_similarity.py --config configs/sbert.yaml
    python -m src.sbert.evaluate_similarity
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# UTF-8-Ausgabe auf Windows erzwingen (Box-Drawing-Zeichen, Umlaute)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Sicherstellen, dass die Submodule importierbar sind,
# unabhängig davon, ob das Skript direkt oder als Modul aufgerufen wird.
sys.path.insert(0, str(Path(__file__).parent))
from similarity   import compute_embeddings, compute_cosine_similarities, assign_category
from evaluation   import compute_metrics
from diagnostics  import DIAGNOSTIC_PAIRS


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Lädt und gibt die YAML-Konfiguration zurück."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Datensatz
# ---------------------------------------------------------------------------

def load_sts_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
) -> tuple[list[str], list[str], list[float]]:
    """
    Lädt den STS-Datensatz von HuggingFace und gibt Satzpaare mit
    Gold-Scores zurück.

    Args:
        dataset_name:   HuggingFace-Datensatz-ID (z. B. ``'mteb/stsb_multi_mt'``).
        dataset_config: Sprachkonfiguration (z. B. ``'de'``).
        split:          Datensatz-Split (``'dev'``, ``'test'`` oder ``'train'``).

    Returns:
        Tupel (sentences1, sentences2, gold_scores), wobei gold_scores
        auf der Originalskala 0–5 liegen.
    """
    print(f"Lade Datensatz: {dataset_name} [{dataset_config}] split='{split}' ...")
    dataset = load_dataset(dataset_name, name=dataset_config, split=split)
    sentences1: list[str]   = list(dataset["sentence1"])
    sentences2: list[str]   = list(dataset["sentence2"])
    gold_scores: list[float] = [float(s) for s in dataset["similarity_score"]]
    print(f"  {len(sentences1)} Satzpaare geladen.")
    return sentences1, sentences2, gold_scores


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_by_category(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Stratifiziertes Sampling: Gibt genau ``n`` Beispiele pro Kategorie zurück.

    Args:
        df:   DataFrame mit einer Spalte ``'category'``.
        n:    Maximale Anzahl Beispiele pro Kategorie.
        seed: Zufalls-Seed für Reproduzierbarkeit.

    Returns:
        Neues DataFrame mit höchstens ``n × |Kategorien|`` Zeilen.
    """
    return (
        df.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Diagnostische Satzpaare
# ---------------------------------------------------------------------------

def evaluate_diagnostic_pairs(
    model: SentenceTransformer,
    pairs: list[dict],
    dissimilar_max: float,
    similar_min: float,
) -> pd.DataFrame:
    """
    Wertet die diagnostischen Satzpaare aus und vergleicht die vom Modell
    vorhergesagte Kategorie mit dem manuell annotierten Erwartungswert.

    Args:
        model:          Geladenes SentenceTransformer-Modell.
        pairs:          Liste von :class:`~diagnostics.DiagnosticPair`-Dicts.
        dissimilar_max: Schwelle für 'unähnlich' (aus Konfiguration).
        similar_min:    Schwelle für 'sehr ähnlich' (aus Konfiguration).

    Returns:
        DataFrame mit einer Zeile je Paar und den Spalten:
        ``id``, ``linguistic_category``, ``sentence1``, ``sentence2``,
        ``expected_category``, ``predicted_cosine_similarity``,
        ``predicted_category``, ``match``.
    """
    sentences1 = [p["sentence1"] for p in pairs]
    sentences2 = [p["sentence2"] for p in pairs]

    emb1 = compute_embeddings(model, sentences1, batch_size=32, desc="Diagnose Satz1")
    emb2 = compute_embeddings(model, sentences2, batch_size=32, desc="Diagnose Satz2")
    cosine_sims = compute_cosine_similarities(emb1, emb2)

    rows = []
    for i, p in enumerate(pairs):
        sim = float(cosine_sims[i])
        predicted_cat = assign_category(sim, dissimilar_max, similar_min)
        rows.append(
            {
                "id":                          p["id"],
                "linguistic_category":         p["category"],
                "sentence1":                   p["sentence1"],
                "sentence2":                   p["sentence2"],
                "expected_category":           p["expected"],
                "predicted_cosine_similarity": round(sim, 4),
                "predicted_category":          predicted_cat,
                "match":                       predicted_cat == p["expected"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SBERT-Evaluation semantischer Satzähnlichkeit (Deutsch)."
    )
    parser.add_argument(
        "--config",
        default="configs/sbert.yaml",
        help="Pfad zur YAML-Konfigurationsdatei (Standard: configs/sbert.yaml)",
    )
    args = parser.parse_args()

    # ── Konfiguration laden ──────────────────────────────────────────────────
    config = load_config(args.config)
    model_name:          str       = config["model_name"]
    dataset_name:        str       = config["dataset_name"]
    dataset_config:      str       = config["dataset_config"]
    split:               str       = config["split"]
    sample_per_category: int | None = config.get("sample_per_category")
    random_seed:         int       = config.get("random_seed", 42)
    output_dir                     = Path(config.get("output_dir", "outputs/tables"))
    thresholds: dict               = config.get("thresholds", {})
    dissimilar_max: float          = thresholds.get("dissimilar_max", 0.3)
    similar_min:    float          = thresholds.get("similar_min",    0.7)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Modell laden ─────────────────────────────────────────────────────────
    print(f"\nLade Modell: {model_name}")
    model = SentenceTransformer(model_name)

    # ── Datensatz-Evaluation ─────────────────────────────────────────────────
    print("\n=== Datensatz-Evaluation ===")
    sentences1, sentences2, gold_scores = load_sts_dataset(
        dataset_name, dataset_config, split
    )

    print("\nBerechne Embeddings ...")
    emb1 = compute_embeddings(model, sentences1, desc="sentence1")
    emb2 = compute_embeddings(model, sentences2, desc="sentence2")

    cosine_sims    = compute_cosine_similarities(emb1, emb2)
    gold_arr       = np.array(gold_scores)
    gold_normalized = gold_arr / 5.0          # Skala 0–5 → 0–1

    categories = [assign_category(s, dissimilar_max, similar_min) for s in gold_normalized]
    abs_errors  = np.abs(gold_normalized - cosine_sims)

    # Ergebnistabelle zusammenstellen
    results_df = pd.DataFrame(
        {
            "id":                          range(len(sentences1)),
            "sentence1":                   sentences1,
            "sentence2":                   sentences2,
            "gold_score":                  gold_arr.round(2),
            "gold_score_normalized":       gold_normalized.round(4),
            "predicted_cosine_similarity": cosine_sims.round(4),
            "category":                    categories,
            "absolute_error":              abs_errors.round(4),
        }
    )

    # Optionales stratifiziertes Sampling
    if sample_per_category is not None:
        print(f"\nSample {sample_per_category} Beispiele pro Kategorie ...")
        results_df = sample_by_category(results_df, sample_per_category, random_seed)

    # Metriken berechnen
    metrics = compute_metrics(
        predicted=results_df["predicted_cosine_similarity"].to_numpy(),
        gold=results_df["gold_score_normalized"].to_numpy(),
    )
    metrics["model_name"]            = model_name
    metrics["dataset"]               = f"{dataset_name}/{dataset_config}/{split}"
    metrics["thresholds"]            = thresholds
    metrics["category_distribution"] = results_df["category"].value_counts().to_dict()

    # Ergebnisse speichern
    results_path = output_dir / "sbert_similarity_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")

    summary_path = output_dir / "sbert_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Konsolenausgabe
    print("\n─── Evaluierungsergebnisse ───────────────────────────────────")
    print(f"  Modell:            {model_name}")
    print(f"  Datensatz:         {dataset_name} [{dataset_config}] / {split}")
    print(f"  Anzahl Satzpaare:  {metrics['n_samples']}")
    print(f"  Pearson-r:         {metrics['pearson_r']:.4f}  (p = {metrics['pearson_p']:.2e})")
    print(f"  Spearman-r:        {metrics['spearman_r']:.4f}  (p = {metrics['spearman_p']:.2e})")
    print(f"  MAE:               {metrics['mae']:.4f}")
    print(f"  Kategorienverteilung: {metrics['category_distribution']}")
    print(f"\n  Ergebnisse     → {results_path}")
    print(f"  Zusammenfassung → {summary_path}")

    # ── Diagnostische Satzpaare ───────────────────────────────────────────────
    print("\n=== Diagnostische Satzpaare ===")
    diag_df = evaluate_diagnostic_pairs(
        model, DIAGNOSTIC_PAIRS, dissimilar_max, similar_min
    )

    diag_path = output_dir / "sbert_diagnostic_results.csv"
    diag_df.to_csv(diag_path, index=False, encoding="utf-8")

    accuracy = float(diag_df["match"].mean())
    correct  = int(diag_df["match"].sum())
    total    = len(diag_df)
    print(f"\n  Trefferquote: {accuracy:.0%}  ({correct}/{total} korrekt)")
    print(f"  Diagnose-Ergebnisse → {diag_path}")

    print("\n  Detaillierte Diagnose:")
    print(f"  {'':2} {'Kategorie':26} {'cos':>5}  {'Vorhersage':20}  Erwartet")
    print("  " + "─" * 75)
    for _, row in diag_df.iterrows():
        mark = "✓" if row["match"] else "✗"
        print(
            f"  {mark} {row['linguistic_category']:26}"
            f"  {row['predicted_cosine_similarity']:5.3f}"
            f"  {row['predicted_category']:20}"
            f"  {row['expected_category']}"
        )

    # ── Abschließende Zusammenfassung der gespeicherten Dateien ──────────────
    print("\n" + "═" * 65)
    print("  Gespeicherte Ergebnisse")
    print("═" * 65)
    print(f"  {results_path}")
    print(f"  {summary_path}")
    print(f"  {diag_path}")
    print("═" * 65)


if __name__ == "__main__":
    main()
