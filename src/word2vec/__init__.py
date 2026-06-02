"""
src.word2vec – Word2Vec Prototyp (Prototyp 1)

Trainiert statische deutsche Wortvektoren auf dem Leipzig-Korpus
(deu_news_2010_100K) und analysiert die nächsten Wortnachbarschaften.

Module:
    config     – lädt configs/word2vec.yaml und leitet Parameter + Pfade ab
    corpus     – Leipzig-Parsing, Bereinigung, Tokenisierung (reine Funktionen)
    download   – Korpusarchiv herunterladen          (CLI: python -m src.word2vec.download)
    prepare    – Korpus aufbereiten und tokenisieren  (CLI: python -m src.word2vec.prepare)
    train      – Gensim-Word2Vec-Modell trainieren    (CLI: python -m src.word2vec.train)
    evaluate   – nächste Nachbarn je Zielwort         (CLI: python -m src.word2vec.evaluate)
    visualize  – PCA-Visualisierung der Nachbarschaften
    pipeline   – kompletter Ablauf                    (CLI: python -m src.word2vec.pipeline)

Re-exportiert wird die wiederverwendbare Bibliotheks-Oberfläche (config, corpus,
visualize). Die ausführbaren Schritt-Module werden bewusst NICHT hier importiert,
damit ``python -m src.word2vec.<schritt>`` ohne runpy-Warnung läuft; ihre
Funktionen sind direkt über das jeweilige Submodul erreichbar, z. B.
``from src.word2vec.train import train_model``.
"""
import sys

# UTF-8-Ausgabe auf Windows erzwingen (Umlaute, JSON mit ensure_ascii=False).
# Liegt im Paket-Init, damit jeder Einstieg via 'python -m src.word2vec.<modul>'
# davon profitiert (das Paket-Init läuft vor dem Submodul).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .config import TARGET_WORDS, WORD2VEC_PARAMS
from .corpus import (
    clean_text,
    corpus_stats,
    iter_sentences_from_archive,
    load_tokenized_sentences,
    read_leipzig_sentences,
    sample_sentences,
    save_tokenized_sentences,
    tokenize,
    tokenize_sentences,
)
from .visualize import plot_pca, plot_target_pcas, slugify_word

__version__ = "0.1.0"

__all__ = [
    "WORD2VEC_PARAMS",
    "TARGET_WORDS",
    "read_leipzig_sentences",
    "iter_sentences_from_archive",
    "clean_text",
    "tokenize",
    "sample_sentences",
    "tokenize_sentences",
    "save_tokenized_sentences",
    "load_tokenized_sentences",
    "corpus_stats",
    "plot_pca",
    "plot_target_pcas",
    "slugify_word",
    "__version__",
]
