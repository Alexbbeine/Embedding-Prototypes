"""
config.py
=========
Zentrale Konfiguration des Word2Vec-Prototyps.

Liest ``configs/word2vec.yaml`` (Single Source of Truth) ein und leitet daraus
alle Parameter und Pfade ab. Die übrigen Module importieren ausschließlich von
hier, sodass Hyperparameter, Zielwörter und Ablageorte an einer Stelle gepflegt
werden – analog zu ``configs/sbert.yaml`` beim SBERT-Prototyp.

Pfade werden relativ zum Projektroot aufgelöst (zwei Ebenen über dieser Datei:
``src/word2vec/`` → Projektroot), damit Skripte unabhängig vom aktuellen
Arbeitsverzeichnis dieselben Ausgabeorte verwenden.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/word2vec/config.py  →  parents[2] == Projektroot
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "word2vec.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Lädt und gibt die YAML-Konfiguration zurück."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = load_config()

# ---------------------------------------------------------------------------
# Korpus
# ---------------------------------------------------------------------------
CORPUS_ID:   str = _config["corpus_id"]
CORPUS_URL:  str = _config["corpus_url"]
CORPUS_PAGE: str = _config["corpus_page"]

# ---------------------------------------------------------------------------
# Stichprobe / Training
# ---------------------------------------------------------------------------
# null im YAML → vollständigen Korpus verwenden (großer Default-Wert).
_sample_size = _config.get("sample_size")
DEFAULT_SAMPLE_SIZE: int = _sample_size if _sample_size is not None else 100_000
DEFAULT_SEED: int = _config.get("random_seed", 42)
DEFAULT_MIN_SENTENCE_TOKENS: int = _config.get("min_sentence_tokens", 2)
DEFAULT_TOPN: int = _config.get("topn", 5)

# Trainingsparameter; der Seed stammt aus random_seed (eine zentrale Quelle).
WORD2VEC_PARAMS: dict[str, int] = dict(_config["word2vec_params"])
WORD2VEC_PARAMS.setdefault("seed", DEFAULT_SEED)

# Zielwörter mit Ersatzwörtern (Fallbacks).
TARGET_WORDS: dict[str, list[str]] = {
    word: list(fallbacks) for word, fallbacks in _config["target_words"].items()
}

# ---------------------------------------------------------------------------
# Verzeichnisse (relativ zum Projektroot)
# ---------------------------------------------------------------------------
DATA_DIR           = PROJECT_ROOT / _config.get("data_dir", "data")
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR          = PROJECT_ROOT / _config.get("model_dir", "outputs/models")
OUTPUT_TABLES_DIR  = PROJECT_ROOT / _config.get("output_tables_dir", "outputs/tables")
OUTPUT_FIGURES_DIR = PROJECT_ROOT / _config.get("output_figures_dir", "outputs/figures")

# ---------------------------------------------------------------------------
# Konkrete Dateipfade
# ---------------------------------------------------------------------------
# Korpus-Rohdaten und tokenisierte Zwischenstände
ARCHIVE_PATH          = RAW_DATA_DIR / f"{CORPUS_ID}.tar.gz"
SENTENCES_JSONL       = PROCESSED_DATA_DIR / "sentences_100k.jsonl"
PREPARE_METADATA_PATH = PROCESSED_DATA_DIR / "prepare_metadata.json"

# Modell
MODEL_PATH = MODEL_DIR / "word2vec_deu_news_2010_100k.model"

# Ergebnisse (mit Präfix word2vec_, da outputs/ mit dem SBERT-Prototyp geteilt wird)
MODEL_METADATA_PATH = OUTPUT_TABLES_DIR / "word2vec_model_metadata.json"
NEIGHBORS_CSV       = OUTPUT_TABLES_DIR / "word2vec_neighbors.csv"
NEIGHBORS_MD        = OUTPUT_TABLES_DIR / "word2vec_neighbors.md"
TARGETS_JSON        = OUTPUT_TABLES_DIR / "word2vec_target_words.json"
PCA_FIGURE          = OUTPUT_FIGURES_DIR / "word2vec_pca_neighbors.png"
TARGET_FIGURE_DIR   = OUTPUT_FIGURES_DIR / "word2vec_targets"
