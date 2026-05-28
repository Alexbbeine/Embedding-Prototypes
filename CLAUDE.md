# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Academic seminar project: "Vom Wort zum Vektor – Embeddings für Texte" (DHBW).  
Two self-contained Python prototypes — Word2Vec (Prototype 1, largely stubs) and Sentence-BERT (Prototype 2, fully implemented). No web server, no database, no API.

## Environment

Python virtual environment is in `.venv/`. Always use it explicitly — there is no activation script expected to be sourced.

```powershell
# Windows (PowerShell) — all commands assume project root as working directory
.venv\Scripts\python.exe <script>
.venv\Scripts\pip install <package>
```

All scripts **must be run from the project root**, because output paths (e.g. `outputs/tables/`) are relative to the working directory, not to `__file__`.

## Running the SBERT prototype

```powershell
# Main evaluation script (loads model + dataset, computes cosine similarity, saves results)
.venv\Scripts\python.exe src/sbert/evaluate_similarity.py

# With explicit config path
.venv\Scripts\python.exe src/sbert/evaluate_similarity.py --config configs/sbert.yaml

# As module (requires src/__init__.py and src/sbert/__init__.py, both present)
.venv\Scripts\python.exe -m src.sbert.evaluate_similarity
```

For the Jupyter notebook, `ipykernel` must be installed first:

```powershell
.venv\Scripts\pip install ipykernel
# Then open notebooks/02_sbert_similarity.ipynb in VS Code and select .venv as kernel
```

## Architecture

### SBERT prototype (fully implemented)

```
configs/sbert.yaml              ← single source of truth for all parameters
src/sbert/
    evaluate_similarity.py      ← main script; reads config, orchestrates pipeline
    similarity.py               ← compute_embeddings, compute_cosine_similarities,
                                   assign_category, compute_absolute_deviation
    evaluation.py               ← compute_metrics (Pearson, Spearman, MAE)
    diagnostics.py              ← DIAGNOSTIC_PAIRS (12 German edge-case pairs)
    __init__.py                 ← re-exports public API of all three modules
src/__init__.py
outputs/tables/                 ← sbert_similarity_results.csv, sbert_summary.json,
                                   sbert_diagnostic_results.csv  (created on first run)
outputs/figures/                ← PNG plots (created by the notebook only)
notebooks/02_sbert_similarity.ipynb  ← narrative walkthrough, same logic as the script
```

**Data flow in `evaluate_similarity.py`:**
1. Load config from YAML → load `SentenceTransformer` model
2. Load `mteb/stsb_multi_mt` (German, dev split, 1500 pairs) via `datasets`
3. `compute_embeddings()` for `sentence1` and `sentence2` separately
4. `compute_cosine_similarities()` → `assign_category()` using YAML thresholds
5. Normalize gold scores 0–5 → 0–1 (divide by 5)
6. `compute_metrics()` → Pearson-r, Spearman-r, MAE via `scipy` / `sklearn`
7. Save CSV + JSON to `output_dir`
8. Repeat steps 3–4 for `DIAGNOSTIC_PAIRS` from `diagnostics.py`, save separate CSV

**Import note:** `evaluate_similarity.py` inserts `Path(__file__).parent` into `sys.path` so that `similarity`, `evaluation`, and `diagnostics` are importable regardless of call style (direct script or `-m` module).

**Windows encoding:** `sys.stdout.reconfigure(encoding="utf-8")` is called at module load to handle German Umlauts and box-drawing characters on cp1252 consoles.

### Key config parameters (`configs/sbert.yaml`)

| Key | Effect |
|---|---|
| `model_name` | HuggingFace model ID for `SentenceTransformer` |
| `dataset_name` | Must be `"mteb/stsb_multi_mt"` — the old `"stsb_multi_mt"` (no namespace) is broken in current `huggingface_hub` |
| `sample_per_category` | `null` = use all 1500 pairs; integer = stratified sample N per category |
| `thresholds.dissimilar_max` / `similar_min` | Applied to `gold_score_normalized` (0–1 scale) for the three-way categorization |

### Word2Vec prototype (Prototype 1)

`src/word2vec/` and `configs/word2vec.yaml` are present but **not yet implemented** (empty stubs). Do not modify these unless explicitly asked.

### Notebook vs. script

`notebooks/02_sbert_similarity.ipynb` and `src/sbert/evaluate_similarity.py` implement the same pipeline. The notebook adds matplotlib visualizations and LaTeX-rendered explanations for the paper. Changes to the pipeline logic should be kept in sync between both.

## Dependencies

`requirements.txt` lists the minimal set:
`sentence-transformers`, `datasets`, `pandas`, `numpy`, `pyyaml`, `scikit-learn`, `scipy`.  
`matplotlib` is used in the notebook and is already available as a transitive dependency of `sentence-transformers`.  
`ipykernel` is **not** in `requirements.txt` — install separately only when running notebooks.

## Docker

Each prototype has its own Dockerfile under `docker/<prototype>/`:

```
docker/
    sbert/
        Dockerfile      ← Prototyp 2: vollständig lauffähig
    word2vec/
        Dockerfile      ← Prototyp 1: Platzhalter (noch nicht implementiert)
docker-compose.yaml     ← verwaltet beide Services
.dockerignore           ← schließt .venv/, sbert-prototyp/, outputs/ aus
```

**SBERT-Container starten (Docker Desktop oder CLI):**

```powershell
# Evaluation ausführen (Ergebnisse erscheinen in ./outputs/tables/)
docker compose run sbert

# Interaktive Shell im Container
docker compose run sbert bash

# Image neu bauen (nach Code-Änderungen)
docker compose build sbert
```

Der Build-Kontext ist immer der Projektroot (`.`). Das HuggingFace-Modell wird beim ersten Start heruntergeladen und im benannten Volume `hf-cache` gespeichert – kein erneuter Download bei Neustarts.

Der Word2Vec-Service ist in `docker-compose.yaml` auskommentiert und wird aktiviert, sobald `src/word2vec/` implementiert ist.
