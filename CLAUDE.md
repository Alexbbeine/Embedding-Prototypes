# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Academic seminar project: "Vom Wort zum Vektor – Embeddings für Texte" (DHBW).  
Two self-contained Python prototypes — Word2Vec (Prototype 1) and Sentence-BERT (Prototype 2), both fully implemented. No web server, no database, no API.

## Environment

Python virtual environment is in `.venv/`. Always use it explicitly — there is no activation script expected to be sourced.

```powershell
# Windows (PowerShell) — all commands assume project root as working directory
.venv\Scripts\python.exe <script>
.venv\Scripts\pip install <package>
```

Both prototypes anchor their output paths to the project root (via `Path(__file__).resolve().parents[2]`), so results land in the repo regardless of the current working directory. Running from the project root is still recommended for consistency (and the relative `--config` default resolves against the root).

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

## Running the Word2Vec prototype

The Word2Vec prototype is a multi-step CLI (download → prepare → train → evaluate). Run it as a module from the project root:

```powershell
# Full workflow
.venv\Scripts\python.exe -m src.word2vec.pipeline

# Skip download if the archive already exists in data/raw/
.venv\Scripts\python.exe -m src.word2vec.pipeline --skip-download

# Individual steps
.venv\Scripts\python.exe -m src.word2vec.download
.venv\Scripts\python.exe -m src.word2vec.prepare    # optional: --sample-size 20000
.venv\Scripts\python.exe -m src.word2vec.train
.venv\Scripts\python.exe -m src.word2vec.evaluate
```

**`gensim`:** required for `train`/`evaluate`. `gensim 4.4.0` ships a Python 3.13 wheel and is verified working in this `.venv` (Python 3.13 + numpy 2.4). If a future environment cannot install `gensim`, the Docker container (`docker compose run word2vec`, Python 3.12 base) is a fallback. The corpus modules and unit tests run without `gensim` (it is imported lazily).

## Running the tests

```powershell
.venv\Scripts\pip install pytest        # if not already installed
.venv\Scripts\python.exe -m pytest      # run from project root
```

`tests/conftest.py` puts the project root on `sys.path` so tests import `src.word2vec...`. The Word2Vec smoke test is skipped automatically when `gensim` (or another ML dependency) is missing; the corpus/target-word unit tests always run.

## Architecture

### SBERT prototype (fully implemented)

```
configs/sbert.yaml              ← single source of truth for all parameters
src/sbert/
    evaluate_similarity.py      ← main script; reads config, orchestrates pipeline
    similarity.py               ← compute_embeddings, compute_cosine_similarities,
                                   assign_category
    evaluation.py               ← compute_metrics (Pearson, Spearman, MAE)
    diagnostics.py              ← DIAGNOSTIC_PAIRS (12 German edge-case pairs)
    visualize.py                ← the five notebook figures (lazy matplotlib, Agg backend)
    __init__.py                 ← re-exports public API of all four modules
src/__init__.py
outputs/tables/                 ← sbert_similarity_results.csv, sbert_summary.json,
                                   sbert_diagnostic_results.csv  (created on first run)
outputs/figures/                ← demo_similarity_heatmap.png, gold_score_distribution.png,
                                   sbert_scatter_gold_vs_predicted.png, sbert_cosine_by_category.png,
                                   sbert_diagnostic_results.png  (created by script AND notebook)
notebooks/02_sbert_similarity.ipynb  ← narrative walkthrough, same logic as the script
docs/prototyp_sbert.md          ← method write-up for the paper
```

**Data flow in `evaluate_similarity.py`:**
1. Load config from YAML → load `SentenceTransformer` model
2. Load `mteb/stsb_multi_mt` (German, dev split, 1500 pairs) via `datasets`
3. `compute_embeddings()` for `sentence1` and `sentence2` separately
4. `compute_cosine_similarities()` → `assign_category()` using YAML thresholds
5. Normalize gold scores 0–5 → 0–1 (divide by 5)
6. `compute_metrics()` → Pearson-r, Spearman-r, MAE via `scipy` / `sklearn`
7. Save CSV + JSON to `output_dir`; render the four dataset figures via `visualize.py` to `figures_dir`
8. Repeat steps 3–4 for `DIAGNOSTIC_PAIRS` from `diagnostics.py`, save separate CSV + diagnostic figure

**Import note:** `evaluate_similarity.py` inserts `Path(__file__).parent` into `sys.path` so that `similarity`, `evaluation`, `diagnostics`, and `visualize` are importable regardless of call style (direct script or `-m` module). `visualize.py` imports `matplotlib` lazily (forces the `Agg` backend), so `import src.sbert` works without matplotlib.

**Windows encoding:** `sys.stdout.reconfigure(encoding="utf-8")` is called at module load to handle German Umlauts and box-drawing characters on cp1252 consoles.

### Key config parameters (`configs/sbert.yaml`)

| Key | Effect |
|---|---|
| `model_name` | HuggingFace model ID for `SentenceTransformer` |
| `dataset_name` | Must be `"mteb/stsb_multi_mt"` — the old `"stsb_multi_mt"` (no namespace) is broken in current `huggingface_hub` |
| `sample_per_category` | `null` = use all 1500 pairs; integer = stratified sample N per category |
| `thresholds.dissimilar_max` / `similar_min` | Applied to `gold_score_normalized` (0–1 scale) for the three-way categorization |
| `output_dir` / `figures_dir` | Output dirs for tables (CSV/JSON) and figures (PNG); relative values are anchored to the project root |

### Word2Vec prototype (fully implemented)

Trains static German word vectors with `gensim` on the Leipzig corpus (`deu_news_2010_100K`) and analyses nearest-neighbour relations for selected target words.

```
configs/word2vec.yaml           ← single source of truth (corpus, hyperparameters,
                                   target words, seed, sample size, topn, paths)
src/word2vec/
    config.py                   ← loads YAML, derives all parameters + paths
    corpus.py                   ← Leipzig parsing, cleaning, tokenization (pure fns)
    download.py                 ← download_corpus (Leipzig archive)
    prepare.py                  ← prepare_corpus (sample, tokenize, JSONL + metadata)
    train.py                    ← train_model (gensim Word2Vec; lazy gensim import)
    evaluate.py                 ← resolve_target_words, build_neighbor_rows,
                                   evaluate_model (lazy gensim import)
    visualize.py                ← plot_pca, plot_target_pcas (lazy matplotlib/sklearn)
    pipeline.py                 ← orchestrates download → prepare → train → evaluate
    __init__.py                 ← re-exports public API; sets UTF-8 stdout
    Dockerfile                  ← Prototyp 1 (Python 3.12 base, gensim wheels)
data/                           ← raw archive + processed JSONL (gitignored)
outputs/models/                 ← trained model (gitignored)
outputs/tables/                 ← word2vec_neighbors.csv/.md, word2vec_target_words.json,
                                   word2vec_model_metadata.json
outputs/figures/                ← word2vec_pca_neighbors.png, word2vec_targets/pca_*.png
notebooks/01_word2vec_exploration.ipynb  ← narrative walkthrough of the same workflow
tests/                          ← test_corpus, test_evaluate, test_smoke (+ conftest)
docs/prototyp_word2vec.md       ← method write-up for the paper
```

**Data flow (`pipeline.py`):**
1. `download_corpus()` → Leipzig `.tar.gz` into `data/raw/`
2. `prepare_corpus()` → read sentences, optional sample, clean + tokenize, write `data/processed/sentences_100k.jsonl` + metadata
3. `train_model()` → gensim `Word2Vec` on the tokenized sentences → `outputs/models/...model` + metadata JSON
4. `evaluate_model()` → for each target word, `topn` nearest neighbours by cosine similarity → CSV/Markdown; PCA plots (global + per target); `word2vec_target_words.json`

**Import / path notes:** modules use package-relative imports and are run as `python -m src.word2vec.<module>`. `config.py` anchors all paths to the project root via `Path(__file__).resolve().parents[2]`, so outputs land in the repo regardless of CWD (the SBERT prototype anchors the same way) — but still run from the root for consistency. `train.py` and `evaluate.py` import `gensim` lazily (inside functions) and `visualize.py` imports matplotlib/sklearn lazily, so `import src.word2vec` works without those heavy/optional dependencies.

**Key config parameters (`configs/word2vec.yaml`):**

| Key | Effect |
|---|---|
| `sample_size` | `null` = full 100K corpus; integer = smaller random subset (test runs) |
| `random_seed` | Seed for sampling and (injected into) `word2vec_params.seed` |
| `word2vec_params` | gensim hyperparameters (`vector_size`, `window`, `min_count`, `sg`, `epochs`, `workers`) |
| `topn` | Number of nearest neighbours reported per target word |
| `target_words` | Target word → fallback list (first in-vocabulary candidate is used) |

### Notebook vs. script

`notebooks/02_sbert_similarity.ipynb` and `src/sbert/evaluate_similarity.py` implement the same pipeline. The notebook adds matplotlib visualizations and LaTeX-rendered explanations for the paper. Changes to the pipeline logic should be kept in sync between both.

## Dependencies

`requirements.txt` covers both prototypes:
- **SBERT (Prototype 2):** `sentence-transformers`, `datasets`
- **Word2Vec (Prototype 1):** `gensim`, `tabulate` (`tabulate` is required by `pandas.to_markdown` for `word2vec_neighbors.md`)
- **Shared:** `pandas`, `numpy`, `pyyaml`, `scikit-learn`, `scipy`, `matplotlib`
- **Notebooks & tests:** `ipykernel`, `pytest`

`gensim 4.4.0` is verified working on this Python 3.13 `.venv` (it provides a cp313 wheel); the Docker image (Python 3.12) is a fallback if a future environment can't install it.

## Docker

Each prototype has its own Dockerfile under `src/<prototype>/Dockerfile`:

```
src/sbert/Dockerfile        ← Prototyp 2 (Python 3.12)
src/word2vec/Dockerfile     ← Prototyp 1 (Python 3.12; gensim wheels)
docker-compose.yaml         ← verwaltet beide Services (sbert, word2vec)
.dockerignore               ← schließt .venv/, sbert-prototyp/, outputs/, data/ aus
```

**Container starten (Docker Desktop oder CLI):**

```powershell
# SBERT-Evaluation (Ergebnisse in ./outputs/tables/)
docker compose run sbert

# Word2Vec-Ablauf (Download → Training → Auswertung; Ergebnisse in ./outputs/)
docker compose run word2vec

# Interaktive Shell im Container
docker compose run sbert bash

# Image neu bauen (nach Code-Änderungen)
docker compose build word2vec
```

Der Build-Kontext ist immer der Projektroot (`.`). Beim SBERT-Service wird das HuggingFace-Modell im benannten Volume `hf-cache` zwischengespeichert. Beim Word2Vec-Service werden `./outputs` und `./data` als Volumes gemountet, sodass Modell, Ergebnisse und der heruntergeladene Korpus auf dem Host erhalten bleiben.
