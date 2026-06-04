# Vom Wort zum Vektor – Embeddings für Texte

Projekt (DHBW) zum Thema Text-Embeddings. Das Repository enthält zwei
in sich abgeschlossene Python-Prototypen, die zwei unterschiedliche Ansätze zur
Vektorrepräsentation von Sprache demonstrieren:

- **Prototyp 1 – Word2Vec:** statische Wortvektoren, trainiert auf einem
  deutschen Nachrichtenkorpus. Untersucht werden die nächsten Nachbarn
  ausgewählter Zielwörter.
- **Prototyp 2 – Sentence-BERT (SBERT):** kontextuelle Satz-Embeddings eines
  vortrainierten multilingualen Modells. Untersucht wird die semantische
  Ähnlichkeit von Satzpaaren gegen einen Gold-Standard.

Beide Prototypen werden **über Docker** ausgeführt – das ist der empfohlene und
einfachste Weg. Zusätzlich liegen sie jeweils als ausführbares Skript bzw. CLI
**und** als narratives Jupyter-Notebook vor; die lokale Ausführung ohne Docker
ist optional (siehe unten). Es gibt keinen Webserver, keine Datenbank und keine
API.

---

## Projektstruktur

```
Embedding-Prototypes/
│
├── README.md                       Dieses Dokument
│
├── docker-compose.yaml             Beide Prototypen als Container-Services
│
├── configs/                        Zentrale Konfiguration je Prototyp
│   ├── word2vec.yaml               Korpus, Hyperparameter, Zielwörter, Pfade
│   └── sbert.yaml                  Modell, Datensatz, Schwellenwerte, Pfade
│
├── src/
│   ├── word2vec/                   Prototyp 1 (Word2Vec)
│   │   ├── config.py               Lädt YAML, leitet Parameter + Pfade ab
│   │   ├── corpus.py               Leipzig-Parsing, Bereinigung, Tokenisierung
│   │   ├── download.py             Korpus-Archiv herunterladen
│   │   ├── prepare.py              Sampeln, tokenisieren, JSONL + Metadaten
│   │   ├── train.py                gensim-Word2Vec-Training (lazy import)
│   │   ├── evaluate.py             Nächste Nachbarn der Zielwörter
│   │   ├── visualize.py            PCA-Plots (lazy matplotlib/sklearn)
│   │   ├── pipeline.py             Orchestriert download -> prepare -> train -> evaluate
│   │   └── Dockerfile              Prototyp-1-Container (Python 3.12)
│   │
│   └── sbert/                      Prototyp 2 (Sentence-BERT)
│       ├── evaluate_similarity.py  Hauptskript; liest Config, steuert Pipeline
│       ├── similarity.py           Embeddings, Cosinus-Ähnlichkeit, Kategorisierung
│       ├── evaluation.py           Metriken (Pearson, Spearman, MAE)
│       ├── diagnostics.py          12 deutsche Edge-Case-Satzpaare
│       ├── visualize.py            Notebook-Abbildungen (lazy matplotlib)
│       └── Dockerfile              Prototyp-2-Container (Python 3.12)
│
├── notebooks/                      Narrative Walkthroughs (gleiche Logik wie src/)
│   ├── 01_word2vec_exploration.ipynb
│   └── 02_sbert_similarity.ipynb
│
├── data/                           Rohkorpus + verarbeitete Daten
├── outputs/
│   ├── models/                     Trainierte Word2Vec-Modelle
│   ├── tables/                     CSV / JSON / Markdown-Ergebnisse
│   └── figures/                    PNG-Abbildungen
│
├── tests/                          pytest (Korpus, Evaluate, Smoke-Test)
├── docs/                           Methodische Ausarbeitung für die Arbeit
│
├── requirements.txt                Komplettumgebung (beide Prototypen + Tools)
├── requirements-word2vec.txt       Nur Prototyp 1
└── requirements-sbert.txt          Nur Prototyp 2
```

### Abhängigkeitsüberblick

| Bereich | Pakete |
|---|---|
| Word2Vec (Prototyp 1) | `gensim`, `tabulate` |
| SBERT (Prototyp 2) | `sentence-transformers`, `datasets` |
| Gemeinsam genutzt | `pandas`, `numpy`, `pyyaml`, `scikit-learn`, `scipy`, `matplotlib` |
| Notebooks & Tests | `ipykernel`, `pytest`, `nbconvert`, `nbclient` |

Bei der Ausführung über Docker müssen diese Abhängigkeiten **nicht** manuell
installiert werden – die Container-Images bringen sie mit. Die
`requirements-*.txt` werden nur für die optionale lokale Ausführung benötigt.

---

## Voraussetzungen

Für den empfohlenen Weg genügt **Docker**:

- **Docker Desktop** (Windows/macOS) bzw. **Docker Engine + Compose-Plugin**
  (Linux).
- Internetzugang beim ersten Lauf:
  - Word2Vec lädt den Leipzig-Korpus `deu_news_2010_100K` herunter.
  - SBERT lädt das HuggingFace-Modell und den Datensatz `mteb/stsb_multi_mt`.

Mehr ist für die Container-Ausführung nicht nötig: Python, `pip` und alle
Bibliotheken stecken bereits in den Images.

Alle Befehle gehen vom **Projektroot** als Arbeitsverzeichnis aus. Der
Build-Kontext der Container ist ebenfalls immer der Projektroot.

---

## Schnellstart (Docker)

```powershell
# Prototyp 2 – SBERT: Image bauen und Evaluation ausführen
docker compose build sbert
docker compose run sbert

# Prototyp 1 – Word2Vec: Image bauen und kompletten Ablauf ausführen
docker compose build word2vec
docker compose run word2vec
```

Die Ergebnisse (CSV/JSON/Markdown/PNG) erscheinen automatisch unter
`./outputs/` auf dem Host.

---

## Verwendung über Docker

`docker-compose.yaml` verwaltet beide Prototypen als eigene Services. Jeder
Service besitzt sein eigenes Dockerfile unter `src/<prototyp>/Dockerfile`.

### Prototyp 1 – Word2Vec

```powershell
# Kompletter Ablauf: Download -> Aufbereitung -> Training -> Auswertung
docker compose run word2vec

# Image nach Code-Änderungen neu bauen
docker compose build word2vec

# Interaktive Shell im Container (z. B. für Einzelschritte)
docker compose run word2vec bash
```

Ergebnisse:

- `outputs/models/` – trainiertes Modell + Metadaten
- `outputs/tables/` – `word2vec_neighbors.csv` / `.md`,
  `word2vec_target_words.json`, `word2vec_model_metadata.json`
- `outputs/figures/` – `word2vec_pca_neighbors.png` und Per-Zielwort-PCA-Plots

`./outputs` und `./data` sind als Volumes gemountet, sodass Modell, Ergebnisse
und der heruntergeladene Korpus auf dem Host erhalten bleiben.

Konfiguration über `configs/word2vec.yaml` (Korpus, Hyperparameter wie
`vector_size`/`window`/`min_count`/`sg`/`epochs`, `topn`, Zielwörter, Seed,
`sample_size`).

### Prototyp 2 – Sentence-BERT

```powershell
# Modell + Datensatz laden, Cosinus-Ähnlichkeit berechnen, Ergebnisse speichern
docker compose run sbert

# Image nach Code-Änderungen neu bauen
docker compose build sbert

# Interaktive Shell im Container
docker compose run sbert bash
```

Ergebnisse:

- `outputs/tables/` – `sbert_similarity_results.csv`, `sbert_summary.json`,
  `sbert_diagnostic_results.csv`
- `outputs/figures/` – Heatmap, Score-Verteilung, Scatter (Gold vs. Predicted),
  Cosinus nach Kategorie, Diagnose-Abbildung

Das HuggingFace-Modell wird beim ersten Start heruntergeladen und im benannten
Volume `hf-cache` zwischengespeichert – kein erneuter Download bei Neustarts.

Konfiguration über `configs/sbert.yaml` (Modellname, Datensatz, Split,
`sample_per_category`, Schwellenwerte für die Ähnlichkeitskategorien, Pfade).

---

## Optional: Lokale Ausführung ohne Docker

Wer ohne Container arbeiten möchte (z. B. zum Debuggen oder für die Notebooks),
kann die Prototypen direkt in einem virtuellen Environment ausführen.

> Empfohlene Umgebung: **Python 3.13** (verifiziert; `gensim 4.4.0` liefert ein
> passendes cp313-Wheel). Sollte `gensim` lokal nicht installierbar sein, ist
> der Docker-Container der Fallback.

### Installation

```powershell
# Environment anlegen
python -m venv .venv

# Komplettumgebung (beide Prototypen + Notebooks + Tests)
.venv\Scripts\pip install -r requirements.txt
```

Nur einen Prototyp installieren:

```powershell
.venv\Scripts\pip install -r requirements-word2vec.txt   # nur Word2Vec
.venv\Scripts\pip install -r requirements-sbert.txt      # nur SBERT
```

Unter Linux/macOS entsprechend `.venv/bin/python` bzw. `.venv/bin/pip`
verwenden. Es wird kein Aktivierungsskript vorausgesetzt – der Interpreter wird
explizit über seinen Pfad aufgerufen.

### Word2Vec

```powershell
# Kompletter Ablauf
.venv\Scripts\python.exe -m src.word2vec.pipeline

# Download überspringen, falls das Archiv schon in data/raw/ liegt
.venv\Scripts\python.exe -m src.word2vec.pipeline --skip-download

# Einzelschritte
.venv\Scripts\python.exe -m src.word2vec.download
.venv\Scripts\python.exe -m src.word2vec.prepare    # optional: --sample-size 20000
.venv\Scripts\python.exe -m src.word2vec.train
.venv\Scripts\python.exe -m src.word2vec.evaluate
```

### Sentence-BERT

```powershell
# Hauptauswertung
.venv\Scripts\python.exe src/sbert/evaluate_similarity.py

# Mit explizitem Config-Pfad
.venv\Scripts\python.exe src/sbert/evaluate_similarity.py --config configs/sbert.yaml

# Als Modul
.venv\Scripts\python.exe -m src.sbert.evaluate_similarity
```

### Notebook-Prototypen

Zu jedem Prototyp gibt es ein Jupyter-Notebook mit demselben Ablauf wie das
jeweilige Skript, zusätzlich aber mit Visualisierungen und erläuterndem Text
(inkl. LaTeX) für die schriftliche Ausarbeitung:

- `notebooks/01_word2vec_exploration.ipynb` – narrativer Durchlauf des
  Word2Vec-Workflows
- `notebooks/02_sbert_similarity.ipynb` – narrativer Durchlauf der
  SBERT-Ähnlichkeitsanalyse

Vorbereitung (einmalig):

```powershell
.venv\Scripts\pip install ipykernel
```

Anschließend das Notebook in VS Code oder Jupyter öffnen und `.venv` als Kernel
auswählen. Alternativ headless ausführen:

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks\02_sbert_similarity.ipynb
```

> Die Pipeline-Logik in Notebook und Skript wird bewusst synchron gehalten.
> Änderungen an der Logik sollten in beiden gepflegt werden.

### Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

Die Korpus- und Zielwort-Unit-Tests laufen immer; der Word2Vec-Smoke-Test wird
automatisch übersprungen, wenn `gensim` (oder eine andere ML-Abhängigkeit)
fehlt. `tests/conftest.py` setzt den Projektroot auf den `sys.path`.

---

## Weiterführende Dokumentation

- `docs/prototyp_word2vec.md` – Methodik-Ausarbeitung Prototyp 1 (Word2Vec)
- `docs/prototyp_sbert.md` – Methodik-Ausarbeitung Prototyp 2 (Sentence-BERT)
