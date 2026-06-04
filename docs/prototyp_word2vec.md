# Prototyp 1: Word2Vec zur Analyse statischer Wortvektoren

## Ziel

Der Prototyp untersucht, welche semantisch oder thematisch plausiblen Wortnachbarschaften in einem selbst trainierten Word2Vec-Modell auf einem begrenzten deutschsprachigen Korpus entstehen. Die Untersuchung bewertet nicht Word2Vec allgemein, sondern nur die Ergebnisse unter den hier festgelegten Bedingungen.

## Korpus und Stichprobe

Verwendet wird der Korpus `deu_news_2010_100K` der Leipzig Corpora Collection. Es handelt sich um einen deutschen Nachrichten-Teilkorpus aus dem Jahr 2010 mit 100.000 Sätzen. Nach der projekteigenen Tokenisierung (Kleinschreibung, nur Wort-Tokens, Satzzeichen und Ziffern verworfen) verbleiben 1.625.516 Tokens.

Quelle: <https://corpora.wortschatz-leipzig.de/en?corpusId=deu_news_2010_100K>

Für das Training wird der vollständige 100K-Korpus verwendet. Dadurch ist die Datengrundlage fest und bei erneuter Ausführung identisch, solange dieselbe Leipzig-Korpusdatei genutzt wird. Ein kleinerer zufälliger Ausschnitt ist technisch weiterhin möglich (`sample_size` in `configs/word2vec.yaml` bzw. `--sample-size`), wird aber nur für schnelle Testläufe verwendet.

## Vorverarbeitung

Die Leipzig-Datei wird im Format `Sentence_ID<TAB>Sentence` gelesen. Danach werden HTML-Reste, leere Zeilen, fehlerhafte technische Zeichen und doppelte Leerzeichen entfernt. Die Texte werden kleingeschrieben und auf Wortebene tokenisiert. Deutsche Umlaute bleiben erhalten, Satzzeichen werden nicht als eigene Tokens übernommen.

## Trainingsparameter

Alle Parameter sind in `configs/word2vec.yaml` unter `word2vec_params` gepflegt.

| Parameter | Wert | Bedeutung |
| --- | ---: | --- |
| `vector_size` | 100 | Dimension der Wortvektoren |
| `window` | 5 | Kontextfenster links und rechts |
| `min_count` | 5 | Mindesthäufigkeit eines Wortes |
| `sg` | 1 | Skip-Gram-Modell |
| `epochs` | 10 | Trainingsdurchläufe |
| `seed` | 42 | Reproduzierbarkeit (aus `random_seed`) |
| `workers` | 1 | Stabilere Reproduzierbarkeit |

Die Parameter sind nicht als optimal zu verstehen, sondern als nachvollziehbare Basiskonfiguration für eine prototypische Umsetzung.

## Zielwörter und Auswertung

Untersucht werden Zielwörter aus Politik, Wirtschaft, Technik und mehrdeutigen Bereichen (konfiguriert unter `target_words`):

`regierung`, `wahl`, `partei`, `markt`, `unternehmen`, `bank`, `daten`, `software`, `modell`, `netz`

Falls ein Zielwort nicht im trainierten Vokabular enthalten ist, wird ein vordefiniertes häufigeres Ersatzwort aus derselben Kategorie verwendet. Für jedes Zielwort werden die fünf nächsten Nachbarn anhand der Cosine Similarity ausgegeben.

Die Ergebnisdateien liegen nach der Ausführung in:

- `outputs/tables/word2vec_neighbors.csv`
- `outputs/tables/word2vec_neighbors.md`
- `outputs/tables/word2vec_target_words.json`
- `outputs/tables/word2vec_model_metadata.json`
- `outputs/figures/word2vec_pca_neighbors.png`
- `outputs/figures/word2vec_targets/pca_<zielwort>.png`

Das trainierte Modell liegt unter `outputs/models/`, die aufbereiteten Korpusdaten unter `data/`.

Die Nachbarschaften werden anschließend fachlich in drei Gruppen eingeordnet (Spalte `einordnung`):

| Gruppe | Bedeutung |
| --- | --- |
| semantisch plausibel | ähnliche Bedeutung |
| thematisch plausibel | gleiches Themenfeld |
| problematisch | unklare oder irreführende Nähe |

## Visualisierung

Die globale PCA-Visualisierung reduziert nur Zielwörter und deren direkte Nachbarn auf zwei Dimensionen. Zusätzlich wird für jedes einzelne Zielwort eine eigene PCA-Grafik mit Zielwort und direkten Nachbarn erzeugt. Beide Formen dienen der explorativen Veranschaulichung und werden nicht als eigenständiger Leistungsnachweis interpretiert.

## Ausführung

```powershell
# Kompletter Ablauf (Download -> Aufbereitung -> Training -> Auswertung)
.venv\Scripts\python.exe -m src.word2vec.pipeline

# Einzelschritte
.venv\Scripts\python.exe -m src.word2vec.download
.venv\Scripts\python.exe -m src.word2vec.prepare
.venv\Scripts\python.exe -m src.word2vec.train
.venv\Scripts\python.exe -m src.word2vec.evaluate
```

Hinweis: `gensim` wird für `train`/`evaluate` benötigt. `gensim 4.4.0` stellt ein Python-3.13-Wheel bereit und läuft in der vorhandenen `.venv` (Python 3.13). Alternativ steht der Docker-Container (`docker compose run word2vec`, Basis Python 3.12) bereit.

## Grenzen

Die Ergebnisse hängen stark vom verwendeten Korpus, der Stichprobengröße, der Vorverarbeitung und den Trainingsparametern ab. Word2Vec erzeugt statische Wortvektoren und kann unterschiedliche Bedeutungen eines Wortes nicht kontextabhängig unterscheiden. Die PCA-Darstellung ist zudem eine vereinfachte zweidimensionale Projektion eines höherdimensionalen Vektorraums.

## Zwischenfazit

Der Prototyp zeigt, wie aus einem begrenzten deutschsprachigen Korpus statische Wortvektoren trainiert und Wortnachbarschaften sichtbar gemacht werden können. Die Ergebnisse können plausible semantische oder thematische Relationen zeigen, bleiben aber methodisch an die konkrete Datenbasis und Parametrisierung gebunden.
