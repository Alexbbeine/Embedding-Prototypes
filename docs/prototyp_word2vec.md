# Prototyp 1: Word2Vec zur Analyse statischer Wortvektoren

## Ziel

Der Prototyp untersucht, welche semantisch oder thematisch plausiblen Wortnachbarschaften in einem selbst trainierten Word2Vec-Modell auf einem begrenzten deutschsprachigen Korpus entstehen. Die Untersuchung bewertet nicht Word2Vec allgemein, sondern nur die Ergebnisse unter den hier festgelegten Bedingungen.

## Korpus und Stichprobe

Verwendet wird der Korpus `deu_news_2010_100K` der Leipzig Corpora Collection. Es handelt sich um einen deutschen Nachrichten-Teilkorpus aus dem Jahr 2010 mit 100.000 Saetzen und 1.705.473 Tokens.

Quelle: <https://corpora.wortschatz-leipzig.de/en?corpusId=deu_news_2010_100K>

Fuer das Training wird der vollstaendige 100K-Korpus verwendet. Dadurch ist die Datengrundlage fest und bei erneuter Ausfuehrung identisch, solange dieselbe Leipzig-Korpusdatei genutzt wird. Ein kleinerer zufaelliger Ausschnitt ist technisch weiterhin moeglich (`sample_size` in `configs/word2vec.yaml` bzw. `--sample-size`), wird aber nur fuer schnelle Testlaeufe verwendet.

## Vorverarbeitung

Die Leipzig-Datei wird im Format `Sentence_ID<TAB>Sentence` gelesen. Danach werden HTML-Reste, leere Zeilen, fehlerhafte technische Zeichen und doppelte Leerzeichen entfernt. Die Texte werden kleingeschrieben und auf Wortebene tokenisiert. Deutsche Umlaute bleiben erhalten, Satzzeichen werden nicht als eigene Tokens uebernommen.

## Trainingsparameter

Alle Parameter sind in `configs/word2vec.yaml` unter `word2vec_params` gepflegt.

| Parameter | Wert | Bedeutung |
| --- | ---: | --- |
| `vector_size` | 100 | Dimension der Wortvektoren |
| `window` | 5 | Kontextfenster links und rechts |
| `min_count` | 5 | Mindesthaeufigkeit eines Wortes |
| `sg` | 1 | Skip-Gram-Modell |
| `epochs` | 10 | Trainingsdurchlaeufe |
| `seed` | 42 | Reproduzierbarkeit (aus `random_seed`) |
| `workers` | 1 | Stabilere Reproduzierbarkeit |

Die Parameter sind nicht als optimal zu verstehen, sondern als nachvollziehbare Basiskonfiguration fuer eine prototypische Umsetzung.

## Zielwoerter und Auswertung

Untersucht werden Zielwoerter aus Politik, Wirtschaft, Technik und mehrdeutigen Bereichen (konfiguriert unter `target_words`):

`regierung`, `wahl`, `partei`, `markt`, `unternehmen`, `bank`, `daten`, `software`, `modell`, `netz`

Falls ein Zielwort nicht im trainierten Vokabular enthalten ist, wird ein vordefiniertes haeufigeres Ersatzwort aus derselben Kategorie verwendet. Fuer jedes Zielwort werden die fuenf naechsten Nachbarn anhand der Cosine Similarity ausgegeben.

Die Ergebnisdateien liegen nach der Ausfuehrung in:

- `outputs/tables/word2vec_neighbors.csv`
- `outputs/tables/word2vec_neighbors.md`
- `outputs/tables/word2vec_target_words.json`
- `outputs/tables/word2vec_model_metadata.json`
- `outputs/figures/word2vec_pca_neighbors.png`
- `outputs/figures/word2vec_targets/pca_<zielwort>.png`

Das trainierte Modell liegt unter `outputs/models/`, die aufbereiteten Korpusdaten unter `data/`.

Die Nachbarschaften werden anschliessend fachlich in drei Gruppen eingeordnet (Spalte `einordnung`):

| Gruppe | Bedeutung |
| --- | --- |
| semantisch plausibel | aehnliche Bedeutung |
| thematisch plausibel | gleiches Themenfeld |
| problematisch | unklare oder irrefuehrende Naehe |

## Visualisierung

Die globale PCA-Visualisierung reduziert nur Zielwoerter und deren direkte Nachbarn auf zwei Dimensionen. Zusaetzlich wird fuer jedes einzelne Zielwort eine eigene PCA-Grafik mit Zielwort und direkten Nachbarn erzeugt. Beide Formen dienen der explorativen Veranschaulichung und werden nicht als eigenstaendiger Leistungsnachweis interpretiert.

## Ausfuehrung

```powershell
# Kompletter Ablauf (Download -> Aufbereitung -> Training -> Auswertung)
.venv\Scripts\python.exe -m src.word2vec.pipeline

# Einzelschritte
.venv\Scripts\python.exe -m src.word2vec.download
.venv\Scripts\python.exe -m src.word2vec.prepare
.venv\Scripts\python.exe -m src.word2vec.train
.venv\Scripts\python.exe -m src.word2vec.evaluate
```

Hinweis: `gensim` wird fuer `train`/`evaluate` benoetigt. `gensim 4.4.0` stellt ein Python-3.13-Wheel bereit und laeuft in der vorhandenen `.venv` (Python 3.13). Alternativ steht der Docker-Container (`docker compose run word2vec`, Basis Python 3.12) bereit.

## Grenzen

Die Ergebnisse haengen stark vom verwendeten Korpus, der Stichprobengroesse, der Vorverarbeitung und den Trainingsparametern ab. Word2Vec erzeugt statische Wortvektoren und kann unterschiedliche Bedeutungen eines Wortes nicht kontextabhaengig unterscheiden. Die PCA-Darstellung ist zudem eine vereinfachte zweidimensionale Projektion eines hoeherdimensionalen Vektorraums.

## Zwischenfazit

Der Prototyp zeigt, wie aus einem begrenzten deutschsprachigen Korpus statische Wortvektoren trainiert und Wortnachbarschaften sichtbar gemacht werden koennen. Die Ergebnisse koennen plausible semantische oder thematische Relationen zeigen, bleiben aber methodisch an die konkrete Datenbasis und Parametrisierung gebunden.
