# Prototyp 2: Sentence-BERT zur semantischen Satzähnlichkeit

## Ziel

Der Prototyp untersucht, wie gut ein vortrainiertes Sentence-BERT-Modell die semantische Ähnlichkeit deutscher Satzpaare einschätzt. Gemessen wird, wie stark die vom Modell berechnete Kosinusähnlichkeit mit menschlich annotierten Ähnlichkeitsurteilen übereinstimmt. Zusätzlich wird das Verhalten an gezielt konstruierten linguistischen Grenzfällen betrachtet. Die Untersuchung bewertet nicht Sentence-BERT allgemein, sondern nur die Ergebnisse unter den hier festgelegten Bedingungen.

## Modell und Datensatz

Verwendet wird das Modell `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Es erzeugt für einen gesamten Satz einen einzelnen Vektor (Satz-Embedding) und ist mehrsprachig trainiert, deckt also auch Deutsch ab. Im Gegensatz zu statischen Wortvektoren entsteht das Embedding kontextabhängig: derselbe Begriff kann je nach Satz unterschiedlich repräsentiert werden.

Als Datengrundlage dient der deutsche Teil des STS-Benchmark in der Fassung `mteb/stsb_multi_mt` (Sprachkonfiguration `de`, Split `dev`, 1500 Satzpaare). Jedes Paar trägt einen menschlich vergebenen Gold-Score auf der Skala 0 bis 5, wobei 0 für völlig unähnlich und 5 für bedeutungsgleich steht.

Quelle: <https://huggingface.co/datasets/mteb/stsb_multi_mt>

## Vorverarbeitung und Embeddings

Die Satzpaare werden als zwei getrennte Satzlisten (`sentence1`, `sentence2`) geladen. Beide Listen werden unabhängig voneinander mit dem Modell enkodiert. Eine eigene Textbereinigung ist nicht nötig, da der Tokenizer des Modells die Vorverarbeitung übernimmt. Die Gold-Scores werden von der Skala 0 bis 5 auf den Bereich 0 bis 1 normalisiert (Division durch 5), damit sie direkt mit der Kosinusähnlichkeit vergleichbar sind.

## Ähnlichkeitsberechnung und Kategorisierung

Für jedes Satzpaar wird die Kosinusähnlichkeit der beiden Embeddings berechnet. Die Vektoren werden dafür zuvor explizit L2-normalisiert, sodass das Ergebnis unabhängig davon korrekt ist, ob das Modell bereits normierte Vektoren liefert.

Auf Basis des Ähnlichkeitswerts (bzw. des normalisierten Gold-Scores) erfolgt eine Einordnung in drei lesbare Kategorien. Die Schwellenwerte sind in `configs/sbert.yaml` unter `thresholds` gepflegt:

| Kategorie | Bedingung |
| --- | --- |
| unähnlich | Wert < 0.3 |
| teilweise ähnlich | 0.3 <= Wert < 0.7 |
| sehr ähnlich | Wert >= 0.7 |

Die Schwellen sind nicht als optimal zu verstehen, sondern als nachvollziehbare Basiskonfiguration für eine prototypische Umsetzung.

## Evaluationsmetriken

Der Vergleich zwischen vorhergesagter Kosinusähnlichkeit und normalisiertem Gold-Score erfolgt über drei Kennzahlen:

| Metrik | Bedeutung |
| --- | --- |
| Pearson-Korrelation | linearer Zusammenhang zwischen Vorhersage und Gold-Score |
| Spearman-Korrelation | monotoner Rangzusammenhang, robuster gegen Ausreißer |
| Mean Absolute Error (MAE) | durchschnittliche absolute Abweichung auf der Skala 0 bis 1 |

Pearson und Spearman zeigen, wie gut das Modell die menschliche Rangfolge der Ähnlichkeit trifft; der MAE ist direkt interpretierbar (ein MAE von 0.16 bedeutet im Schnitt 16 Prozentpunkte Abweichung vom menschlichen Urteil).

## Diagnostische Satzpaare

Ergänzend zur Datensatz-Evaluation werden zwölf manuell konstruierte Satzpaare ausgewertet (`src/sbert/diagnostics.py`). Jedes Paar prüft ein linguistisches Phänomen, das embeddingbasierte Modelle herausfordert. Für jedes Paar ist die semantisch korrekte Kategorie (`expected`) hinterlegt; abweichende Vorhersagen sind für die Fehleranalyse aufschlussreich.

| Kategorie | Erwartung | Untersuchte Schwierigkeit |
| --- | --- | --- |
| Synonyme | sehr ähnlich | Bedeutungsgleichheit trotz anderer Wörter |
| Negation | unähnlich | Bedeutungsumkehr durch "nicht" |
| Zahlenunterschiede | teilweise ähnlich | gleiche Struktur, andere Fakten |
| Eigennamen | teilweise ähnlich | gleiche Rolle, andere Entität |
| Ähnliche Wortoberfläche | unähnlich | gleiche Wörter, andere Bedeutung |
| Mehrdeutige Begriffe | unähnlich | Homonymie und Polysemie |

Besonders die Negation gilt als bekannte Schwäche: Sentence-BERT unterschätzt ihren Effekt oft, weil "nicht" nur ein einzelnes Token mit geringem Gewicht ist. Ausgegeben wird je Paar die vorhergesagte Kategorie, die erwartete Kategorie und ob beide übereinstimmen, sowie eine Gesamttrefferquote.

## Ergebnisdateien

Die Ergebnisdateien liegen nach der Ausführung in:

- `outputs/tables/sbert_similarity_results.csv`
- `outputs/tables/sbert_summary.json`
- `outputs/tables/sbert_diagnostic_results.csv`
- `outputs/figures/demo_similarity_heatmap.png`
- `outputs/figures/gold_score_distribution.png`
- `outputs/figures/sbert_scatter_gold_vs_predicted.png`
- `outputs/figures/sbert_cosine_by_category.png`
- `outputs/figures/sbert_diagnostic_results.png`

## Visualisierung

Erzeugt werden fünf Grafiken: eine einleitende Demo-Ähnlichkeitsmatrix dreier Beispielsätze, die Verteilung der Gold-Scores mit eingezeichneten Schwellen, ein Streudiagramm von Gold-Score gegen vorhergesagte Kosinusähnlichkeit (mit Idealgerade y = x), die Verteilung der Kosinusähnlichkeit je Kategorie sowie das Ergebnis der diagnostischen Satzpaare (korrekt gegen falsch). Die Grafiken dienen der explorativen Veranschaulichung und werden nicht als eigenständiger Leistungsnachweis interpretiert.

## Ausführung

```powershell
# Vollständige Evaluation (Datensatz + Diagnostik)
.venv\Scripts\python.exe src/sbert/evaluate_similarity.py

# Mit explizitem Konfigurationspfad
.venv\Scripts\python.exe src/sbert/evaluate_similarity.py --config configs/sbert.yaml

# Als Modul
.venv\Scripts\python.exe -m src.sbert.evaluate_similarity
```

Hinweis: Beim ersten Lauf werden das Modell und der Datensatz von HuggingFace heruntergeladen. Alternativ steht der Docker-Container (`docker compose run sbert`, Basis Python 3.12, CPU-only PyTorch) bereit; dort wird das Modell im benannten Volume `hf-cache` zwischengespeichert.

## Grenzen

Die Ergebnisse hängen vom gewählten Modell, dem Datensatz und den festgelegten Schwellenwerten ab. Die Dreiteilung in Kategorien ist eine bewusste Vereinfachung eines kontinuierlichen Ähnlichkeitswerts. Bekannte Schwächen kontextueller Satz-Embeddings, insbesondere bei Negation und feinen faktischen Unterschieden, treten auch hier auf. Die diagnostischen Satzpaare sind manuell konstruiert und klein in der Zahl; sie illustrieren typisches Verhalten, erlauben aber keine statistisch belastbaren Aussagen.

## Zwischenfazit

Der Prototyp zeigt, wie ein vortrainiertes Sentence-BERT-Modell semantische Satzähnlichkeit für Deutsch berechnet und wie sich die Vorhersagen quantitativ gegen einen Gold-Standard und qualitativ gegen linguistische Grenzfälle prüfen lassen. Die Korrelationsmaße geben Aufschluss über die allgemeine Güte, während die diagnostischen Paare gezielt Stärken und Schwächen sichtbar machen. Die Aussagekraft bleibt an Modell, Datenbasis und Schwellenwahl gebunden.
</content>
