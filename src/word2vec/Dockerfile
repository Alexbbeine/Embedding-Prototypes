# ─────────────────────────────────────────────────────────────────────────────
# Prototyp 1 – Word2Vec
# Trainiert statische deutsche Wortvektoren (Gensim) auf dem Leipzig-Korpus und
# analysiert Wortnachbarschaften.
#
# Verwendung (aus dem Projektroot):
#   docker compose run word2vec                # kompletten Ablauf ausführen
#   docker compose run word2vec bash           # interaktive Shell
#   docker compose build word2vec              # Image neu bauen (nach Code-Änderungen)
#
# Basis ist Python 3.12 – dafür stellen gensim/numpy/scipy fertige Wheels bereit
# (kein Compiler nötig). Das vermeidet die gensim-Probleme unter Python 3.13.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# Abhängigkeiten zuerst installieren (Docker-Layer-Caching).
# Nur die Word2Vec-spezifische requirements-word2vec.txt wird verwendet –
# ohne SBERT-, Notebook- und Test-Pakete.
COPY requirements-word2vec.txt .
RUN pip install --no-cache-dir -r requirements-word2vec.txt

# Quellcode & Konfiguration kopieren
COPY src/     src/
COPY configs/ configs/

# Ausgabe- und Datenverzeichnisse anlegen (werden zur Laufzeit als Volumes gemountet)
RUN mkdir -p outputs/tables outputs/figures outputs/models data/raw data/processed

# UTF-8 für Umlaute in der Konsolenausgabe
ENV PYTHONIOENCODING=utf-8

# Standard-Befehl: vollständiger Ablauf (Download → Aufbereitung → Training → Auswertung).
# Überschreibbar, z. B.:
#   docker compose run word2vec python -m src.word2vec.evaluate
CMD ["python", "-m", "src.word2vec.pipeline"]
