# ─────────────────────────────────────────────────────────────────────────────
# Prototyp 2 – Sentence-BERT
# Semantische Satzähnlichkeit mit sentence-transformers auf dem deutschen STS-B.
#
# Verwendung (aus dem Projektroot):
#   docker compose build sbert               # Image bauen
#   docker compose run sbert                 # Evaluation ausführen
#   docker compose run sbert bash            # Interaktive Shell
# ─────────────────────────────────────────────────────────────────────────────

# Basis-Image
# python:3.12-slim: minimale Debian-Basis mit CPython 3.12.
# Keine Desktop-Tools, kein Compiler – reicht für reine Python-Pakete.
FROM python:3.12-slim

# Arbeitsverzeichnis
WORKDIR /app

# Abhängigkeiten installieren
# requirements-sbert.txt wird zuerst kopiert, damit Docker den Layer cached.
# Nur bei Änderungen daran wird neu installiert, nicht bei jeder Code-Änderung.
# Ohne Word2Vec-, Notebook- und Test-Pakete.
COPY requirements-sbert.txt .
# CPU-only PyTorch zuerst installieren. Dadurch zieht sentence-transformers
# keine CUDA-Bibliotheken nach (~1,5 GB weniger: cudnn, cublas, triton, etc.).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-sbert.txt

# Quellcode & Konfiguration kopieren
COPY src/     src/
COPY configs/ configs/

# Ausgabeverzeichnisse anlegen
# Ohne diesen Schritt schlägt der CSV/JSON-Export fehl, wenn das outputs-Volume beim ersten Start noch leer ist.
RUN mkdir -p outputs/tables outputs/figures

# Umgebung
# UTF-8 für Umlaute und Box-Drawing-Zeichen in der Konsolenausgabe
ENV PYTHONIOENCODING=utf-8
# Unterdrückt verbose Warnungen von HuggingFace (kein API-Token erforderlich)
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_VERBOSITY=warning

# Standard-Befehl
# Führt die vollständige SBERT-Evaluation aus (Datensatz + Diagnostik).
# Kann beim Container-Start mit einem anderen Befehl überschrieben werden:
#   docker compose run sbert python src/sbert/evaluate_similarity.py --config ...
CMD ["python", "src/sbert/evaluate_similarity.py", "--config", "configs/sbert.yaml"]
