# ─────────────────────────────────────────────────────────────────────────────
# Prototyp 2 – Sentence-BERT
# Semantische Satzähnlichkeit mit sentence-transformers auf dem deutschen STS-B.
#
# Verwendung (aus dem Projektroot):
#   docker compose run sbert                 # Evaluation ausführen
#   docker compose run sbert bash            # Interaktive Shell
#   docker build -f docker/sbert/Dockerfile .  # Manueller Build
# ─────────────────────────────────────────────────────────────────────────────

# ── Basis-Image ───────────────────────────────────────────────────────────────
# python:3.12-slim: minimale Debian-Basis mit CPython 3.12.
# Keine Desktop-Tools, kein Compiler – reicht für reine Python-Pakete.
FROM python:3.12-slim

# ── Arbeitsverzeichnis ────────────────────────────────────────────────────────
WORKDIR /app

# ── Abhängigkeiten installieren ───────────────────────────────────────────────
# requirements.txt wird zuerst kopiert, damit Docker den Layer cached.
# Nur bei Änderungen an requirements.txt wird neu installiert,
# nicht bei jeder Code-Änderung.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Quellcode & Konfiguration kopieren ────────────────────────────────────────
COPY src/     src/
COPY configs/ configs/

# ── Ausgabeverzeichnisse anlegen ──────────────────────────────────────────────
# Ohne diesen Schritt schlägt der CSV/JSON-Export fehl,
# wenn das outputs-Volume beim ersten Start noch leer ist.
RUN mkdir -p outputs/tables outputs/figures

# ── Umgebung ──────────────────────────────────────────────────────────────────
# UTF-8 für Umlaute und Box-Drawing-Zeichen in der Konsolenausgabe
ENV PYTHONIOENCODING=utf-8
# Unterdrückt verbose Warnungen von HuggingFace (kein API-Token erforderlich)
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_VERBOSITY=warning

# ── Standard-Befehl ───────────────────────────────────────────────────────────
# Führt die vollständige SBERT-Evaluation aus (Datensatz + Diagnostik).
# Kann beim Container-Start mit einem anderen Befehl überschrieben werden:
#   docker compose run sbert python src/sbert/evaluate_similarity.py --config ...
CMD ["python", "src/sbert/evaluate_similarity.py", "--config", "configs/sbert.yaml"]
