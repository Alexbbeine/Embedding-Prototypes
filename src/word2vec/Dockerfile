# ─────────────────────────────────────────────────────────────────────────────
# Prototyp 1 – Word2Vec  [PLATZHALTER – noch nicht implementiert]
#
# Diese Dockerfile wird befüllt, sobald src/word2vec/ fertiggestellt ist.
# Der Aufbau folgt dem gleichen Schema wie docker/sbert/Dockerfile.
#
# Verwendung (sobald implementiert):
#   docker compose run word2vec
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/     src/
COPY configs/ configs/

RUN mkdir -p outputs/tables outputs/figures

ENV PYTHONIOENCODING=utf-8

# CMD wird ergänzt, sobald das Einstiegsskript existiert:
# CMD ["python", "src/word2vec/train.py", "--config", "configs/word2vec.yaml"]
CMD ["bash"]
