"""
corpus.py
=========
Einlesen, Bereinigen und Tokenisieren des Leipzig-Korpus.

Reine Hilfsfunktionen ohne Konfigurations- oder Pfadabhängigkeiten, damit sie
unabhängig getestet werden können (siehe ``tests/test_corpus.py``).

Die Leipzig-Sätze liegen im Format ``Sentence_ID<TAB>Sentence`` vor. Es werden
HTML-Reste entfernt, Texte kleingeschrieben und auf Wortebene tokenisiert;
deutsche Umlaute bleiben erhalten, Satzzeichen werden verworfen.
"""
from __future__ import annotations

import html
import json
import random
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

TOKEN_RE = re.compile(r"[a-zäöüß]+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def read_leipzig_sentences(lines: Iterable[str]) -> Iterator[str]:
    """Liefert den Satztext aus Leipzig-Zeilen: Sentence_ID<TAB>Sentence."""
    for line in lines:
        row = line.strip()
        if not row:
            continue
        if "\t" not in row:
            continue
        sentence_id, sentence = row.split("\t", 1)
        if not sentence_id.isdigit():
            continue
        sentence = sentence.strip()
        if sentence:
            yield sentence


def iter_sentences_from_archive(archive_path: Path) -> Iterator[str]:
    """Iteriert die Sätze direkt aus dem Leipzig-Archiv (*.tar.gz)."""
    with tarfile.open(archive_path, "r:gz") as archive:
        sentence_members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and (member.name.endswith("_sentences.txt") or member.name.endswith("-sentences.txt"))
        ]
        if not sentence_members:
            raise FileNotFoundError(f"No *_sentences.txt file found in {archive_path}.")

        member = sentence_members[0]
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Could not read {member.name} from {archive_path}.")

        for raw_line in extracted:
            yield from read_leipzig_sentences([raw_line.decode("utf-8", errors="replace")])


def clean_text(text: str) -> str:
    """Entfernt HTML-Reste und doppelte Leerzeichen."""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace(chr(0xFEFF), " ")  # Byte Order Mark (BOM) entfernen
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Bereinigt, schreibt klein und zerlegt in Wort-Tokens (Umlaute bleiben erhalten)."""
    cleaned = clean_text(text).lower()
    return TOKEN_RE.findall(cleaned)


def sample_sentences(sentences: Sequence[str], sample_size: int, seed: int) -> list[str]:
    """Zieht eine deterministische Zufallsstichprobe (oder gibt alle Sätze zurück)."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")
    if sample_size >= len(sentences):
        return list(sentences)
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(sentences)), sample_size))
    return [sentences[index] for index in indexes]


def tokenize_sentences(
    sentences: Iterable[str],
    min_sentence_tokens: int = 2,
) -> list[list[str]]:
    """Tokenisiert alle Sätze und verwirft zu kurze."""
    tokenized: list[list[str]] = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        if len(tokens) >= min_sentence_tokens:
            tokenized.append(tokens)
    return tokenized


def save_tokenized_sentences(sentences: Iterable[list[str]], path: Path) -> None:
    """Speichert tokenisierte Sätze als JSON Lines (eine Liste je Zeile)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for tokens in sentences:
            file.write(json.dumps(tokens, ensure_ascii=False) + "\n")


def load_tokenized_sentences(path: Path) -> list[list[str]]:
    """Lädt tokenisierte Sätze aus einer JSON-Lines-Datei."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def corpus_stats(tokenized_sentences: Sequence[Sequence[str]]) -> dict[str, int]:
    """Berechnet Satz-, Token- und Vokabularumfang des tokenisierten Korpus."""
    counts = Counter(token for sentence in tokenized_sentences for token in sentence)
    return {
        "sentences": len(tokenized_sentences),
        "tokens": sum(counts.values()),
        "raw_vocabulary_size": len(counts),
    }
