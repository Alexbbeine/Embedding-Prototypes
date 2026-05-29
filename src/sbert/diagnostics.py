"""
diagnostics.py
==============
Manuell konstruierte deutsche Satzpaare für die qualitative SBERT-Analyse.

Jedes Paar testet ein spezifisches linguistisches Phänomen, das Embedding-basierte Ähnlichkeitsmodelle vor Herausforderungen stellt.
Das Feld ``expected`` gibt die semantisch korrekte Kategorie an – Abweichungen des Modells sind aufschlussreich für die Fehleranalyse.

Kategorien
----------
Synonyme
    Gleiche Bedeutung, unterschiedliche Wörter.
    Erwartung: Modell erkennt semantische Äquivalenz trotz anderer Oberfläche.

Negation
    Logische Umkehrung durch 'nicht' oder negierende Quantoren.
    Bekannte Schwäche: SBERT behandelt negierte Sätze oft als ähnlich, weil sich die Token-Vektoren kaum unterscheiden.

Zahlenunterschiede
    Identische Satzstruktur und Domäne, aber verschiedene Zahlenwerte.
    Erwartung: Gleiche Struktur, aber andere Fakten → teilweise ähnlich.

Eigennamen
    Gleiche Rolle oder Institution, unterschiedliche Entitäten.
    Erwartung: Strukturelle Ähnlichkeit sollte nicht dominieren.

Ähnliche Wortoberfläche
    Ähnliche Wörter (Homonyme, polyseme Verben), aber verschiedene Bedeutung.
    Typische Fehlerquelle für bag-of-words-basierte Modelle.

Mehrdeutige Begriffe
    Polysemie und Homonymie: gleicher Begriff, völlig verschiedene Bedeutungen.
    Test, ob SBERT Disambiguierung durch Kontext leistet.
"""
from __future__ import annotations

from typing import TypedDict


class DiagnosticPair(TypedDict):
    id:       str
    category: str   # linguistische Kategorie des Tests
    sentence1: str
    sentence2: str
    expected: str   # 'unähnlich' | 'teilweise ähnlich' | 'sehr ähnlich'


DIAGNOSTIC_PAIRS: list[DiagnosticPair] = [

    # Synonyme
    # Erwartung: Modell erkennt semantische Äquivalenz trotz anderer Wörter.
    {
        "id":        "diag_01",
        "category":  "Synonyme",
        "sentence1": "Das Auto fährt sehr schnell.",
        "sentence2": "Das Fahrzeug ist äußerst flott.",
        "expected":  "sehr ähnlich",
    },
    {
        "id":        "diag_02",
        "category":  "Synonyme",
        "sentence1": "Der Mann ist sehr müde.",
        "sentence2": "Der Herr wirkt äußerst erschöpft.",
        "expected":  "sehr ähnlich",
    },

    # Negation
    # Erwartung: Negation kehrt die Bedeutung um → unähnlich.
    # Bekannte Schwäche: SBERT unterschätzt den Effekt von Negation, weil 'nicht' nur ein einziges Token mit geringem Gewicht ist.
    {
        "id":        "diag_03",
        "category":  "Negation",
        "sentence1": "Das Essen schmeckt hervorragend.",
        "sentence2": "Das Essen schmeckt nicht hervorragend.",
        "expected":  "unähnlich",
    },
    {
        "id":        "diag_04",
        "category":  "Negation",
        "sentence1": "Die Prüfung war einfach.",
        "sentence2": "Die Prüfung war nicht einfach.",
        "expected":  "unähnlich",
    },

    # Zahlenunterschiede
    # Erwartung: Gleiche Satzstruktur und Domäne, aber andere Fakten → teilweise ähnlich.
    {
        "id":        "diag_05",
        "category":  "Zahlenunterschiede",
        "sentence1": "Der Zug fährt um 8 Uhr ab.",
        "sentence2": "Der Zug fährt um 20 Uhr ab.",
        "expected":  "teilweise ähnlich",
    },
    {
        "id":        "diag_06",
        "category":  "Zahlenunterschiede",
        "sentence1": "Das Produkt kostet 10 Euro.",
        "sentence2": "Das Produkt kostet 1000 Euro.",
        "expected":  "teilweise ähnlich",
    },

    # Eigennamen
    # Erwartung: Gleiche Rolle/Institution, andere Person → teilweise ähnlich.
    # SBERT könnte die strukturelle Ähnlichkeit überbewerten.
    {
        "id":        "diag_07",
        "category":  "Eigennamen",
        "sentence1": "Angela Merkel war Bundeskanzlerin von Deutschland.",
        "sentence2": "Friedrich Merz ist Bundeskanzler von Deutschland.",
        "expected":  "teilweise ähnlich",
    },
    {
        "id":        "diag_08",
        "category":  "Eigennamen",
        "sentence1": "Der FC Bayern München gewann den DFB-Pokal.",
        "sentence2": "VfB Stuttgart verlor das Finale.",
        "expected":  "teilweise ähnlich",
    },

    # Ähnliche Wortoberfläche, andere Bedeutung
    # Erwartung: Oberflächliche Ähnlichkeit täuscht – semantischer Inhalt ist verschieden.
    # Polyseme Verben wie 'ziehen' oder 'schlagen' erschweren die Disambiguierung.
    {
        "id":        "diag_09",
        "category":  "Ähnliche Wortoberfläche",
        "sentence1": "Er zieht nach Berlin um.",
        "sentence2": "Er zieht sich die Jacke an.",
        "expected":  "unähnlich",
    },
    {
        "id":        "diag_10",
        "category":  "Ähnliche Wortoberfläche",
        "sentence1": "Der Schüler schlägt das Buch auf.",
        "sentence2": "Der Schüler schläft im Unterricht.",
        "expected":  "unähnlich",
    },

    # Mehrdeutige Begriffe (Homonymie)
    # Erwartung: Gleicher Begriff, völlig verschiedene Bedeutungen → unähnlich.
    # Test, ob SBERT Disambiguierung durch Kontext leistet.
    {
        "id":        "diag_11",
        "category":  "Mehrdeutige Begriffe",
        "sentence1": "Die Bank steht direkt am Fluss.",
        "sentence2": "Die Bank hat heute wegen eines Feiertags geschlossen.",
        "expected":  "unähnlich",
    },
    {
        "id":        "diag_12",
        "category":  "Mehrdeutige Begriffe",
        "sentence1": "Das Schloss liegt auf einem Hügel.",
        "sentence2": "Das Schloss an der Tür ist kaputt.",
        "expected":  "unähnlich",
    },
]
