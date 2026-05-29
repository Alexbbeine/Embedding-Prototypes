"""
evaluation.py
=============
Auswertungsmetriken für den Vergleich von modellberechneten Kosinusähnlichkeiten mit menschlich annotierten Gold-Scores.

Verwendete Metriken
-------------------
Pearson-Korrelation
    Misst den linearen Zusammenhang zwischen Vorhersage und Gold-Score.
    Sensibel gegenüber Ausreißern; Werte nahe ±1 zeigen starke lineare Bindung.

Spearman-Korrelation
    Misst den monotonen Rangzusammenhang. Robuster als Pearson, da nur die relative Reihenfolge der Werte betrachtet wird, nicht ihre absoluten Abstände.

Mean Absolute Error (MAE)
    Durchschnittliche absolute Abweichung auf der normierten Skala [0, 1].
    Direkt interpretierbar: MAE = 0.16 bedeutet im Schnitt 16 Prozentpunkte Abweichung vom menschlichen Urteil.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr       # type: ignore
from sklearn.metrics import mean_absolute_error   # type: ignore


def compute_metrics(
    predicted: list[float] | np.ndarray,
    gold: list[float] | np.ndarray,
) -> dict[str, float | int]:
    """
    Berechnet Pearson-Korrelation, Spearman-Korrelation und MAE.

    Args:
        predicted: Vom Modell berechnete Kosinusähnlichkeiten (Skala 0–1).
        gold:      Normalisierte Gold-Scores aus dem Datensatz (Skala 0–1).

    Returns:
        Dictionary mit den Schlüsseln:

        ``pearson_r``
            Pearson-Korrelationskoeffizient (−1 bis 1).
        ``pearson_p``
            Zweiseitiger p-Wert der Pearson-Korrelation.
        ``spearman_r``
            Spearman-Rangkorrelationskoeffizient (−1 bis 1).
        ``spearman_p``
            Zweiseitiger p-Wert der Spearman-Korrelation.
        ``mae``
            Mean Absolute Error auf der normierten Skala [0, 1].
        ``n_samples``
            Anzahl ausgewerteter Satzpaare.
    """
    pred_arr = np.array(predicted, dtype=float)
    gold_arr = np.array(gold,      dtype=float)

    pearson_r,  pearson_p  = pearsonr(pred_arr,  gold_arr)
    spearman_r, spearman_p = spearmanr(pred_arr, gold_arr)
    mae = mean_absolute_error(gold_arr, pred_arr)

    return {
        "pearson_r":  round(float(pearson_r),  4),
        "pearson_p":  round(float(pearson_p),  6),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 6),
        "mae":        round(float(mae),        4),
        "n_samples":  int(len(predicted)),
    }
