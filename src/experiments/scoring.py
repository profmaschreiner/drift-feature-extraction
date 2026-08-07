"""
Implementation of Algorithm 1 (Individual Scoring of Drift Detector Alarms)
from Section 4 of the paper.

This module isolates the pure, per-feature scoring function from the rest of
the experimental pipeline (detector execution, caching, I/O). It takes the
positions of drift alarms already detected for a single feature and produces
the sequentially accumulated score S_j described by the recurrence:

    phi_j = phi_b * ln(1 + w_j)
    beta_j = 1 + 1 / ln(1 + w_j)
    S_j = (1 - phi_j) * S_j + beta_j * a_j

where w_j is the number of consecutive instances since the last alarm on
that feature, and phi_b is the only adjustable hyperparameter of the
methodology.

Note on architecture: drift alarm positions are detected once per
feature/detector/fold/normalization (row-major, one online detector pass per column, as in
the pseudocode of Algorithm 1) and cached to disk, since they do not depend
on phi_b. This function is then re-applied per feature over the cached
positions for each phi_b value tried during hyperparameter search, avoiding
re-running the online detectors at every Optuna trial. Because each feature's
recurrence (S_j, w_j, phi_j, beta_j) is fully independent of every other
feature, applying this function column-by-column over cached alarm positions
is mathematically equivalent to the row-major, per-instance loop in the
paper's Algorithm 1.

"""

import numpy as np
import math


def gerar_scores_a_partir_de_posicoes(
    num_rows: int,
    drift_indices: np.ndarray,
    phi_b: float,
) -> np.ndarray:
    """
    Compute the sequentially accumulated drift-score S_j for a single
    feature, given the row indices at which a drift alarm was triggered.

    Parameters
    ----------
    num_rows : int
        Number of instances (rows) in the sequence.
    drift_indices : np.ndarray
        1D array of row indices where a drift alarm was detected for this
        feature. Indices outside [0, num_rows) are ignored.
    phi_b : float
        Memory hyperparameter of the methodology (Section 4), sampled on a
        log scale over [1e-6, 1e-1] during hyperparameter search
        (Table S-I).

    Returns
    -------
    np.ndarray
        Array of shape (num_rows,) with the score S_j at each instance.
    """
    scores = np.zeros(num_rows, dtype=np.float64)
    sem_drift = 1.0
    current_score = 0.0

    flags = np.zeros(num_rows, dtype=bool)
    if len(drift_indices) > 0:
        drift_indices = drift_indices[(drift_indices >= 0) & (drift_indices < num_rows)]
        flags[drift_indices] = True

    for linha in range(num_rows):
        flag = flags[linha]
        if flag:
            phi = phi_b * math.log(1 + sem_drift)
            if phi > 1:
                phi = 1
            score_ant = scores[linha - 1] if linha > 1 else 0.0
            current_score = (1 + 1 / math.log(1 + sem_drift)) + (1 - phi) * score_ant
            sem_drift = 0
        else:
            sem_drift += 1
            phi = phi_b * math.log(1 + sem_drift)
            if phi > 1:
                phi = 1
            score_ant = scores[linha - 1] if linha > 1 else 0.0
            current_score = (1 - phi) * score_ant
        scores[linha] = current_score
    return scores
