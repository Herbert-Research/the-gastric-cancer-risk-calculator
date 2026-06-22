"""Bootstrap confidence interval utilities for model evaluation.

This module provides functions to compute bootstrap confidence intervals
for prediction metrics, enabling uncertainty quantification in model
performance evaluation.

Author: Maximilian Dressler
Purpose: Statistical rigor for PhD portfolio validation
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Note: signatures annotate arrays directly with ``np.ndarray`` (conceptually
# 1-D float64 arrays). A subscripted alias such as ``NDArray[np.float64]`` is
# avoided because the CI mypy run uses ``--ignore-missing-imports``, under which
# numpy resolves to ``Any`` and a module-level alias variable is rejected as a
# type.


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a prediction metric.

    Uses the percentile method to estimate confidence intervals by
    resampling with replacement from the original data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0/1 for binary outcomes).
    y_pred : np.ndarray
        Predicted probabilities.
    metric_fn : Callable[[np.ndarray, np.ndarray], float]
        Function that computes metric(y_true, y_pred) -> float.
    n_bootstrap : int, optional
        Number of bootstrap iterations (default: 1000).
    confidence_level : float, optional
        Confidence level for interval (default: 0.95 for 95% CI).
    random_state : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)

    Examples
    --------
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
    >>> estimate, ci_low, ci_high = bootstrap_metric(y_true, y_pred, brier_score_loss)
    >>> 0 <= ci_low <= estimate <= ci_high <= 1
    True

    Notes
    -----
    The percentile method is simple and robust but may have coverage issues
    for small samples. For clinical applications, consider using bias-corrected
    and accelerated (BCa) bootstrap intervals.

    References
    ----------
    Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall; 1993.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length. " f"Got {len(y_true)} and {len(y_pred)}."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot compute bootstrap on empty arrays.")

    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)

    # Point estimate on original data
    point_estimate = float(metric_fn(y_true, y_pred))

    # Bootstrap iterations
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Handle edge case where bootstrap sample has only one class
        try:
            estimate = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_estimates.append(estimate)
        except (ValueError, ZeroDivisionError):
            # Skip invalid bootstrap samples (e.g., all same class)
            continue

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        raise ValueError(
            f"Too many bootstrap samples failed ({n_bootstrap - len(bootstrap_estimates)}/{n_bootstrap}). "
            "Check if the metric function is appropriate for the data."
        )

    # Percentile method for CI
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def concordance_index(
    event_observed: np.ndarray,
    predicted_risk: np.ndarray,
) -> float:
    """
    Compute Harrell's concordance index (C-index) for binary outcomes.

    For each concordant pair (one event, one non-event), checks whether the
    predicted risk is higher for the patient who experienced the event.

    Parameters
    ----------
    event_observed : np.ndarray
        Binary event labels (0 = no event, 1 = event).
    predicted_risk : np.ndarray
        Predicted risk scores (higher = worse prognosis).

    Returns
    -------
    float
        C-index in [0, 1]. 0.5 = random, 1.0 = perfect discrimination.

    Notes
    -----
    Complexity is O(n_events * n_non_events), i.e. up to O(n^2) in the cohort
    size, because every event/non-event pair is compared. This is acceptable
    for cohort sizes on the order of 10^3 (e.g. the TCGA STAD cohort, n=436),
    but should be vectorized or replaced with a rank-based estimator before use
    on substantially larger datasets.
    """
    event_observed = np.asarray(event_observed, dtype=float)
    predicted_risk = np.asarray(predicted_risk, dtype=float)

    valid = ~(np.isnan(event_observed) | np.isnan(predicted_risk))
    event_observed = event_observed[valid]
    predicted_risk = predicted_risk[valid]

    concordant = 0
    discordant = 0
    tied = 0

    event_idx = np.where(event_observed == 1)[0]
    nonevent_idx = np.where(event_observed == 0)[0]

    for i in event_idx:
        for j in nonevent_idx:
            if predicted_risk[i] > predicted_risk[j]:
                concordant += 1
            elif predicted_risk[i] < predicted_risk[j]:
                discordant += 1
            else:
                tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for Pearson correlation.

    Parameters
    ----------
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    n_bootstrap : int, optional
        Number of bootstrap iterations (default: 1000).
    confidence_level : float, optional
        Confidence level for interval (default: 0.95).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float]
        (correlation, ci_lower, ci_upper)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Remove NaN pairs
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 3:
        raise ValueError("Need at least 3 valid pairs to compute correlation.")

    def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    return bootstrap_metric(
        x_valid,
        y_valid,
        pearson_r,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
