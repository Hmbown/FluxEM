"""Evaluation metrics for arithmetic benchmark."""

from typing import Dict, List

import numpy as np


def compute_metrics(
    predictions: List[float],
    ground_truths: List[float],
    tolerance: float = 0.01,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        ground_truths: Ground truth values
        tolerance: Relative error tolerance (default 1%)

    Returns:
        Dict with mae, relative_error, accuracy metrics
    """
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Absolute errors
    abs_errors = np.abs(predictions - ground_truths)

    # Relative errors
    rel_errors = abs_errors / (np.abs(ground_truths) + 1e-10)

    # For near-zero values, use absolute tolerance
    near_zero = np.abs(ground_truths) < 1.0
    within_tol_rel = rel_errors < tolerance
    within_tol_abs = abs_errors < 0.5
    within_tol = np.where(near_zero, within_tol_abs, within_tol_rel)

    return {
        "mae": float(np.mean(abs_errors)),
        "median_ae": float(np.median(abs_errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "accuracy": float(np.mean(within_tol)),
        "max_relative_error": float(np.max(rel_errors)),
        "n_samples": len(predictions),
    }


def compute_metrics_by_num_ops(
    predictions: List[float],
    ground_truths: List[float],
    num_ops: List[int],
    tolerance: float = 0.01,
) -> Dict[int, Dict[str, float]]:
    """Compute metrics stratified by number of operations."""
    results = {}
    unique_ops = sorted(set(num_ops))

    for n_ops in unique_ops:
        mask = [i for i, n in enumerate(num_ops) if n == n_ops]
        if mask:
            preds = [predictions[i] for i in mask]
            gts = [ground_truths[i] for i in mask]
            results[n_ops] = compute_metrics(preds, gts, tolerance)

    return results
