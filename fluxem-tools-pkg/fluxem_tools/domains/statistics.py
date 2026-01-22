"""Statistics domain - mean, median, variance, correlation.

This module provides deterministic statistical computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def mean(values: List[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def median(values: List[float]) -> float:
    """Compute median (middle value)."""
    if not values:
        raise ValueError("Cannot compute median of empty list")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return sorted_vals[n // 2]


def variance(values: List[float], population: bool = False) -> float:
    """Compute variance (sample by default).

    Args:
        values: List of numbers
        population: If True, compute population variance; otherwise sample variance

    Returns:
        Variance as float
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values for variance")

    mu = mean(values)
    ss = sum((x - mu) ** 2 for x in values)
    n = len(values) if population else len(values) - 1
    return ss / n


def std(values: List[float], population: bool = False) -> float:
    """Compute standard deviation (sample by default)."""
    return math.sqrt(variance(values, population))


def corr(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y):
        raise ValueError("Lists must have same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 values")

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(a * b for a, b in zip(x, y))
    sum_x2 = sum(a ** 2 for a in x)
    sum_y2 = sum(b ** 2 for b in y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

    if denominator == 0:
        return 0.0
    return numerator / denominator


def percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100) using linear interpolation."""
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def mode(values: List[float]) -> float:
    """Compute mode (most frequent value). Returns first mode if multiple."""
    if not values:
        raise ValueError("Cannot compute mode of empty list")
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.keys(), key=lambda x: counts[x])


def z_score(value: float, values: List[float]) -> float:
    """Compute z-score of a value relative to a dataset."""
    mu = mean(values)
    sigma = std(values)
    if sigma == 0:
        return 0.0
    return (value - mu) / sigma


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_values(args) -> List[float]:
    if isinstance(args, dict):
        vals = args.get("values", args.get("data", args.get("nums")))
        if vals is not None:
            return [float(x) for x in vals]
        return [float(x) for x in list(args.values())[0]]
    if isinstance(args, (list, tuple)):
        return [float(x) for x in args]
    raise ValueError(f"Cannot parse values: {args}")


def _parse_two_lists(args) -> Tuple[List[float], List[float]]:
    if isinstance(args, dict):
        x = args.get("x", args.get("a"))
        y = args.get("y", args.get("b"))
        if x is not None and y is not None:
            return [float(v) for v in x], [float(v) for v in y]
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(v) for v in args[0]], [float(v) for v in args[1]]
    raise ValueError(f"Cannot parse two lists: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register statistics tools in the registry."""

    registry.register(ToolSpec(
        name="statistics_mean",
        function=lambda args: mean(_parse_values(args)),
        description="Computes arithmetic mean of a numeric list.",
        parameters={
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}, "description": "List of numbers"}
            },
            "required": ["values"]
        },
        returns="Mean as float",
        examples=[
            {"input": {"values": [1, 2, 3, 4, 5]}, "output": 3.0},
            {"input": {"values": [10, 20, 30]}, "output": 20.0},
        ],
        domain="statistics",
        tags=["mean", "average", "central tendency"],
    ))

    registry.register(ToolSpec(
        name="statistics_median",
        function=lambda args: median(_parse_values(args)),
        description="Computes median (middle value) of a numeric list.",
        parameters={
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}, "description": "List of numbers"}
            },
            "required": ["values"]
        },
        returns="Median as float",
        examples=[
            {"input": {"values": [1, 2, 3, 4, 5]}, "output": 3},
            {"input": {"values": [1, 2, 3, 4]}, "output": 2.5},
        ],
        domain="statistics",
        tags=["median", "central tendency"],
    ))

    registry.register(ToolSpec(
        name="statistics_variance",
        function=lambda args: variance(_parse_values(args)),
        description="Computes sample variance of a numeric list.",
        parameters={
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}, "description": "List of numbers"}
            },
            "required": ["values"]
        },
        returns="Variance as float",
        examples=[
            {"input": {"values": [2, 4, 4, 4, 5, 5, 7, 9]}, "output": 4.0},
        ],
        domain="statistics",
        tags=["variance", "dispersion", "spread"],
    ))

    registry.register(ToolSpec(
        name="statistics_std",
        function=lambda args: std(_parse_values(args)),
        description="Computes sample standard deviation of a numeric list.",
        parameters={
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}, "description": "List of numbers"}
            },
            "required": ["values"]
        },
        returns="Standard deviation as float",
        examples=[
            {"input": {"values": [2, 4, 4, 4, 5, 5, 7, 9]}, "output": 2.0},
        ],
        domain="statistics",
        tags=["std", "standard deviation", "dispersion"],
    ))

    registry.register(ToolSpec(
        name="statistics_corr",
        function=lambda args: corr(*_parse_two_lists(args)),
        description="Computes Pearson correlation coefficient between two lists.",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "array", "items": {"type": "number"}, "description": "First list"},
                "y": {"type": "array", "items": {"type": "number"}, "description": "Second list"},
            },
            "required": ["x", "y"]
        },
        returns="Correlation coefficient (-1 to 1)",
        examples=[
            {"input": {"x": [1, 2, 3], "y": [1, 2, 3]}, "output": 1.0},
            {"input": {"x": [1, 2, 3], "y": [3, 2, 1]}, "output": -1.0},
        ],
        domain="statistics",
        tags=["correlation", "pearson", "relationship"],
    ))
