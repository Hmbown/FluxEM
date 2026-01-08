"""Statistics domain tools and encoder."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, List, Sequence

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean of values."""
    if not values:
        raise ValueError("values must be non-empty")
    return statistics.fmean(values)


def median(values: Sequence[float]) -> float:
    """Median of values."""
    if not values:
        raise ValueError("values must be non-empty")
    return statistics.median(values)


def variance(values: Sequence[float], sample: bool = True) -> float:
    """Variance of values (sample by default)."""
    if len(values) < 2:
        raise ValueError("values must contain at least 2 elements")
    return statistics.variance(values) if sample else statistics.pvariance(values)


def std(values: Sequence[float], sample: bool = True) -> float:
    """Standard deviation of values (sample by default)."""
    if len(values) < 2:
        raise ValueError("values must contain at least 2 elements")
    return statistics.stdev(values) if sample else statistics.pstdev(values)


def corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation for two sequences."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must be same length >= 2")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x)
    den_y = sum((b - mean_y) ** 2 for b in y)
    if den_x == 0 or den_y == 0:
        raise ValueError("variance is zero; correlation undefined")
    return num / (den_x ** 0.5 * den_y ** 0.5)


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


COUNT_OFFSET = 0
MEAN_OFFSET = 2
MEDIAN_OFFSET = 4
VAR_OFFSET = 6
STD_OFFSET = 8
MIN_OFFSET = 10
MAX_OFFSET = 12
SUM_OFFSET = 14


@dataclass(frozen=True)
class StatsSummary:
    """Summary statistics for a numeric dataset."""

    count: int
    mean: float
    median: float
    variance: float
    std: float
    min_value: float
    max_value: float
    total: float


class StatisticsEncoder:
    """Encoder for summary statistics over numeric data."""

    domain_tag = DOMAIN_TAGS["stats_summary"]
    domain_name = "stats_summary"

    def encode(self, value: Sequence[float] | StatsSummary) -> Any:
        """Encode values or a summary into an embedding."""
        if isinstance(value, StatsSummary):
            summary = value
        else:
            values = list(value)
            if not values:
                raise ValueError("values must be non-empty")
            count = len(values)
            mean_val = statistics.fmean(values)
            median_val = statistics.median(values)
            min_val = min(values)
            max_val = max(values)
            total_val = sum(values)
            if count > 1:
                var_val = statistics.variance(values)
                std_val = math.sqrt(var_val)
            else:
                var_val = 0.0
                std_val = 0.0
            summary = StatsSummary(
                count=count,
                mean=mean_val,
                median=median_val,
                variance=var_val,
                std=std_val,
                min_value=min_val,
                max_value=max_val,
                total=total_val,
            )

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        count_sign, count_log = log_encode_value(float(summary.count))
        emb = backend.at_add(emb, 16 + COUNT_OFFSET, count_sign)
        emb = backend.at_add(emb, 16 + COUNT_OFFSET + 1, count_log)

        mean_sign, mean_log = log_encode_value(summary.mean)
        emb = backend.at_add(emb, 16 + MEAN_OFFSET, mean_sign)
        emb = backend.at_add(emb, 16 + MEAN_OFFSET + 1, mean_log)

        median_sign, median_log = log_encode_value(summary.median)
        emb = backend.at_add(emb, 16 + MEDIAN_OFFSET, median_sign)
        emb = backend.at_add(emb, 16 + MEDIAN_OFFSET + 1, median_log)

        var_sign, var_log = log_encode_value(summary.variance)
        emb = backend.at_add(emb, 16 + VAR_OFFSET, var_sign)
        emb = backend.at_add(emb, 16 + VAR_OFFSET + 1, var_log)

        std_sign, std_log = log_encode_value(summary.std)
        emb = backend.at_add(emb, 16 + STD_OFFSET, std_sign)
        emb = backend.at_add(emb, 16 + STD_OFFSET + 1, std_log)

        min_sign, min_log = log_encode_value(summary.min_value)
        emb = backend.at_add(emb, 16 + MIN_OFFSET, min_sign)
        emb = backend.at_add(emb, 16 + MIN_OFFSET + 1, min_log)

        max_sign, max_log = log_encode_value(summary.max_value)
        emb = backend.at_add(emb, 16 + MAX_OFFSET, max_sign)
        emb = backend.at_add(emb, 16 + MAX_OFFSET + 1, max_log)

        sum_sign, sum_log = log_encode_value(summary.total)
        emb = backend.at_add(emb, 16 + SUM_OFFSET, sum_sign)
        emb = backend.at_add(emb, 16 + SUM_OFFSET + 1, sum_log)

        return emb

    def decode(self, emb: Any) -> StatsSummary:
        """Decode an embedding back to summary statistics."""
        count = int(round(log_decode_value(_scalar(emb[16 + COUNT_OFFSET]), _scalar(emb[16 + COUNT_OFFSET + 1]))))
        mean_val = log_decode_value(_scalar(emb[16 + MEAN_OFFSET]), _scalar(emb[16 + MEAN_OFFSET + 1]))
        median_val = log_decode_value(_scalar(emb[16 + MEDIAN_OFFSET]), _scalar(emb[16 + MEDIAN_OFFSET + 1]))
        var_val = log_decode_value(_scalar(emb[16 + VAR_OFFSET]), _scalar(emb[16 + VAR_OFFSET + 1]))
        std_val = log_decode_value(_scalar(emb[16 + STD_OFFSET]), _scalar(emb[16 + STD_OFFSET + 1]))
        min_val = log_decode_value(_scalar(emb[16 + MIN_OFFSET]), _scalar(emb[16 + MIN_OFFSET + 1]))
        max_val = log_decode_value(_scalar(emb[16 + MAX_OFFSET]), _scalar(emb[16 + MAX_OFFSET + 1]))
        total_val = log_decode_value(_scalar(emb[16 + SUM_OFFSET]), _scalar(emb[16 + SUM_OFFSET + 1]))
        return StatsSummary(
            count=count,
            mean=mean_val,
            median=median_val,
            variance=var_val,
            std=std_val,
            min_value=min_val,
            max_value=max_val,
            total=total_val,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid statistics summary."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "StatsSummary",
    "StatisticsEncoder",
    "mean",
    "median",
    "variance",
    "std",
    "corr",
]
