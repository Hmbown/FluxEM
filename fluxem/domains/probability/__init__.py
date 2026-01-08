"""Probability domain tools and encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def _validate_prob(p: float) -> None:
    if p < 0.0 or p > 1.0:
        raise ValueError("Probability must be between 0 and 1")


def bernoulli_pmf(p: float, x: int) -> float:
    """Bernoulli PMF for x in {0,1}."""
    _validate_prob(p)
    if x not in (0, 1):
        raise ValueError("x must be 0 or 1")
    return p if x == 1 else 1.0 - p


def binomial_pmf(n: int, k: int, p: float) -> float:
    """Binomial PMF for n trials and k successes."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be >= 0")
    if k > n:
        return 0.0
    _validate_prob(p)
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def bayes_rule(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
    """Compute P(A|B) from P(A), P(B|A), P(B|not A)."""
    _validate_prob(p_a)
    _validate_prob(p_b_given_a)
    _validate_prob(p_b_given_not_a)
    denom = p_b_given_a * p_a + p_b_given_not_a * (1.0 - p_a)
    if denom == 0:
        raise ValueError("Denominator is zero; check probabilities")
    return (p_b_given_a * p_a) / denom


def normalize_probabilities(values: Sequence[float]) -> list[float]:
    """Normalize a list of non-negative values into probabilities."""
    total = sum(values)
    if total <= 0:
        raise ValueError("Sum of values must be > 0")
    return [v / total for v in values]


MAX_PROBS = 16
KIND_ORDER = ("bernoulli", "binomial", "categorical")
KIND_OFFSET = 0
P_OFFSET = 3
N_OFFSET = 5
COUNT_OFFSET = 7
PROBS_OFFSET = 8
MEAN_OFFSET = 40
VAR_OFFSET = 42
ENTROPY_OFFSET = 44


@dataclass(frozen=True)
class ProbabilityDistribution:
    """Structured distribution for encoding."""

    kind: str
    p: Optional[float] = None
    n: Optional[int] = None
    probs: Optional[List[float]] = None


def _normalize(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("probabilities must sum to > 0")
    return [v / total for v in values]


def _entropy(probs: Sequence[float]) -> float:
    total = 0.0
    for p in probs:
        if p <= 0:
            continue
        total -= p * (math.log(p) / math.log(2.0))
    return total


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


class ProbabilityEncoder:
    """Encoder for basic probability distributions."""

    domain_tag = DOMAIN_TAGS["prob_dist"]
    domain_name = "prob_dist"

    def encode(self, value: ProbabilityDistribution | Sequence[float]) -> Any:
        """Encode a distribution into an embedding."""
        if isinstance(value, ProbabilityDistribution):
            dist = value
        elif isinstance(value, (list, tuple)):
            dist = ProbabilityDistribution(kind="categorical", probs=list(value))
        else:
            raise ValueError("Unsupported distribution input")

        if dist.kind not in KIND_ORDER:
            raise ValueError(f"Unknown distribution kind: {dist.kind}")

        probs: List[float]
        p_value: float = 0.0
        n_value: int = 0

        if dist.kind == "bernoulli":
            if dist.p is None:
                raise ValueError("p is required for bernoulli distribution")
            _validate_prob(dist.p)
            p_value = dist.p
            probs = [1.0 - dist.p, dist.p]
        elif dist.kind == "binomial":
            if dist.p is None or dist.n is None:
                raise ValueError("p and n are required for binomial distribution")
            _validate_prob(dist.p)
            if dist.n < 0:
                raise ValueError("n must be >= 0")
            p_value = dist.p
            n_value = dist.n
            probs = [binomial_pmf(dist.n, k, dist.p) for k in range(dist.n + 1)]
        else:
            if dist.probs is None:
                raise ValueError("probs are required for categorical distribution")
            probs = _normalize(dist.probs)

        probs = _normalize(probs)

        mean_index = sum(i * p for i, p in enumerate(probs))
        var_index = sum(((i - mean_index) ** 2) * p for i, p in enumerate(probs))
        entropy_value = _entropy(probs)

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        kind_index = KIND_ORDER.index(dist.kind)
        emb = backend.at_add(emb, 16 + KIND_OFFSET + kind_index, 1.0)

        p_sign, p_log = log_encode_value(float(p_value))
        emb = backend.at_add(emb, 16 + P_OFFSET, p_sign)
        emb = backend.at_add(emb, 16 + P_OFFSET + 1, p_log)

        n_sign, n_log = log_encode_value(float(n_value))
        emb = backend.at_add(emb, 16 + N_OFFSET, n_sign)
        emb = backend.at_add(emb, 16 + N_OFFSET + 1, n_log)

        count_norm = min(len(probs), MAX_PROBS) / MAX_PROBS
        emb = backend.at_add(emb, 16 + COUNT_OFFSET, count_norm)

        for i, prob in enumerate(probs[:MAX_PROBS]):
            sign, log_mag = log_encode_value(float(prob))
            offset = 16 + PROBS_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        mean_sign, mean_log = log_encode_value(float(mean_index))
        emb = backend.at_add(emb, 16 + MEAN_OFFSET, mean_sign)
        emb = backend.at_add(emb, 16 + MEAN_OFFSET + 1, mean_log)

        var_sign, var_log = log_encode_value(float(var_index))
        emb = backend.at_add(emb, 16 + VAR_OFFSET, var_sign)
        emb = backend.at_add(emb, 16 + VAR_OFFSET + 1, var_log)

        ent_sign, ent_log = log_encode_value(float(entropy_value))
        emb = backend.at_add(emb, 16 + ENTROPY_OFFSET, ent_sign)
        emb = backend.at_add(emb, 16 + ENTROPY_OFFSET + 1, ent_log)

        return emb

    def decode(self, emb: Any) -> ProbabilityDistribution:
        """Decode an embedding back to a distribution summary."""
        kind_values = [_scalar(emb[16 + KIND_OFFSET + i]) for i in range(len(KIND_ORDER))]
        kind_index = max(range(len(kind_values)), key=lambda i: kind_values[i])
        kind = KIND_ORDER[kind_index]

        p_value = log_decode_value(_scalar(emb[16 + P_OFFSET]), _scalar(emb[16 + P_OFFSET + 1]))
        n_value = int(round(log_decode_value(_scalar(emb[16 + N_OFFSET]), _scalar(emb[16 + N_OFFSET + 1]))))

        count = int(round(_scalar(emb[16 + COUNT_OFFSET]) * MAX_PROBS))
        probs: List[float] = []
        for i in range(max(count, 0)):
            offset = 16 + PROBS_OFFSET + 2 * i
            prob = log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1]))
            probs.append(max(prob, 0.0))

        if probs:
            probs = _normalize(probs)

        if kind == "bernoulli":
            p_value = probs[1] if len(probs) > 1 else max(min(p_value, 1.0), 0.0)
            return ProbabilityDistribution(kind=kind, p=p_value, probs=[1.0 - p_value, p_value])
        if kind == "binomial":
            return ProbabilityDistribution(kind=kind, p=p_value, n=n_value, probs=probs or None)
        return ProbabilityDistribution(kind=kind, probs=probs or None)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid probability distribution."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "ProbabilityDistribution",
    "ProbabilityEncoder",
    "bernoulli_pmf",
    "binomial_pmf",
    "bayes_rule",
    "normalize_probabilities",
]
