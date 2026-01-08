"""Information theory domain tools and encoder."""

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


def _validate_probs(values: Sequence[float]) -> None:
    if not values:
        raise ValueError("values must be non-empty")
    if any(v < 0 for v in values):
        raise ValueError("probabilities must be non-negative")


def entropy(probs: Sequence[float], base: float = 2.0) -> float:
    """Shannon entropy of a probability distribution."""
    _validate_probs(probs)
    total = sum(probs)
    if total <= 0:
        raise ValueError("sum of probabilities must be > 0")
    log_base = math.log(base)
    ent = 0.0
    for p in probs:
        if p <= 0:
            continue
        p_norm = p / total
        ent -= p_norm * (math.log(p_norm) / log_base)
    return ent


def cross_entropy(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Cross-entropy H(p, q)."""
    _validate_probs(p)
    _validate_probs(q)
    if len(p) != len(q):
        raise ValueError("p and q must be same length")
    log_base = math.log(base)
    total_p = sum(p)
    total_q = sum(q)
    if total_p <= 0 or total_q <= 0:
        raise ValueError("probability sums must be > 0")
    ce = 0.0
    for p_i, q_i in zip(p, q):
        if p_i <= 0:
            continue
        p_norm = p_i / total_p
        q_norm = max(q_i / total_q, 1e-12)
        ce -= p_norm * (math.log(q_norm) / log_base)
    return ce


def kl_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """KL divergence D_KL(p || q)."""
    return cross_entropy(p, q, base=base) - entropy(p, base=base)


MAX_PROBS = 16
COUNT_OFFSET = 0
PROBS_OFFSET = 1
ENTROPY_OFFSET = 33
CROSS_ENTROPY_OFFSET = 35
KL_OFFSET = 37
BASE_OFFSET = 39
HAS_REFERENCE_FLAG = 41


@dataclass(frozen=True)
class InformationDistribution:
    """Distribution and information measures."""

    probs: List[float]
    base: float = 2.0
    entropy: Optional[float] = None
    cross_entropy: Optional[float] = None
    kl_divergence: Optional[float] = None


def _normalize(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("probability sum must be > 0")
    return [v / total for v in values]


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


class InformationTheoryEncoder:
    """Encoder for probability distributions with information metrics."""

    domain_tag = DOMAIN_TAGS["info_entropy"]
    domain_name = "info_entropy"

    def encode(
        self,
        value: InformationDistribution | Sequence[float],
        reference: Optional[Sequence[float]] = None,
    ) -> Any:
        """Encode probabilities and derived metrics into an embedding."""
        if isinstance(value, InformationDistribution):
            probs = value.probs
            base = value.base
            reference = reference if reference is not None else None
        else:
            probs = list(value)
            base = 2.0

        probs = _normalize(probs)

        has_reference = reference is not None
        if has_reference:
            reference_probs = _normalize(reference)
            ce_val = cross_entropy(probs, reference_probs, base=base)
            kl_val = kl_divergence(probs, reference_probs, base=base)
        else:
            ce_val = 0.0
            kl_val = 0.0

        ent_val = entropy(probs, base=base)

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        count_norm = min(len(probs), MAX_PROBS) / MAX_PROBS
        emb = backend.at_add(emb, 16 + COUNT_OFFSET, count_norm)

        for i, prob in enumerate(probs[:MAX_PROBS]):
            sign, log_mag = log_encode_value(float(prob))
            offset = 16 + PROBS_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        ent_sign, ent_log = log_encode_value(float(ent_val))
        emb = backend.at_add(emb, 16 + ENTROPY_OFFSET, ent_sign)
        emb = backend.at_add(emb, 16 + ENTROPY_OFFSET + 1, ent_log)

        ce_sign, ce_log = log_encode_value(float(ce_val))
        emb = backend.at_add(emb, 16 + CROSS_ENTROPY_OFFSET, ce_sign)
        emb = backend.at_add(emb, 16 + CROSS_ENTROPY_OFFSET + 1, ce_log)

        kl_sign, kl_log = log_encode_value(float(kl_val))
        emb = backend.at_add(emb, 16 + KL_OFFSET, kl_sign)
        emb = backend.at_add(emb, 16 + KL_OFFSET + 1, kl_log)

        base_sign, base_log = log_encode_value(float(base))
        emb = backend.at_add(emb, 16 + BASE_OFFSET, base_sign)
        emb = backend.at_add(emb, 16 + BASE_OFFSET + 1, base_log)

        emb = backend.at_add(emb, 16 + HAS_REFERENCE_FLAG, 1.0 if has_reference else 0.0)

        return emb

    def decode(self, emb: Any) -> InformationDistribution:
        """Decode an embedding back to a distribution summary."""
        count = int(round(_scalar(emb[16 + COUNT_OFFSET]) * MAX_PROBS))
        probs: List[float] = []
        for i in range(max(count, 0)):
            offset = 16 + PROBS_OFFSET + 2 * i
            prob = log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1]))
            probs.append(max(prob, 0.0))
        if probs:
            probs = _normalize(probs)

        entropy_val = log_decode_value(_scalar(emb[16 + ENTROPY_OFFSET]), _scalar(emb[16 + ENTROPY_OFFSET + 1]))
        cross_val = log_decode_value(
            _scalar(emb[16 + CROSS_ENTROPY_OFFSET]),
            _scalar(emb[16 + CROSS_ENTROPY_OFFSET + 1]),
        )
        kl_val = log_decode_value(_scalar(emb[16 + KL_OFFSET]), _scalar(emb[16 + KL_OFFSET + 1]))
        base_val = log_decode_value(_scalar(emb[16 + BASE_OFFSET]), _scalar(emb[16 + BASE_OFFSET + 1]))

        return InformationDistribution(
            probs=probs,
            base=base_val if base_val > 0 else 2.0,
            entropy=entropy_val,
            cross_entropy=cross_val,
            kl_divergence=kl_val,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid information theory encoding."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "InformationDistribution",
    "InformationTheoryEncoder",
    "entropy",
    "cross_entropy",
    "kl_divergence",
]
