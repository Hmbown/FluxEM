"""Combinatorics domain tools and encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def factorial(n: int) -> int:
    """Return n! for n >= 0."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return math.factorial(n)


def ncr(n: int, k: int) -> int:
    """Return combinations count C(n, k)."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be >= 0")
    if k > n:
        return 0
    return math.comb(n, k)


def npr(n: int, k: int) -> int:
    """Return permutations count P(n, k)."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be >= 0")
    if k > n:
        return 0
    try:
        return math.perm(n, k)
    except AttributeError:
        return math.factorial(n) // math.factorial(n - k)


def multiset_combinations(n: int, k: int) -> int:
    """Return combinations with repetition: C(n+k-1, k)."""
    if n <= 0 or k < 0:
        raise ValueError("n must be > 0 and k must be >= 0")
    return math.comb(n + k - 1, k)


KIND_ORDER = ("factorial", "ncr", "npr", "multiset")
KIND_OFFSET = 0
N_OFFSET = 4
K_OFFSET = 6
VALUE_OFFSET = 8
HAS_K_FLAG = 10


@dataclass(frozen=True)
class CombinatorialTerm:
    """Structured combinatorics term for encoding."""

    kind: str
    n: int
    k: Optional[int] = None
    value: Optional[int] = None

    def with_value(self) -> "CombinatorialTerm":
        """Return a copy with computed value filled in."""
        if self.value is not None:
            return self
        return CombinatorialTerm(
            kind=self.kind,
            n=self.n,
            k=self.k,
            value=_compute_value(self.kind, self.n, self.k),
        )


def _compute_value(kind: str, n: int, k: Optional[int]) -> int:
    if kind == "factorial":
        return factorial(n)
    if k is None:
        raise ValueError("k is required for this combinatorics term")
    if kind == "ncr":
        return ncr(n, k)
    if kind == "npr":
        return npr(n, k)
    if kind == "multiset":
        return multiset_combinations(n, k)
    raise ValueError(f"Unknown combinatorics kind: {kind}")


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


class CombinatoricsEncoder:
    """Encoder for combinatorial terms (factorial, nCr, nPr, multiset)."""

    domain_tag = DOMAIN_TAGS["comb_term"]
    domain_name = "comb_term"

    def encode(self, value: CombinatorialTerm | Tuple[str, int, Optional[int]]) -> Any:
        """Encode a combinatorial term into an embedding."""
        if isinstance(value, tuple):
            kind, n, k = value if len(value) == 3 else (value[0], value[1], None)
            term = CombinatorialTerm(kind=kind, n=int(n), k=None if k is None else int(k))
        elif isinstance(value, CombinatorialTerm):
            term = value
        else:
            raise ValueError("Unsupported combinatorial term input")

        if term.kind not in KIND_ORDER:
            raise ValueError(f"Unknown combinatorics kind: {term.kind}")

        term = term.with_value()

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        kind_index = KIND_ORDER.index(term.kind)
        emb = backend.at_add(emb, 16 + KIND_OFFSET + kind_index, 1.0)

        n_sign, n_log = log_encode_value(float(term.n))
        emb = backend.at_add(emb, 16 + N_OFFSET, n_sign)
        emb = backend.at_add(emb, 16 + N_OFFSET + 1, n_log)

        k_value = term.k if term.k is not None else 0
        k_sign, k_log = log_encode_value(float(k_value))
        emb = backend.at_add(emb, 16 + K_OFFSET, k_sign)
        emb = backend.at_add(emb, 16 + K_OFFSET + 1, k_log)
        emb = backend.at_add(emb, 16 + HAS_K_FLAG, 1.0 if term.k is not None else 0.0)

        value_sign, value_log = log_encode_value(float(term.value or 0))
        emb = backend.at_add(emb, 16 + VALUE_OFFSET, value_sign)
        emb = backend.at_add(emb, 16 + VALUE_OFFSET + 1, value_log)

        return emb

    def decode(self, emb: Any) -> CombinatorialTerm:
        """Decode an embedding back to a combinatorial term."""
        kind_values = [_scalar(emb[16 + KIND_OFFSET + i]) for i in range(len(KIND_ORDER))]
        kind_index = max(range(len(kind_values)), key=lambda i: kind_values[i])
        kind = KIND_ORDER[kind_index]

        n = int(round(log_decode_value(_scalar(emb[16 + N_OFFSET]), _scalar(emb[16 + N_OFFSET + 1]))))
        k_raw = log_decode_value(_scalar(emb[16 + K_OFFSET]), _scalar(emb[16 + K_OFFSET + 1]))
        has_k = _scalar(emb[16 + HAS_K_FLAG]) > 0.5
        k = int(round(k_raw)) if has_k else None

        value = int(round(log_decode_value(_scalar(emb[16 + VALUE_OFFSET]), _scalar(emb[16 + VALUE_OFFSET + 1]))))
        return CombinatorialTerm(kind=kind, n=n, k=k, value=value)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid combinatorics term."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "CombinatorialTerm",
    "CombinatoricsEncoder",
    "factorial",
    "ncr",
    "npr",
    "multiset_combinations",
]
