"""Optimization domain tools and encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def least_squares_2x2(a: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """Solve a 2x2 linear system A x = b (exact if invertible)."""
    if len(a) != 2 or any(len(row) != 2 for row in a):
        raise ValueError("a must be 2x2")
    if len(b) != 2:
        raise ValueError("b must have length 2")
    a11, a12 = a[0]
    a21, a22 = a[1]
    det = a11 * a22 - a12 * a21
    if det == 0:
        raise ValueError("matrix is singular")
    x1 = (b[0] * a22 - b[1] * a12) / det
    x2 = (a11 * b[1] - a21 * b[0]) / det
    return [x1, x2]


def gradient_step(x: Sequence[float], grad: Sequence[float], lr: float) -> List[float]:
    """Single gradient descent step."""
    if len(x) != len(grad):
        raise ValueError("x and grad must be same length")
    return [xi - lr * gi for xi, gi in zip(x, grad)]


def project_box(x: Sequence[float], lower: float, upper: float) -> List[float]:
    """Project vector into a box constraint [lower, upper]."""
    if lower > upper:
        raise ValueError("lower must be <= upper")
    return [min(max(xi, lower), upper) for xi in x]


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


MAX_VECTOR_LEN = 8
LENGTH_OFFSET = 0
STEP_SIZE_OFFSET = 1
X_NORM_OFFSET = 3
GRAD_NORM_OFFSET = 5
X_VALUES_OFFSET = 7
GRAD_VALUES_OFFSET = 23


@dataclass(frozen=True)
class OptimizationStep:
    """Optimization step representation for encoding."""

    x: List[float]
    grad: List[float]
    step_size: float
    length: int
    x_norm: float
    grad_norm: float


class OptimizationEncoder:
    """Encoder for gradient-based optimization steps."""

    domain_tag = DOMAIN_TAGS["opt_step"]
    domain_name = "opt_step"

    def encode(self, x: Sequence[float], grad: Sequence[float], step_size: float) -> Any:
        """Encode optimization step into an embedding."""
        x_vals = list(x)
        g_vals = list(grad)
        if len(x_vals) != len(g_vals):
            raise ValueError("x and grad must be same length")

        length = len(x_vals)
        x_norm = math.sqrt(sum(v * v for v in x_vals))
        g_norm = math.sqrt(sum(v * v for v in g_vals))

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        length_norm = min(length, MAX_VECTOR_LEN) / MAX_VECTOR_LEN
        emb = backend.at_add(emb, 16 + LENGTH_OFFSET, length_norm)

        step_sign, step_log = log_encode_value(step_size)
        emb = backend.at_add(emb, 16 + STEP_SIZE_OFFSET, step_sign)
        emb = backend.at_add(emb, 16 + STEP_SIZE_OFFSET + 1, step_log)

        x_sign, x_log = log_encode_value(x_norm)
        emb = backend.at_add(emb, 16 + X_NORM_OFFSET, x_sign)
        emb = backend.at_add(emb, 16 + X_NORM_OFFSET + 1, x_log)

        g_sign, g_log = log_encode_value(g_norm)
        emb = backend.at_add(emb, 16 + GRAD_NORM_OFFSET, g_sign)
        emb = backend.at_add(emb, 16 + GRAD_NORM_OFFSET + 1, g_log)

        for i, val in enumerate(x_vals[:MAX_VECTOR_LEN]):
            sign, log_mag = log_encode_value(float(val))
            offset = 16 + X_VALUES_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        for i, val in enumerate(g_vals[:MAX_VECTOR_LEN]):
            sign, log_mag = log_encode_value(float(val))
            offset = 16 + GRAD_VALUES_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        return emb

    def decode(self, emb: Any) -> OptimizationStep:
        """Decode an embedding back to an optimization step summary."""
        length = int(round(_scalar(emb[16 + LENGTH_OFFSET]) * MAX_VECTOR_LEN))
        step_size = log_decode_value(_scalar(emb[16 + STEP_SIZE_OFFSET]), _scalar(emb[16 + STEP_SIZE_OFFSET + 1]))
        x_norm = log_decode_value(_scalar(emb[16 + X_NORM_OFFSET]), _scalar(emb[16 + X_NORM_OFFSET + 1]))
        g_norm = log_decode_value(_scalar(emb[16 + GRAD_NORM_OFFSET]), _scalar(emb[16 + GRAD_NORM_OFFSET + 1]))

        x_vals: List[float] = []
        g_vals: List[float] = []
        count = min(length, MAX_VECTOR_LEN)
        for i in range(count):
            offset = 16 + X_VALUES_OFFSET + 2 * i
            x_val = log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1]))
            x_vals.append(x_val)

        for i in range(count):
            offset = 16 + GRAD_VALUES_OFFSET + 2 * i
            g_val = log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1]))
            g_vals.append(g_val)

        return OptimizationStep(
            x=x_vals,
            grad=g_vals,
            step_size=step_size,
            length=length,
            x_norm=x_norm,
            grad_norm=g_norm,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid optimization step."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "OptimizationStep",
    "OptimizationEncoder",
    "least_squares_2x2",
    "gradient_step",
    "project_box",
]
