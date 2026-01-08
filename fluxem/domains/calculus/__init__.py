"""Calculus domain tools and encoder for polynomials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def poly_derivative(coeffs: List[float]) -> List[float]:
    """Derivative of polynomial coefficients."""
    if not coeffs:
        raise ValueError("coeffs must be non-empty")
    if len(coeffs) == 1:
        return [0.0]
    return [i * coeffs[i] for i in range(1, len(coeffs))]


def poly_integral(coeffs: List[float], constant: float = 0.0) -> List[float]:
    """Indefinite integral of polynomial coefficients."""
    if not coeffs:
        raise ValueError("coeffs must be non-empty")
    integrated = [constant]
    for i, coef in enumerate(coeffs):
        integrated.append(coef / (i + 1))
    return integrated


def poly_evaluate(coeffs: List[float], x: float) -> float:
    """Evaluate polynomial at x."""
    if not coeffs:
        raise ValueError("coeffs must be non-empty")
    result = 0.0
    for coef in reversed(coeffs):
        result = result * x + coef
    return result


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


MAX_COEFFS = 16
DEGREE_OFFSET = 0
COEFF_OFFSET = 2
VALUE_AT_0_OFFSET = 34
VALUE_AT_1_OFFSET = 36
DERIV_AT_1_OFFSET = 38
INTEGRAL_0_1_OFFSET = 40
IS_ZERO_FLAG = 42


@dataclass(frozen=True)
class PolynomialFunction:
    """Polynomial function representation for calculus encoding."""

    coefficients: List[float]


class CalculusPolynomialEncoder:
    """Encoder for polynomial functions with derived calculus features."""

    domain_tag = DOMAIN_TAGS["calc_polynomial"]
    domain_name = "calc_polynomial"

    def encode(self, value: Sequence[float] | PolynomialFunction) -> Any:
        """Encode polynomial coefficients into an embedding."""
        if isinstance(value, PolynomialFunction):
            coefficients = list(value.coefficients)
        else:
            coefficients = list(value)

        if not coefficients:
            raise ValueError("coeffs must be non-empty")

        while len(coefficients) > 1 and abs(coefficients[-1]) < EPSILON:
            coefficients = coefficients[:-1]

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        if all(abs(c) < EPSILON for c in coefficients):
            emb = backend.at_add(emb, 16 + IS_ZERO_FLAG, 1.0)
            emb = backend.at_add(emb, 16 + DEGREE_OFFSET, 1.0)
            emb = backend.at_add(emb, 16 + DEGREE_OFFSET + 1, 0.0)
            return emb

        degree = len(coefficients) - 1
        emb = backend.at_add(emb, 16 + DEGREE_OFFSET, 1.0)
        emb = backend.at_add(emb, 16 + DEGREE_OFFSET + 1, math.log(degree + 1))

        for i, coeff in enumerate(coefficients[:MAX_COEFFS]):
            offset = 16 + COEFF_OFFSET + 2 * i
            sign, log_mag = log_encode_value(float(coeff))
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        value_at_0 = coefficients[0]
        value_at_1 = poly_evaluate(coefficients, 1.0)
        deriv_coeffs = poly_derivative(coefficients)
        deriv_at_1 = poly_evaluate(deriv_coeffs, 1.0)
        integral_coeffs = poly_integral(coefficients, constant=0.0)
        integral_0_1 = poly_evaluate(integral_coeffs, 1.0)

        for offset, value_at in (
            (VALUE_AT_0_OFFSET, value_at_0),
            (VALUE_AT_1_OFFSET, value_at_1),
            (DERIV_AT_1_OFFSET, deriv_at_1),
            (INTEGRAL_0_1_OFFSET, integral_0_1),
        ):
            sign, log_mag = log_encode_value(float(value_at))
            emb = backend.at_add(emb, 16 + offset, sign)
            emb = backend.at_add(emb, 16 + offset + 1, log_mag)

        return emb

    def decode(self, emb: Any) -> PolynomialFunction:
        """Decode an embedding back to polynomial coefficients."""
        if _scalar(emb[16 + IS_ZERO_FLAG]) > 0.5:
            return PolynomialFunction(coefficients=[0.0])

        log_deg = _scalar(emb[16 + DEGREE_OFFSET + 1])
        degree = int(round(math.exp(log_deg) - 1))
        degree = min(degree, MAX_COEFFS - 1)

        coeffs: List[float] = []
        for i in range(degree + 1):
            offset = 16 + COEFF_OFFSET + 2 * i
            coeff = log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1]))
            coeffs.append(coeff)

        return PolynomialFunction(coefficients=coeffs)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid calculus polynomial."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "PolynomialFunction",
    "CalculusPolynomialEncoder",
    "poly_derivative",
    "poly_integral",
    "poly_evaluate",
]
