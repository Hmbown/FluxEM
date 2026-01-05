"""
Polynomial Encoder.

Embeds polynomials by their coefficient vectors.
Addition is add corresponding coefficients.
Scalar multiplication is multiply all coefficients.
Polynomial multiplication requires convolution (decode-operate-encode).
"""

import math
from typing import Any, List, Optional, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
)


# Embedding layout:
# dims 0-1:   Degree (1, log(deg+1))
# dims 2-31:  Coefficients a_0 to a_14 (2 dims each: sign, log|a|+1)
# dims 32-33: Leading coefficient (sign, log|a_n|)
# dims 34:    Is zero polynomial flag
# dims 35:    Is monic flag (leading coeff = 1)
# dims 36-47: Reserved for derived properties
# dims 48-63: Reserved

DEGREE_OFFSET = 0
COEFF_OFFSET = 2
MAX_COEFFS = 15  # Can store degree up to 14
LEADING_COEFF_OFFSET = 32
IS_ZERO_FLAG = 34
IS_MONIC_FLAG = 35


class PolynomialEncoder:
    """
    Encoder for polynomials over the reals.

    Stores coefficient vector with log-magnitude encoding.
    Addition is performed in embedding space.
    """

    domain_tag = DOMAIN_TAGS["math_polynomial"]
    domain_name = "math_polynomial"

    def encode(self, coefficients: Union[List[float], 'Polynomial']) -> Any:
        """
        Encode a polynomial by its coefficients.

        Args:
            coefficients: List [a_0, a_1, ..., a_n] where p(x) = sum(a_i * x^i)

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if hasattr(coefficients, 'coefficients'):
            coefficients = coefficients.coefficients

        # Remove trailing zeros
        while len(coefficients) > 1 and abs(coefficients[-1]) < EPSILON:
            coefficients = coefficients[:-1]

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Check for zero polynomial
        if len(coefficients) == 0 or all(abs(c) < EPSILON for c in coefficients):
            emb = backend.at_add(emb, 8 + IS_ZERO_FLAG, 1.0)
            emb = backend.at_add(emb, 8 + DEGREE_OFFSET, 1.0)
            emb = backend.at_add(emb, 8 + DEGREE_OFFSET + 1, 0.0)  # log(0+1) = 0
            return emb

        degree = len(coefficients) - 1

        # Degree
        emb = backend.at_add(emb, 8 + DEGREE_OFFSET, 1.0)
        emb = backend.at_add(emb, 8 + DEGREE_OFFSET + 1, math.log(degree + 1))

        # Coefficients (up to degree 14)
        for i, c in enumerate(coefficients[:MAX_COEFFS]):
            offset = COEFF_OFFSET + i * 2

            if abs(c) < EPSILON:
                emb = backend.at_add(emb, 8 + offset, 0.0)
                emb = backend.at_add(emb, 8 + offset + 1, 0.0)
            else:
                emb = backend.at_add(emb, 8 + offset, 1.0 if c > 0 else -1.0)
                emb = backend.at_add(emb, 8 + offset + 1, math.log(abs(c) + 1))

        # Leading coefficient
        leading = coefficients[-1] if coefficients else 0
        if abs(leading) >= EPSILON:
            emb = backend.at_add(emb, 8 + LEADING_COEFF_OFFSET, 1.0 if leading > 0 else -1.0)
            emb = backend.at_add(emb, 8 + LEADING_COEFF_OFFSET + 1, math.log(abs(leading)))

            # Is monic?
            emb = backend.at_add(emb, 8 + IS_MONIC_FLAG, 1.0 if abs(leading - 1.0) < EPSILON else 0.0)

        return emb

    def decode(self, emb: Any) -> List[float]:
        """
        Decode embedding to coefficient list.
        """
        if emb[8 + IS_ZERO_FLAG].item() > 0.5:
            return [0.0]

        # Get degree
        log_deg = emb[8 + DEGREE_OFFSET + 1].item()
        degree = int(round(math.exp(log_deg) - 1))
        degree = min(degree, MAX_COEFFS - 1)

        coefficients = []
        for i in range(degree + 1):
            offset = COEFF_OFFSET + i * 2

            sign = emb[8 + offset].item()
            if abs(sign) < 0.5:
                coefficients.append(0.0)
            else:
                log_c = emb[8 + offset + 1].item()
                c = math.exp(log_c) - 1
                coefficients.append(sign * c if sign > 0 else -c)

        return coefficients

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid polynomial."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Operations
    # =========================================================================

    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two polynomials.

        Coefficient-wise addition.
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Get degrees
        deg1 = int(round(math.exp(emb1[8 + DEGREE_OFFSET + 1].item()) - 1))
        deg2 = int(round(math.exp(emb2[8 + DEGREE_OFFSET + 1].item()) - 1))
        max_deg = max(deg1, deg2)

        # Add coefficients
        coefficients = []
        for i in range(min(max_deg + 1, MAX_COEFFS)):
            offset = COEFF_OFFSET + i * 2

            c1 = 0.0
            c2 = 0.0

            # Get coefficient from first polynomial
            if i <= deg1:
                sign1 = emb1[8 + offset].item()
                if abs(sign1) >= 0.5:
                    log_c1 = emb1[8 + offset + 1].item()
                    c1 = (1 if sign1 > 0 else -1) * (math.exp(log_c1) - 1)

            # Get coefficient from second polynomial
            if i <= deg2:
                sign2 = emb2[8 + offset].item()
                if abs(sign2) >= 0.5:
                    log_c2 = emb2[8 + offset + 1].item()
                    c2 = (1 if sign2 > 0 else -1) * (math.exp(log_c2) - 1)

            c_sum = c1 + c2
            coefficients.append(c_sum)

            if abs(c_sum) < EPSILON:
                result = backend.at_add(result, 8 + offset, 0.0)
                result = backend.at_add(result, 8 + offset + 1, 0.0)
            else:
                result = backend.at_add(result, 8 + offset, 1.0 if c_sum > 0 else -1.0)
                result = backend.at_add(result, 8 + offset + 1, math.log(abs(c_sum) + 1))

        # Update degree (find actual degree after addition)
        actual_deg = 0
        for i in range(len(coefficients) - 1, -1, -1):
            if abs(coefficients[i]) >= EPSILON:
                actual_deg = i
                break

        result = backend.at_add(result, 8 + DEGREE_OFFSET, 1.0)
        result = backend.at_add(result, 8 + DEGREE_OFFSET + 1, math.log(actual_deg + 1))

        # Update leading coefficient
        leading = coefficients[actual_deg] if coefficients else 0
        if abs(leading) >= EPSILON:
            result = backend.at_add(result, 8 + LEADING_COEFF_OFFSET, 1.0 if leading > 0 else -1.0)
            result = backend.at_add(result, 8 + LEADING_COEFF_OFFSET + 1, math.log(abs(leading)))
            result = backend.at_add(result, 8 + IS_MONIC_FLAG, 1.0 if abs(leading - 1.0) < EPSILON else 0.0)

        # Check if result is zero
        if all(abs(c) < EPSILON for c in coefficients):
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)

        return result

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """Subtract two polynomials."""
        negated = self.negate(emb2)
        return self.add(emb1, negated)

    def negate(self, emb: Any) -> Any:
        """Negate a polynomial (multiply by -1)."""
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Copy degree
        result = backend.at_add(result, 8 + DEGREE_OFFSET, emb[8 + DEGREE_OFFSET])
        result = backend.at_add(result, 8 + DEGREE_OFFSET + 1, emb[8 + DEGREE_OFFSET + 1])

        # Negate all coefficients (flip signs)
        for i in range(MAX_COEFFS):
            offset = COEFF_OFFSET + i * 2
            result = backend.at_add(result, 8 + offset, -emb[8 + offset])  # Flip sign
            result = backend.at_add(result, 8 + offset + 1, emb[8 + offset + 1])  # Keep magnitude

        # Negate leading coefficient
        result = backend.at_add(result, 8 + LEADING_COEFF_OFFSET, -emb[8 + LEADING_COEFF_OFFSET])
        result = backend.at_add(result, 8 + LEADING_COEFF_OFFSET + 1, emb[8 + LEADING_COEFF_OFFSET + 1])

        # Copy flags
        result = backend.at_add(result, 8 + IS_ZERO_FLAG, emb[8 + IS_ZERO_FLAG])
        result = backend.at_add(result, 8 + IS_MONIC_FLAG, emb[8 + IS_MONIC_FLAG])

        return result

    def scalar_multiply(self, emb: Any, scalar: float) -> Any:
        """
        Multiply polynomial by a scalar.

        This is multiply all coefficients.
        """
        backend = get_backend()
        if abs(scalar) < EPSILON:
            # Result is zero polynomial
            result = create_embedding()
            result = backend.at_add(result, slice(0, 8), self.domain_tag)
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
            return result

        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Copy degree
        result = backend.at_add(result, 8 + DEGREE_OFFSET, emb[8 + DEGREE_OFFSET])
        result = backend.at_add(result, 8 + DEGREE_OFFSET + 1, emb[8 + DEGREE_OFFSET + 1])

        log_scalar = math.log(abs(scalar))
        sign_scalar = 1.0 if scalar > 0 else -1.0

        # Multiply all coefficients
        for i in range(MAX_COEFFS):
            offset = COEFF_OFFSET + i * 2

            sign_c = emb[8 + offset].item()
            if abs(sign_c) < 0.5:
                continue

            # Multiply signs
            new_sign = sign_c * sign_scalar

            # Add log magnitudes (approximately)
            log_c = emb[8 + offset + 1].item()
            c = math.exp(log_c) - 1
            new_c = c * abs(scalar)

            result = backend.at_add(result, 8 + offset, new_sign)
            result = backend.at_add(result, 8 + offset + 1, math.log(new_c + 1))

        # Update leading coefficient
        sign_lead = emb[8 + LEADING_COEFF_OFFSET].item()
        if abs(sign_lead) >= 0.5:
            new_sign = sign_lead * sign_scalar
            log_lead = emb[8 + LEADING_COEFF_OFFSET + 1].item()
            new_log = log_lead + log_scalar

            result = backend.at_add(result, 8 + LEADING_COEFF_OFFSET, new_sign)
            result = backend.at_add(result, 8 + LEADING_COEFF_OFFSET + 1, new_log)

        return result

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two polynomials.

        Requires convolution of coefficient vectors (decode-operate-encode).
        """
        coeffs1 = self.decode(emb1)
        coeffs2 = self.decode(emb2)

        # Convolution
        result_deg = len(coeffs1) + len(coeffs2) - 2
        result_coeffs = [0.0] * (result_deg + 1)

        for i, c1 in enumerate(coeffs1):
            for j, c2 in enumerate(coeffs2):
                if i + j < len(result_coeffs):
                    result_coeffs[i + j] += c1 * c2

        return self.encode(result_coeffs)

    def derivative(self, emb: Any) -> Any:
        """
        Compute the derivative of a polynomial.

        d/dx(sum(a_i * x^i)) = sum(i * a_i * x^(i-1))
        """
        coeffs = self.decode(emb)

        if len(coeffs) <= 1:
            return self.encode([0.0])

        # Derivative: new_a[i] = (i+1) * old_a[i+1]
        deriv_coeffs = [(i + 1) * coeffs[i + 1] for i in range(len(coeffs) - 1)]

        return self.encode(deriv_coeffs)

    def evaluate(self, emb: Any, x: float) -> float:
        """
        Evaluate polynomial at a point.

        Uses Horner's method.
        """
        coeffs = self.decode(emb)

        if not coeffs:
            return 0.0

        # Horner's method
        result = coeffs[-1]
        for c in reversed(coeffs[:-1]):
            result = result * x + c

        return result

    # =========================================================================
    # Queries
    # =========================================================================

    def degree(self, emb: Any) -> int:
        """Get the degree of the polynomial."""
        if emb[8 + IS_ZERO_FLAG].item() > 0.5:
            return -1  # Zero polynomial has degree -1 by convention

        log_deg = emb[8 + DEGREE_OFFSET + 1].item()
        return int(round(math.exp(log_deg) - 1))

    def is_zero(self, emb: Any) -> bool:
        """Check if this is the zero polynomial."""
        return emb[8 + IS_ZERO_FLAG].item() > 0.5

    def is_constant(self, emb: Any) -> bool:
        """Check if this is a constant polynomial."""
        return self.degree(emb) <= 0

    def is_monic(self, emb: Any) -> bool:
        """Check if the polynomial is monic (leading coeff = 1)."""
        return emb[8 + IS_MONIC_FLAG].item() > 0.5

    def leading_coefficient(self, emb: Any) -> float:
        """Get the leading coefficient."""
        sign = emb[8 + LEADING_COEFF_OFFSET].item()
        if abs(sign) < 0.5:
            return 0.0
        log_c = emb[8 + LEADING_COEFF_OFFSET + 1].item()
        return sign * math.exp(log_c)
