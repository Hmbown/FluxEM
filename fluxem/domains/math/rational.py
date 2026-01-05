"""
Rational Number Encoder.

Embeds rational numbers p/q in canonical form (gcd(p,q)=1, q>0).
Multiplication multiplies numerators and denominators.
Addition requires common denominator computation.
"""

import math
from typing import Any, Tuple, Union
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
)


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a if a else 1


def canonical_form(p: int, q: int) -> Tuple[int, int]:
    """Reduce p/q to canonical form."""
    if q == 0:
        raise ValueError("Denominator cannot be zero")
    if p == 0:
        return (0, 1)

    # Ensure q > 0
    if q < 0:
        p, q = -p, -q

    # Reduce by GCD
    g = gcd(p, q)
    return (p // g, q // g)


# Embedding layout:
# dims 0-1:   Numerator (sign, log|p|+1)
# dims 2-3:   Denominator (1, log(q))
# dims 4-5:   Decimal value (sign, log|p/q|)
# dim 6:      Is integer flag (q == 1)
# dim 7:      Is zero flag (p == 0)
# dim 8:      Complexity (log(|p| + q))
# dims 9-15:  Reserved

NUMER_OFFSET = 0
DENOM_OFFSET = 2
DECIMAL_OFFSET = 4
IS_INTEGER_FLAG = 6
IS_ZERO_FLAG = 7
COMPLEXITY_OFFSET = 8


class RationalEncoder:
    """
    Encoder for rational numbers.

    Stores p/q in canonical form with integer values.
    Multiplication is implemented in embedding space.
    """

    domain_tag = DOMAIN_TAGS["math_rational"]
    domain_name = "math_rational"

    def encode(self, value: Union[Tuple[int, int], float, int]) -> Any:
        """
        Encode a rational number.

        Args:
            value: (numerator, denominator) tuple, integer, or float (converted to ratio)

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(value, int):
            p, q = value, 1
        elif isinstance(value, float):
            # Convert float to fraction (limited precision)
            if value == 0:
                p, q = 0, 1
            elif abs(value) < 1e-10:
                p, q = 0, 1
            else:
                # Use continued fraction approximation
                p, q = self._float_to_fraction(value, max_denominator=1000000)
        else:
            p, q = value

        p, q = canonical_form(p, q)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Numerator
        if p == 0:
            emb = backend.at_add(emb, 8 + NUMER_OFFSET, 0.0)
            emb = backend.at_add(emb, 8 + NUMER_OFFSET + 1, 0.0)
            emb = backend.at_add(emb, 8 + IS_ZERO_FLAG, 1.0)
        else:
            emb = backend.at_add(emb, 8 + NUMER_OFFSET, 1.0 if p > 0 else -1.0)
            emb = backend.at_add(emb, 8 + NUMER_OFFSET + 1, math.log(abs(p) + 1))

        # Denominator (always positive)
        emb = backend.at_add(emb, 8 + DENOM_OFFSET, 1.0)
        emb = backend.at_add(emb, 8 + DENOM_OFFSET + 1, math.log(q))

        # Decimal approximation
        if p != 0:
            decimal = p / q
            emb = backend.at_add(emb, 8 + DECIMAL_OFFSET, 1.0 if decimal > 0 else -1.0)
            emb = backend.at_add(emb, 8 + DECIMAL_OFFSET + 1, math.log(abs(decimal)))

        # Flags
        emb = backend.at_add(emb, 8 + IS_INTEGER_FLAG, 1.0 if q == 1 else 0.0)

        # Complexity
        emb = backend.at_add(emb, 8 + COMPLEXITY_OFFSET, math.log(abs(p) + q + 1))

        return emb

    def _float_to_fraction(self, x: float, max_denominator: int = 1000000) -> Tuple[int, int]:
        """Convert float to fraction using continued fractions."""
        if x == 0:
            return (0, 1)

        sign = 1 if x > 0 else -1
        x = abs(x)

        # Simple approach: multiply until we get close to an integer
        for q in range(1, max_denominator + 1):
            p = round(x * q)
            if abs(x - p / q) < 1e-10:
                return (sign * p, q)

        # Fallback: use large denominator
        q = max_denominator
        p = round(x * q)
        return (sign * p, q)

    def decode(self, emb: Any) -> Tuple[int, int]:
        """
        Decode embedding to (numerator, denominator).
        """
        if emb[8 + IS_ZERO_FLAG].item() > 0.5:
            return (0, 1)

        # Numerator
        sign_p = emb[8 + NUMER_OFFSET].item()
        log_p = emb[8 + NUMER_OFFSET + 1].item()
        p = int(round(math.exp(log_p) - 1))
        if sign_p < 0:
            p = -p

        # Denominator
        log_q = emb[8 + DENOM_OFFSET + 1].item()
        q = max(1, int(round(math.exp(log_q))))

        return canonical_form(p, q)

    def decode_float(self, emb: Any) -> float:
        """Decode to floating point value."""
        p, q = self.decode(emb)
        return p / q if q != 0 else 0.0

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid rational."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Operations
    # =========================================================================

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two rationals.

        (p1/q1) * (p2/q2) = (p1*p2)/(q1*q2)
        In embedding space: add log-numerators, add log-denominators.
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Check for zeros
        is_zero1 = emb1[8 + IS_ZERO_FLAG].item() > 0.5
        is_zero2 = emb2[8 + IS_ZERO_FLAG].item() > 0.5

        if is_zero1 or is_zero2:
            result = backend.at_add(result, 8 + IS_ZERO_FLAG, 1.0)
            result = backend.at_add(result, 8 + DENOM_OFFSET, 1.0)
            return result

        # Multiply numerators (add logs)
        sign1 = emb1[8 + NUMER_OFFSET].item()
        sign2 = emb2[8 + NUMER_OFFSET].item()
        log_p1 = emb1[8 + NUMER_OFFSET + 1].item()
        log_p2 = emb2[8 + NUMER_OFFSET + 1].item()

        result_sign = sign1 * sign2
        # log(p1 * p2 + 1) ≈ log(exp(log_p1 - 1 + 1) * exp(log_p2 - 1 + 1) + 1)
        # Simplified: just add the logs (approximate but works for moderate values)
        p1 = math.exp(log_p1) - 1
        p2 = math.exp(log_p2) - 1
        result_p = p1 * p2

        result = backend.at_add(result, 8 + NUMER_OFFSET, result_sign)
        result = backend.at_add(result, 8 + NUMER_OFFSET + 1, math.log(result_p + 1))

        # Multiply denominators (add logs)
        log_q1 = emb1[8 + DENOM_OFFSET + 1].item()
        log_q2 = emb2[8 + DENOM_OFFSET + 1].item()

        result = backend.at_add(result, 8 + DENOM_OFFSET, 1.0)
        result = backend.at_add(result, 8 + DENOM_OFFSET + 1, log_q1 + log_q2)

        # Update decimal and flags
        p, q = self.decode(result)
        if q != 0:
            decimal = p / q
            result = backend.at_add(result, 8 + DECIMAL_OFFSET, 1.0 if decimal > 0 else -1.0)
            result = backend.at_add(result, 8 + DECIMAL_OFFSET + 1, 
                math.log(abs(decimal)) if abs(decimal) > EPSILON else -100
            )

        result = backend.at_add(result, 8 + IS_INTEGER_FLAG, 1.0 if q == 1 else 0.0)
        result = backend.at_add(result, 8 + COMPLEXITY_OFFSET, math.log(abs(p) + q + 1))

        return result

    def divide(self, emb1: Any, emb2: Any) -> Any:
        """
        Divide two rationals.

        (p1/q1) / (p2/q2) = (p1*q2)/(q1*p2)
        """
        # Invert second rational and multiply
        inverted = self.reciprocal(emb2)
        return self.multiply(emb1, inverted)

    def reciprocal(self, emb: Any) -> Any:
        """
        Reciprocal of a rational.

        (p/q) -> (q/p)
        """
        backend = get_backend()
        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        if emb[8 + IS_ZERO_FLAG].item() > 0.5:
            # 1/0 = infinity (encode as very large)
            result = backend.at_add(result, 8 + NUMER_OFFSET, 1.0)
            result = backend.at_add(result, 8 + NUMER_OFFSET + 1, 100.0)
            result = backend.at_add(result, 8 + DENOM_OFFSET, 1.0)
            return result

        # Swap numerator and denominator
        # New numerator = old denominator (always positive in canonical form)
        result = backend.at_add(result, 8 + NUMER_OFFSET, emb[8 + NUMER_OFFSET])  # Keep sign
        result = backend.at_add(result, 8 + NUMER_OFFSET + 1, emb[8 + DENOM_OFFSET + 1])

        # New denominator = |old numerator|
        log_p = emb[8 + NUMER_OFFSET + 1].item()
        result = backend.at_add(result, 8 + DENOM_OFFSET, 1.0)
        result = backend.at_add(result, 8 + DENOM_OFFSET + 1, log_p)

        # Update decimal
        p, q = self.decode(result)
        if q != 0 and p != 0:
            decimal = p / q
            result = backend.at_add(result, 8 + DECIMAL_OFFSET, 1.0 if decimal > 0 else -1.0)
            result = backend.at_add(result, 8 + DECIMAL_OFFSET + 1, math.log(abs(decimal)))

        return result

    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two rationals.

        Requires decode-operate-encode (finding common denominator).
        """
        p1, q1 = self.decode(emb1)
        p2, q2 = self.decode(emb2)

        # Common denominator: q1 * q2 / gcd(q1, q2)
        g = gcd(q1, q2)
        lcm_q = (q1 * q2) // g

        # Scale numerators
        new_p = p1 * (lcm_q // q1) + p2 * (lcm_q // q2)

        return self.encode((new_p, lcm_q))

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """Subtract two rationals."""
        p1, q1 = self.decode(emb1)
        p2, q2 = self.decode(emb2)

        g = gcd(q1, q2)
        lcm_q = (q1 * q2) // g

        new_p = p1 * (lcm_q // q1) - p2 * (lcm_q // q2)

        return self.encode((new_p, lcm_q))

    def negate(self, emb: Any) -> Any:
        """Negate a rational."""
        backend = get_backend()
        result = emb.copy() if hasattr(emb, 'copy') else backend.array(emb)
        # Flip numerator sign
        current_sign = result[8 + NUMER_OFFSET].item()
        result = backend.at_add(result, 8 + NUMER_OFFSET, -2 * current_sign)
        # Flip decimal sign
        current_dec_sign = result[8 + DECIMAL_OFFSET].item()
        result = backend.at_add(result, 8 + DECIMAL_OFFSET, -2 * current_dec_sign)
        return result

    # =========================================================================
    # Queries
    # =========================================================================

    def is_integer(self, emb: Any) -> bool:
        """Check if the rational is an integer."""
        return emb[8 + IS_INTEGER_FLAG].item() > 0.5

    def is_zero(self, emb: Any) -> bool:
        """Check if the rational is zero."""
        return emb[8 + IS_ZERO_FLAG].item() > 0.5

    def is_positive(self, emb: Any) -> bool:
        """Check if the rational is positive."""
        return (emb[8 + NUMER_OFFSET].item() > 0.5 and
                not self.is_zero(emb))

    def is_negative(self, emb: Any) -> bool:
        """Check if the rational is negative."""
        return emb[8 + NUMER_OFFSET].item() < -0.5

    def compare(self, emb1: Any, emb2: Any) -> int:
        """
        Compare two rationals.

        Returns: -1 if r1 < r2, 0 if equal, 1 if r1 > r2
        """
        # Use decimal approximations for comparison
        d1 = emb1[8 + DECIMAL_OFFSET].item() * math.exp(emb1[8 + DECIMAL_OFFSET + 1].item())
        d2 = emb2[8 + DECIMAL_OFFSET].item() * math.exp(emb2[8 + DECIMAL_OFFSET + 1].item())

        if abs(d1 - d2) < EPSILON:
            return 0
        return 1 if d1 > d2 else -1
