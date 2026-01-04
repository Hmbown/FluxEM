"""
Extended operations for FluxEM.

This module extends arithmetic beyond +, -, *, / to powers, roots, and
exponentials using logarithmic embeddings.

Key identities (magnitude space):
    log_mag(a^b) = b * log_mag(a)
    sqrt(a) = a^(1/2)  -> 0.5 * log_mag(a)
    cbrt(a) = a^(1/3)  -> (1/3) * log_mag(a)
    exp(x) = e^x       -> embedding with log magnitude = x
    ln(x)              -> extract log magnitude from embedding

Sign is handled separately for negative bases; fractional powers of
negative numbers return magnitude only.

Reference: Flux Mathematics textbook, Chapter 8
"""

from __future__ import annotations

import math
from typing import Union, Any

from ..backend import get_backend
from .log_encoder import LogarithmicNumberEncoder
from .linear_encoder import NumberEncoder


class ExtendedOps:
    """
    Extended operations using logarithmic embeddings.

    Provides power, roots, exp, and ln operations with embedding arithmetic.

    The mathematical basis:
    - power(a, b): log_embed(a^b) = b * log_embed(a)  [scalar multiplication]
    - sqrt(a): power(a, 0.5)
    - cbrt(a): power(a, 1/3)
    - exp(x): e^x, embedding where log(result) = x
    - ln(a): natural log, extracts log magnitude from embedding

    Parameters
    ----------
    dim : int
        Embedding dimension.
    log_scale : float
        Scale factor for log values.
    seed : int
        Random seed for direction vectors.
    """

    log_encoder: LogarithmicNumberEncoder
    linear_encoder: NumberEncoder
    dim: int

    def __init__(
        self,
        dim: int = 256,
        log_scale: float = 25.0,
        linear_scale: float = 1e10,
        seed: int = 42,
    ):
        self.log_encoder = LogarithmicNumberEncoder(
            dim=dim, log_scale=log_scale, seed=seed
        )
        self.linear_encoder = NumberEncoder(
            dim=dim, scale=linear_scale, seed=seed + 100
        )
        self.dim = dim

    # =========================================================================
    # POWER OPERATIONS
    # =========================================================================

    def power_from_embedding(
        self,
        base_emb: Any,
        exponent: float
    ) -> Any:
        """
        Compute a^b in embedding space.

        Mathematical basis (positive a):
            log(a^b) = b * log(a)
        Therefore:
            log_mag(a^b) = b * log_mag(a)

        Sign handling:
        - Negative base, integer exponent: sign = (-1)^b
        - Negative base, fractional exponent: magnitude only (real branch)

        Parameters
        ----------
        base_emb : Array
            Logarithmic embedding of the base, shape [dim].
        exponent : float
            The exponent value.

        Returns
        -------
        Array
            Logarithmic embedding of base^exponent, shape [dim].
        """
        backend = get_backend()

        log_mag = backend.dot(base_emb, self.log_encoder.direction)
        result_log_mag = exponent * log_mag

        sign_proj = backend.dot(base_emb, self.log_encoder.sign_direction) / 0.5
        base_sign = backend.sign(sign_proj)
        base_sign = backend.where(backend.abs(sign_proj) < 0.1, backend.array(1.0), base_sign)

        # Check if exponent is odd integer
        exp_is_int = (exponent == int(exponent))
        exp_is_odd = (int(exponent) % 2 == 1) if exp_is_int else False
        result_sign = -1.0 if (float(base_sign) < 0 and exp_is_odd) else 1.0

        mag_emb = float(result_log_mag) * self.log_encoder.direction
        sign_emb = result_sign * self.log_encoder.sign_direction * 0.5
        result = mag_emb + sign_emb

        zero_base = self.log_encoder._is_zero_embedding(base_emb)
        if exponent < 0:
            zero_result = self.log_encoder._inf_embedding(backend.array(1.0))
        elif exponent == 0:
            zero_result = self.log_encoder.sign_direction * 0.5
        else:
            zero_result = backend.zeros(self.dim)

        return backend.where(zero_base, zero_result, result)

    def power(self, base: float, exponent: float) -> float:
        """
        Compute base^exponent using logarithmic embeddings.

        Negative bases with non-integer exponents return magnitude only.

        Parameters
        ----------
        base : float
            The base value.
        exponent : float
            The exponent value.

        Returns
        -------
        float
            base^exponent computed via embedding space.
        """
        if base == 0:
            return 0.0 if exponent > 0 else float('inf')
        if base < 0 and exponent != int(exponent):
            base = abs(base)

        base_emb = self.log_encoder.encode_number(base)
        result_emb = self.power_from_embedding(base_emb, exponent)
        return self.log_encoder.decode(result_emb)

    def sqrt_from_embedding(self, emb: Any) -> Any:
        """
        Compute sqrt(a) from embedding.

        sqrt(a) = a^(1/2), so:
            log_embed(sqrt(a)) = 0.5 * log_embed(a)

        Parameters
        ----------
        emb : Array
            Logarithmic embedding of a, shape [dim].

        Returns
        -------
        Array
            Logarithmic embedding of sqrt(a), shape [dim].
        """
        return self.power_from_embedding(emb, 0.5)

    def sqrt(self, a: float) -> float:
        """
        Compute sqrt(a) using logarithmic embeddings.

        Parameters
        ----------
        a : float
            The value to take square root of.

        Returns
        -------
        float
            sqrt(a) computed via embedding space.
        """
        if a < 0:
            a = abs(a)
        return self.power(a, 0.5)

    def cbrt_from_embedding(self, emb: Any) -> Any:
        """
        Compute cbrt(a) from embedding.

        cbrt(a) = a^(1/3), so:
            log_embed(cbrt(a)) = (1/3) * log_embed(a)

        Parameters
        ----------
        emb : Array
            Logarithmic embedding of a, shape [dim].

        Returns
        -------
        Array
            Logarithmic embedding of cbrt(a), shape [dim].
        """
        return self.power_from_embedding(emb, 1.0 / 3.0)

    def cbrt(self, a: float) -> float:
        """
        Compute cbrt(a) using logarithmic embeddings.

        Cube root handles negative numbers (cbrt(-8) = -2).

        Parameters
        ----------
        a : float
            The value to take cube root of.

        Returns
        -------
        float
            cbrt(a) computed via embedding space.
        """
        if a < 0:
            return -self.power(abs(a), 1.0 / 3.0)
        return self.power(a, 1.0 / 3.0)

    def nth_root(self, a: float, n: int) -> float:
        """
        Compute the nth root of a.

        nth_root(a) = a^(1/n)

        Parameters
        ----------
        a : float
            The value to take nth root of.
        n : int
            The root degree.

        Returns
        -------
        float
            nth_root(a) computed via embedding space.
        """
        if n == 0:
            return float('inf')
        if a < 0 and n % 2 == 1:
            return -self.power(abs(a), 1.0 / n)
        elif a < 0:
            a = abs(a)
        return self.power(a, 1.0 / n)

    # =========================================================================
    # EXPONENTIAL AND LOGARITHM
    # =========================================================================

    def exp_to_embedding(self, x: float) -> Any:
        """
        Compute embedding of e^x.

        Since log(e^x) = x, the log magnitude is simply x.

        Parameters
        ----------
        x : float
            The exponent.

        Returns
        -------
        Array
            Logarithmic embedding of e^x, shape [dim].
        """
        log_normalized = x / self.log_encoder.log_scale

        mag_emb = log_normalized * self.log_encoder.direction
        sign_emb = 1.0 * self.log_encoder.sign_direction * 0.5

        return mag_emb + sign_emb

    def exp(self, x: float) -> float:
        """
        Compute e^x using logarithmic embeddings.

        Parameters
        ----------
        x : float
            The exponent.

        Returns
        -------
        float
            e^x computed via embedding space.
        """
        emb = self.exp_to_embedding(x)
        return self.log_encoder.decode(emb)

    def ln_from_embedding(self, emb: Any) -> float:
        """
        Compute ln(a) from embedding.

        ln(a) is simply the log magnitude extracted from the embedding.

        Parameters
        ----------
        emb : Array
            Logarithmic embedding of a, shape [dim].

        Returns
        -------
        float
            ln(a).
        """
        backend = get_backend()
        log_normalized = backend.dot(emb, self.log_encoder.direction)
        return float(log_normalized * self.log_encoder.log_scale)

    def ln(self, a: float) -> float:
        """
        Compute ln(a) using logarithmic embeddings.

        Parameters
        ----------
        a : float
            The value to take natural log of. Must be positive.

        Returns
        -------
        float
            ln(a) computed via embedding space.
        """
        if a <= 0:
            return float('-inf')
        emb = self.log_encoder.encode_number(a)
        return self.ln_from_embedding(emb)

    def log_base(self, a: float, base: float) -> float:
        """
        Compute log_base(a) using the change of base formula.

        log_b(a) = ln(a) / ln(b)

        Parameters
        ----------
        a : float
            The argument.
        base : float
            The base of the logarithm.

        Returns
        -------
        float
            log_base(a).
        """
        if a <= 0 or base <= 0 or base == 1:
            return float('nan')
        return self.ln(a) / self.ln(base)

    def log10(self, a: float) -> float:
        """Compute log base 10."""
        return self.log_base(a, 10.0)

    def log2(self, a: float) -> float:
        """Compute log base 2."""
        return self.log_base(a, 2.0)

    # =========================================================================
    # VERIFICATION FUNCTIONS
    # =========================================================================

    def verify_power_theorem(
        self,
        base: float,
        exponent: float,
        tol: float = 0.01
    ) -> bool:
        """
        Verify that log_embed(a^b) ~ b * log_embed(a).

        Parameters
        ----------
        base, exponent : float
            Values to test.
        tol : float
            Relative tolerance.

        Returns
        -------
        bool
            True if the power theorem holds.
        """
        if base <= 0:
            return True

        expected = base ** exponent
        computed = self.power(base, exponent)

        if expected == 0:
            return abs(computed) < tol

        rel_error = abs(computed - expected) / abs(expected)
        return rel_error < tol

    def verify_exp_ln_inverse(self, x: float, tol: float = 0.01) -> bool:
        """
        Verify that ln(exp(x)) ~ x.

        Parameters
        ----------
        x : float
            Value to test.
        tol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if exp and ln are inverses.
        """
        exp_x = self.exp(x)
        ln_exp_x = self.ln(exp_x)
        return abs(ln_exp_x - x) < tol


def create_extended_ops(
    dim: int = 256,
    log_scale: float = 25.0,
    seed: int = 42,
) -> ExtendedOps:
    """Create an ExtendedOps instance."""
    return ExtendedOps(dim=dim, log_scale=log_scale, seed=seed)


if __name__ == "__main__":
    print("ExtendedOps demo")

    backend = get_backend()
    print(f"Using backend: {backend.name}")

    ops = create_extended_ops(dim=256, log_scale=25.0, seed=42)

    print("\nPower operations: log_embed(a^b) = b * log_embed(a)")

    power_tests = [
        (2, 3, 8),
        (2, 10, 1024),
        (10, 2, 100),
        (10, 6, 1000000),
        (3, 4, 81),
        (5, 5, 3125),
        (2, 20, 1048576),
    ]

    for base, exp, expected in power_tests:
        result = ops.power(base, exp)
        rel_error = abs(result - expected) / expected if expected != 0 else 0
        status = "PASS" if rel_error < 0.01 else "FAIL"
        print(f"  {base}^{exp} = {result:.2f} (expected: {expected}, rel_err: {rel_error:.2e}) [{status}]")

    print("\nRoot operations: log_embed(nth_root(a)) = (1/n) * log_embed(a)")

    root_tests = [
        ("sqrt", 16, 4),
        ("sqrt", 100, 10),
        ("sqrt", 10000, 100),
        ("sqrt", 1000000, 1000),
        ("cbrt", 8, 2),
        ("cbrt", 27, 3),
        ("cbrt", 1000, 10),
        ("cbrt", 1000000, 100),
    ]

    for op, val, expected in root_tests:
        if op == "sqrt":
            result = ops.sqrt(val)
            symbol = "sqrt"
        else:
            result = ops.cbrt(val)
            symbol = "cbrt"
        rel_error = abs(result - expected) / expected if expected != 0 else 0
        status = "PASS" if rel_error < 0.01 else "FAIL"
        print(f"  {symbol}({val}) = {result:.2f} (expected: {expected}, rel_err: {rel_error:.2e}) [{status}]")

    print("\nExponential: exp(x) = e^x")

    exp_tests = [
        (0, 1),
        (1, math.e),
        (2, math.e ** 2),
        (5, math.e ** 5),
        (10, math.e ** 10),
    ]

    for x, expected in exp_tests:
        result = ops.exp(x)
        rel_error = abs(result - expected) / expected if expected != 0 else 0
        status = "PASS" if rel_error < 0.01 else "FAIL"
        print(f"  exp({x}) = {result:.2f} (expected: {expected:.2f}, rel_err: {rel_error:.2e}) [{status}]")

    print("\nNatural log: ln(x)")

    ln_tests = [
        (1, 0),
        (math.e, 1),
        (10, math.log(10)),
        (100, math.log(100)),
        (1000, math.log(1000)),
    ]

    for x, expected in ln_tests:
        result = ops.ln(x)
        abs_error = abs(result - expected)
        status = "PASS" if abs_error < 0.1 else "FAIL"
        print(f"  ln({x}) = {result:.4f} (expected: {expected:.4f}, abs_err: {abs_error:.2e}) [{status}]")

    print("\nExp-ln inverse: ln(exp(x)) = x")

    for x in [0, 1, 2, 5, 10]:
        exp_x = ops.exp(x)
        ln_exp_x = ops.ln(exp_x)
        error = abs(ln_exp_x - x)
        status = "PASS" if error < 0.1 else "FAIL"
        print(f"  ln(exp({x})) = {ln_exp_x:.4f} (expected: {x}, error: {error:.2e}) [{status}]")
