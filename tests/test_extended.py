"""
Tests for ExtendedOps (powers, roots, exp, ln).
"""

import math
import pytest
from fluxem import create_extended_ops


@pytest.fixture
def ops():
    """Create ExtendedOps instance for testing."""
    return create_extended_ops(dim=256, log_scale=25.0, seed=42)


class TestPower:
    """Tests for power operations."""

    def test_power_basic(self, ops):
        """Test basic power computations."""
        test_cases = [(2, 2, 4), (2, 3, 8), (3, 2, 9), (3, 3, 27)]
        for base, exp, expected in test_cases:
            result = ops.power(base, exp)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {base}^{exp}: got {result}"

    def test_power_ood(self, ops):
        """Test OOD power computations."""
        test_cases = [
            (2, 10, 1024),
            (2, 20, 1048576),
            (10, 6, 1000000),
        ]
        for base, exp, expected in test_cases:
            result = ops.power(base, exp)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {base}^{exp}: got {result}"

    def test_power_identity(self, ops):
        """Test a^1 = a."""
        for a in [2, 5, 10, 100]:
            result = ops.power(a, 1)
            rel_error = abs(result - a) / a
            assert rel_error < 0.01, f"Failed for {a}^1"

    def test_power_zero_exponent(self, ops):
        """Test a^0 = 1."""
        for a in [2, 5, 10, 100]:
            result = ops.power(a, 0)
            assert abs(result - 1) < 0.1, f"Failed for {a}^0: got {result}"


class TestSqrt:
    """Tests for square root."""

    def test_sqrt_square_numbers(self, ops):
        """Test sqrt of square numbers."""
        test_cases = [(4, 2), (9, 3), (16, 4), (25, 5), (100, 10)]
        for n, expected in test_cases:
            result = ops.sqrt(n)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for sqrt({n}): got {result}"

    def test_sqrt_ood(self, ops):
        """Test OOD sqrt."""
        test_cases = [(10000, 100), (1000000, 1000)]
        for n, expected in test_cases:
            result = ops.sqrt(n)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for sqrt({n}): got {result}"


class TestCbrt:
    """Tests for cube root."""

    def test_cbrt_cube_numbers(self, ops):
        """Test cbrt of cube numbers."""
        test_cases = [(8, 2), (27, 3), (64, 4), (125, 5), (1000, 10)]
        for n, expected in test_cases:
            result = ops.cbrt(n)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for cbrt({n}): got {result}"

    def test_cbrt_negative(self, ops):
        """Test cbrt of negative numbers."""
        test_cases = [(-8, -2), (-27, -3)]
        for n, expected in test_cases:
            result = ops.cbrt(n)
            rel_error = abs(result - expected) / abs(expected)
            assert rel_error < 0.01, f"Failed for cbrt({n}): got {result}"


class TestExp:
    """Tests for exponential function."""

    def test_exp_zero(self, ops):
        """Test e^0 = 1."""
        result = ops.exp(0)
        assert abs(result - 1) < 0.1, f"exp(0) = {result}"

    def test_exp_one(self, ops):
        """Test e^1 = e."""
        result = ops.exp(1)
        expected = math.e
        rel_error = abs(result - expected) / expected
        assert rel_error < 0.01, f"exp(1) = {result}"

    def test_exp_basic(self, ops):
        """Test exp for small values."""
        for x in [2, 3, 5]:
            result = ops.exp(x)
            expected = math.exp(x)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"exp({x}) = {result}"


class TestLn:
    """Tests for natural logarithm."""

    def test_ln_one(self, ops):
        """Test ln(1) = 0."""
        result = ops.ln(1)
        assert abs(result) < 0.1, f"ln(1) = {result}"

    def test_ln_e(self, ops):
        """Test ln(e) = 1."""
        result = ops.ln(math.e)
        assert abs(result - 1) < 0.1, f"ln(e) = {result}"

    def test_ln_basic(self, ops):
        """Test ln for basic values."""
        for x in [2, 5, 10, 100]:
            result = ops.ln(x)
            expected = math.log(x)
            abs_error = abs(result - expected)
            assert abs_error < 0.1, f"ln({x}) = {result}"


class TestExpLnInverse:
    """Tests for exp-ln inverse relationship."""

    def test_ln_exp_identity(self, ops):
        """Test ln(exp(x)) = x."""
        for x in [0, 1, 2, 5]:
            exp_x = ops.exp(x)
            result = ops.ln(exp_x)
            abs_error = abs(result - x)
            assert abs_error < 0.2, f"ln(exp({x})) = {result}"

    def test_exp_ln_identity(self, ops):
        """Test exp(ln(x)) = x."""
        for x in [1, 2, 5, 10, 100]:
            ln_x = ops.ln(x)
            result = ops.exp(ln_x)
            rel_error = abs(result - x) / x
            assert rel_error < 0.01, f"exp(ln({x})) = {result}"


class TestLogBase:
    """Tests for logarithms with other bases."""

    def test_log10(self, ops):
        """Test log base 10."""
        test_cases = [(10, 1), (100, 2), (1000, 3)]
        for x, expected in test_cases:
            result = ops.log10(x)
            assert abs(result - expected) < 0.1, f"log10({x}) = {result}"

    def test_log2(self, ops):
        """Test log base 2."""
        test_cases = [(2, 1), (8, 3), (1024, 10)]
        for x, expected in test_cases:
            result = ops.log2(x)
            assert abs(result - expected) < 0.2, f"log2({x}) = {result}"
