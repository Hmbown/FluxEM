"""
Tests for UnifiedArithmeticModel (all four operations).
"""

import pytest
from fluxem import create_unified_model, evaluate_all_operations_ood


@pytest.fixture
def model():
    """Create UnifiedArithmeticModel instance for testing."""
    return create_unified_model(dim=256, linear_scale=1e7, log_scale=25.0, seed=42)


class TestUnifiedModel:
    """Tests for UnifiedArithmeticModel."""

    def test_addition(self, model):
        """Test addition."""
        test_cases = [("42+58=", 100), ("123+456=", 579), ("1000+2000=", 3000)]
        for expr, expected in test_cases:
            result = model.compute(expr)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {expr}: got {result}"

    def test_subtraction(self, model):
        """Test subtraction."""
        test_cases = [("100-37=", 63), ("1000-500=", 500), ("99999-88888=", 11111)]
        for expr, expected in test_cases:
            result = model.compute(expr)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {expr}: got {result}"

    def test_multiplication(self, model):
        """Test multiplication."""
        test_cases = [("6*7=", 42), ("12*12=", 144), ("100*100=", 10000)]
        for expr, expected in test_cases:
            result = model.compute(expr)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {expr}: got {result}"

    def test_division(self, model):
        """Test division."""
        test_cases = [("100/4=", 25), ("144/12=", 12), ("1000/8=", 125)]
        for expr, expected in test_cases:
            result = model.compute(expr)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {expr}: got {result}"


class TestOOD:
    """Tests for out-of-distribution generalization."""

    def test_ood_addition(self, model):
        """Test OOD addition."""
        test_cases = [("12345+54321=", 66666), ("50000+50000=", 100000)]
        for expr, expected in test_cases:
            result = model.compute(expr)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {expr}: got {result}"

    def test_ood_multiplication(self, model):
        """Test OOD multiplication."""
        test_cases = [("456*789=", 359784), ("1000*1000=", 1000000)]
        for expr, expected in test_cases:
            result = model.compute(expr)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {expr}: got {result}"

    def test_ood_division(self, model):
        """Test OOD division."""
        test_cases = [("56088/123=", 456)]
        for expr, expected in test_cases:
            result = model.compute(expr)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {expr}: got {result}"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_division_by_zero_positive(self, model):
        """Test division by zero with positive numerator."""
        result = model.compute("5/0")
        assert result == float('inf')

    def test_division_by_zero_negative(self, model):
        """Test division by zero with negative numerator."""
        result = model.compute("-5/0")
        assert result == float('-inf')

    def test_zero_divided_by_number(self, model):
        """Test zero divided by a number."""
        result = model.compute("0/5")
        assert result == 0.0


class TestStatisticalAccuracy:
    """Statistical accuracy tests."""

    def test_all_operations_ood(self, model):
        """Test OOD accuracy on all operations."""
        results = evaluate_all_operations_ood(model, n_samples=50)

        # Each operation should have high accuracy
        assert results['+'] >= 0.95, f"Addition accuracy: {results['+']}"
        assert results['-'] >= 0.95, f"Subtraction accuracy: {results['-']}"
        assert results['*'] >= 0.95, f"Multiplication accuracy: {results['*']}"
        assert results['/'] >= 0.90, f"Division accuracy: {results['/']}"
        assert results['overall'] >= 0.90, f"Overall accuracy: {results['overall']}"
