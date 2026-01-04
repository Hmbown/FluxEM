"""
Tests for composed/chained operations.

Validates that cross-space operations work correctly and
quantifies error accumulation with chain depth.
"""

import pytest
from fluxem import create_unified_model


@pytest.fixture
def model():
    """Create UnifiedArithmeticModel instance for testing."""
    return create_unified_model(dim=256, linear_scale=1e7, log_scale=25.0, seed=42)


class TestCrossSpaceComposition:
    """Tests for operations that cross linear/log spaces."""

    def test_add_then_multiply(self, model):
        """(a + b) * c: linear -> log"""
        # Compute (10 + 20) * 3 = 90
        sum_result = int(model.compute("10+20="))
        product = model.compute(f"{sum_result}*3=")

        expected = 90.0
        rel_error = abs(product - expected) / expected
        assert rel_error < 0.01, f"(10+20)*3 = {product}, expected {expected}"

    def test_multiply_then_add(self, model):
        """(a * b) + c: log -> linear"""
        # Compute (5 * 6) + 10 = 40
        product = int(model.compute("5*6="))
        sum_result = model.compute(f"{product}+10=")

        expected = 40.0
        rel_error = abs(sum_result - expected) / expected
        assert rel_error < 0.01, f"(5*6)+10 = {sum_result}, expected {expected}"

    def test_divide_then_subtract(self, model):
        """(a / b) - c: log -> linear"""
        # Compute (100 / 4) - 5 = 20
        quotient = int(model.compute("100/4="))
        diff = model.compute(f"{quotient}-5=")

        expected = 20.0
        rel_error = abs(diff - expected) / expected
        assert rel_error < 0.01, f"(100/4)-5 = {diff}, expected {expected}"

    def test_subtract_then_divide(self, model):
        """(a - b) / c: linear -> log"""
        # Compute (100 - 20) / 4 = 20
        diff = int(model.compute("100-20="))
        quotient = model.compute(f"{diff}/4=")

        expected = 20.0
        rel_error = abs(quotient - expected) / expected
        assert rel_error < 0.01, f"(100-20)/4 = {quotient}, expected {expected}"

    def test_triple_chain_linear_log_linear(self, model):
        """((a + b) * c) + d: linear -> log -> linear"""
        # ((10 + 20) * 3) + 5 = 95
        step1 = int(model.compute("10+20="))
        step2 = int(model.compute(f"{step1}*3="))
        step3 = model.compute(f"{step2}+5=")

        expected = 95.0
        rel_error = abs(step3 - expected) / expected
        assert rel_error < 0.01, f"((10+20)*3)+5 = {step3}, expected {expected}"

    def test_triple_chain_log_linear_log(self, model):
        """((a * b) + c) * d: log -> linear -> log"""
        # ((4 * 5) + 10) * 2 = 60
        step1 = int(model.compute("4*5="))
        step2 = int(model.compute(f"{step1}+10="))
        step3 = model.compute(f"{step2}*2=")

        expected = 60.0
        rel_error = abs(step3 - expected) / expected
        assert rel_error < 0.01, f"((4*5)+10)*2 = {step3}, expected {expected}"

    def test_four_operation_chain(self, model):
        """(((a + b) * c) - d) / e"""
        # (((10 + 10) * 5) - 50) / 10 = 5
        step1 = int(model.compute("10+10="))
        step2 = int(model.compute(f"{step1}*5="))
        step3 = int(model.compute(f"{step2}-50="))
        step4 = model.compute(f"{step3}/10=")

        expected = 5.0
        rel_error = abs(step4 - expected) / expected
        assert rel_error < 0.01, f"Chain result = {step4}, expected {expected}"


class TestSameSpaceChains:
    """Tests for operations staying in the same space."""

    def test_pure_addition_chain(self, model):
        """a + b + c + d: stays in linear space"""
        # 10 + 20 + 30 + 40 = 100
        step1 = int(model.compute("10+20="))
        step2 = int(model.compute(f"{step1}+30="))
        step3 = model.compute(f"{step2}+40=")

        expected = 100.0
        rel_error = abs(step3 - expected) / expected
        assert rel_error < 0.001, f"Pure add chain = {step3}, expected {expected}"

    def test_pure_multiplication_chain(self, model):
        """a * b * c * d: stays in log space"""
        # 2 * 3 * 4 * 5 = 120
        step1 = int(model.compute("2*3="))
        step2 = int(model.compute(f"{step1}*4="))
        step3 = model.compute(f"{step2}*5=")

        expected = 120.0
        rel_error = abs(step3 - expected) / expected
        assert rel_error < 0.01, f"Pure mul chain = {step3}, expected {expected}"

    def test_mixed_add_sub(self, model):
        """a + b - c + d: stays in linear space"""
        # 100 + 50 - 30 + 20 = 140
        step1 = int(model.compute("100+50="))
        step2 = int(model.compute(f"{step1}-30="))
        step3 = model.compute(f"{step2}+20=")

        expected = 140.0
        rel_error = abs(step3 - expected) / expected
        assert rel_error < 0.01, f"Add/sub chain = {step3}, expected {expected}"

    def test_mixed_mul_div(self, model):
        """a * b / c * d: stays in log space"""
        # 100 * 4 / 8 * 2 = 100
        step1 = int(model.compute("100*4="))
        step2 = int(model.compute(f"{step1}/8="))
        step3 = model.compute(f"{step2}*2=")

        expected = 100.0
        rel_error = abs(step3 - expected) / expected
        assert rel_error < 0.03, f"Mul/div chain = {step3}, expected {expected}"


class TestCompositionErrorAccumulation:
    """Quantify error growth with chain depth."""

    def test_repeated_multiply_divide(self, model):
        """
        Chain: x -> x*2 -> /2 -> *2 -> /2 -> ...
        Each round-trip should accumulate ~1e-6 error.
        """
        x = 1000
        errors = []

        for depth in range(1, 11):
            result = float(x)
            for _ in range(depth):
                result = model.compute(f"{int(result)}*2=")
                result = model.compute(f"{int(result)}/2=")

            rel_error = abs(result - x) / x
            errors.append((depth, rel_error))

        # Error should grow roughly linearly with depth
        # At depth 10, expect < 1e-2 total error (accumulates ~1e-3 per round-trip)
        final_error = errors[-1][1]
        assert final_error < 1e-2, f"Error at depth 10: {final_error}"

    def test_repeated_add_subtract(self, model):
        """
        Chain: x -> x+100 -> -100 -> +100 -> -100 -> ...
        Linear space should accumulate minimal error.
        """
        x = 5000
        errors = []

        for depth in range(1, 11):
            result = float(x)
            for _ in range(depth):
                result = model.compute(f"{int(result)}+100=")
                result = model.compute(f"{int(result)}-100=")

            rel_error = abs(result - x) / x
            errors.append((depth, rel_error))

        # Linear operations should have low error (accumulates with int() truncation)
        final_error = errors[-1][1]
        assert final_error < 1e-2, f"Linear chain error at depth 10: {final_error}"

    def test_alternating_spaces_deep(self, model):
        """
        Deep chain alternating between add and multiply.
        Start with 100, repeatedly: +10, *1.1
        This test uses the embedding API directly to avoid float parsing issues.
        """
        # Use embedding-level operations for this test
        x = 100.0

        for _ in range(5):
            # Add 10
            emb_x = model.linear_encoder.encode_number(x)
            emb_10 = model.linear_encoder.encode_number(10.0)
            x = model.linear_encoder.decode(emb_x + emb_10)

            # Multiply by 1.1
            log_x = model.log_encoder.encode_number(x)
            log_11 = model.log_encoder.encode_number(1.1)
            result_emb = model.log_encoder.multiply(log_x, log_11)
            x = model.log_encoder.decode(result_emb)

        # Manual calculation:
        # 100 +10=110 *1.1=121 +10=131 *1.1=144.1 +10=154.1 *1.1=169.51
        # +10=179.51 *1.1=197.461 +10=207.461 *1.1=228.2071
        expected = 228.2071
        rel_error = abs(x - expected) / expected
        assert rel_error < 0.01, f"Deep alternating chain: {x}, expected {expected}"

    def test_error_bounds_documentation(self, model):
        """
        Verify documented error bounds for composition.
        Each space crossing should add ~1e-6 error.
        Uses embedding API directly to test the actual error accumulation.
        """
        x = 1000.0
        for _ in range(10):
            # Add 1 (linear)
            emb_x = model.linear_encoder.encode_number(x)
            emb_1 = model.linear_encoder.encode_number(1.0)
            x = model.linear_encoder.decode(emb_x + emb_1)

            # Multiply by 1.001 (crossing to log)
            log_x = model.log_encoder.encode_number(x)
            log_factor = model.log_encoder.encode_number(1.001)
            x = model.log_encoder.decode(model.log_encoder.multiply(log_x, log_factor))

            # Divide by 1.001 (stay in log)
            log_x = model.log_encoder.encode_number(x)
            x = model.log_encoder.decode(model.log_encoder.divide(log_x, log_factor))

            # Subtract 1 (crossing to linear)
            emb_x = model.linear_encoder.encode_number(x)
            x = model.linear_encoder.decode(emb_x - emb_1)

        # Should be close to original 1000
        rel_error = abs(x - 1000.0) / 1000.0
        assert rel_error < 1e-3, f"After 10 round-trips: {x}, error: {rel_error}"


class TestCompositionEdgeCases:
    """Edge cases in composition."""

    def test_zero_propagation_through_multiply(self, model):
        """Zero should propagate correctly through multiply chains."""
        # (10 + (-10)) * 5 = 0
        step1 = int(model.compute("10+-10="))
        step2 = model.compute(f"{step1}*5=")
        assert abs(step2) < 0.01, f"Zero propagation: {step2}"

    def test_zero_in_addition_chain(self, model):
        """Zero should work correctly in addition chains."""
        # 0 * 5 + 10 = 10
        step1 = int(model.compute("0*5="))
        step2 = model.compute(f"{step1}+10=")
        assert abs(step2 - 10.0) < 0.01, f"0*5+10 = {step2}"

    def test_negative_through_chain(self, model):
        """Negative numbers should propagate correctly."""
        # (-10 + 5) * 2 = -10
        step1 = int(model.compute("-10+5="))
        step2 = model.compute(f"{step1}*2=")
        assert abs(step2 - (-10.0)) < 0.5, f"(-10+5)*2 = {step2}"

    def test_large_numbers_chain(self, model):
        """Large numbers should compose correctly."""
        # (10000 + 20000) * 2 = 60000
        step1 = int(model.compute("10000+20000="))
        step2 = model.compute(f"{step1}*2=")

        expected = 60000.0
        rel_error = abs(step2 - expected) / expected
        assert rel_error < 0.01, f"Large chain = {step2}, expected {expected}"

    def test_small_numbers_chain(self, model):
        """Small numbers should compose correctly using embedding API."""
        # (0.5 + 0.5) * 10 = 10
        # Use embedding API since string parser doesn't handle decimals
        emb_half = model.linear_encoder.encode_number(0.5)
        sum_emb = emb_half + emb_half
        sum_val = model.linear_encoder.decode(sum_emb)

        log_sum = model.log_encoder.encode_number(sum_val)
        log_10 = model.log_encoder.encode_number(10.0)
        result = model.log_encoder.decode(model.log_encoder.multiply(log_sum, log_10))

        expected = 10.0
        rel_error = abs(result - expected) / expected
        assert rel_error < 0.01, f"Small chain = {result}, expected {expected}"


class TestPowerComposition:
    """Tests for composition involving power operations."""

    def test_add_then_power(self, model):
        """(a + b) ** c"""
        # (2 + 2) ** 3 = 64
        step1 = int(model.compute("2+2="))
        step2 = model.compute(f"{step1}**3=")

        expected = 64.0
        rel_error = abs(step2 - expected) / expected
        assert rel_error < 0.01, f"(2+2)**3 = {step2}, expected {expected}"

    def test_power_then_add(self, model):
        """(a ** b) + c"""
        # (2 ** 4) + 10 = 26
        step1 = int(model.compute("2**4="))
        step2 = model.compute(f"{step1}+10=")

        expected = 26.0
        rel_error = abs(step2 - expected) / expected
        assert rel_error < 0.01, f"(2**4)+10 = {step2}, expected {expected}"

    def test_multiply_then_power(self, model):
        """(a * b) ** c: stays in log space"""
        # (2 * 3) ** 2 = 36
        step1 = int(model.compute("2*3="))
        step2 = model.compute(f"{step1}**2=")

        expected = 36.0
        rel_error = abs(step2 - expected) / expected
        assert rel_error < 0.01, f"(2*3)**2 = {step2}, expected {expected}"
