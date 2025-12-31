"""
Tests for LogarithmicNumberEncoder (logarithmic embeddings for multiplication/division).
"""

import pytest
from fluxem import LogarithmicNumberEncoder, verify_multiplication_theorem, verify_division_theorem


@pytest.fixture
def encoder():
    """Create LogarithmicNumberEncoder instance for testing."""
    return LogarithmicNumberEncoder(dim=256, log_scale=20.0, seed=42)


class TestLogarithmicEncoder:
    """Tests for LogarithmicNumberEncoder."""

    def test_encode_decode_roundtrip(self, encoder):
        """Test that encoding and decoding recovers the original number."""
        test_numbers = [1, 2, 10, 42, 100, 1000, 10000]
        for n in test_numbers:
            emb = encoder.encode_number(n)
            recovered = encoder.decode(emb)
            rel_error = abs(recovered - n) / n
            assert rel_error < 0.01, f"Failed for n={n}: got {recovered}"

    def test_multiplication_theorem(self, encoder):
        """Test that log_mag(a * b) = log_mag(a) + log_mag(b)."""
        test_pairs = [(6, 7), (12, 12), (100, 100), (123, 456)]
        for a, b in test_pairs:
            assert verify_multiplication_theorem(encoder, a, b)

    def test_division_theorem(self, encoder):
        """Test that log_mag(a / b) = log_mag(a) - log_mag(b)."""
        test_pairs = [(42, 6), (100, 4), (1000, 8)]
        for a, b in test_pairs:
            assert verify_division_theorem(encoder, a, b)


class TestMultiplication:
    """Tests for multiplication in embedding space."""

    def test_basic_multiplication(self, encoder):
        """Test basic multiplication."""
        test_cases = [(6, 7, 42), (12, 5, 60), (100, 50, 5000)]
        for a, b, expected in test_cases:
            emb_a = encoder.encode_number(a)
            emb_b = encoder.encode_number(b)
            result_emb = encoder.multiply(emb_a, emb_b)
            result = encoder.decode(result_emb)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {a} * {b}: got {result}"

    def test_ood_multiplication(self, encoder):
        """Test out-of-distribution multiplication."""
        test_cases = [(123, 456, 56088), (1000, 1000, 1000000)]
        for a, b, expected in test_cases:
            emb_a = encoder.encode_number(a)
            emb_b = encoder.encode_number(b)
            result_emb = encoder.multiply(emb_a, emb_b)
            result = encoder.decode(result_emb)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {a} * {b}: got {result}"


class TestDivision:
    """Tests for division in embedding space."""

    def test_basic_division(self, encoder):
        """Test basic division."""
        test_cases = [(42, 6, 7), (100, 4, 25), (1000, 8, 125)]
        for a, b, expected in test_cases:
            emb_a = encoder.encode_number(a)
            emb_b = encoder.encode_number(b)
            result_emb = encoder.divide(emb_a, emb_b)
            result = encoder.decode(result_emb)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {a} / {b}: got {result}"

    def test_ood_division(self, encoder):
        """Test out-of-distribution division."""
        test_cases = [(56088, 123, 456), (1000000, 1000, 1000)]
        for a, b, expected in test_cases:
            emb_a = encoder.encode_number(a)
            emb_b = encoder.encode_number(b)
            result_emb = encoder.divide(emb_a, emb_b)
            result = encoder.decode(result_emb)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {a} / {b}: got {result}"
