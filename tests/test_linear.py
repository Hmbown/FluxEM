"""
Tests for NumberEncoder (linear embeddings for addition/subtraction).
"""

import pytest
from fluxem import NumberEncoder, verify_linear_property


@pytest.fixture
def encoder():
    """Create NumberEncoder instance for testing."""
    return NumberEncoder(dim=256, scale=100000.0, seed=42)


class TestNumberEncoder:
    """Tests for NumberEncoder."""

    def test_encode_decode_roundtrip(self, encoder):
        """Test that encoding and decoding recovers the original number."""
        test_numbers = [0, 1, -1, 42, 1000, 99999, -12345]
        for n in test_numbers:
            emb = encoder.encode_number(n)
            recovered = encoder.decode(emb)
            assert abs(recovered - n) < 0.01, f"Failed for n={n}: got {recovered}"

    def test_encode_string(self, encoder):
        """Test encoding from string."""
        test_cases = [("42", 42), ("100", 100), ("-50", -50), ("0", 0)]
        for s, expected in test_cases:
            emb = encoder.encode_string(s)
            recovered = encoder.decode(emb)
            assert abs(recovered - expected) < 0.01, f"Failed for '{s}'"

    def test_linearity_property_addition(self, encoder):
        """Test that encode(a) + encode(b) = encode(a + b)."""
        test_pairs = [
            (42, 58),
            (1000, 2000),
            (12345, 54321),
            (-100, 100),
            (99999, 1),
        ]
        for a, b in test_pairs:
            emb_a = encoder.encode_number(a)
            emb_b = encoder.encode_number(b)
            emb_sum = encoder.encode_number(a + b)
            computed = emb_a + emb_b

            # Check that the computed sum equals the direct encoding
            from fluxem.backend import get_backend
            backend = get_backend()
            error = float(backend.norm(computed - emb_sum))
            assert error < 1e-6, f"Linearity failed for {a} + {b}"

    def test_linearity_property_subtraction(self, encoder):
        """Test that encode(a) - encode(b) = encode(a - b)."""
        test_pairs = [(100, 37), (1000, 500), (50000, 25000)]
        for a, b in test_pairs:
            emb_a = encoder.encode_number(a)
            emb_b = encoder.encode_number(b)
            emb_diff = encoder.encode_number(a - b)
            computed = emb_a - emb_b

            from fluxem.backend import get_backend
            backend = get_backend()
            error = float(backend.norm(computed - emb_diff))
            assert error < 1e-6, f"Linearity failed for {a} - {b}"

    def test_verify_linear_property(self, encoder):
        """Test the verification function."""
        assert verify_linear_property(encoder, 42, 58)
        assert verify_linear_property(encoder, 1000, 2000)
        assert verify_linear_property(encoder, -100, 100)


class TestOODAddition:
    """Tests for out-of-distribution addition."""

    def test_large_numbers(self, encoder):
        """Test addition of large numbers."""
        test_cases = [
            (12345, 54321, 66666),
            (50000, 50000, 100000),
            (99999, 1, 100000),
        ]
        for a, b, expected in test_cases:
            emb_a = encoder.encode_number(a)
            emb_b = encoder.encode_number(b)
            result = encoder.decode(emb_a + emb_b)
            rel_error = abs(result - expected) / expected
            assert rel_error < 0.01, f"Failed for {a} + {b}: got {result}"
