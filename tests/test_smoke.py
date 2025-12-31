import math

import pytest

from fluxem import create_extended_ops, create_unified_model


def test_unified_model_compute_basic_ops():
    model = create_unified_model()

    assert model.compute("1234 + 5678") == 6912.0
    assert model.compute("250 * 4") == 1000.0
    assert model.compute("1000 / 8") == 125.0
    assert model.compute("3 ** 4") == 81.0


def test_extended_ops_power_and_sqrt():
    ops = create_extended_ops()

    assert ops.power(2, 16) == pytest.approx(65536.0, rel=1e-4)
    assert ops.sqrt(256) == pytest.approx(16.0, rel=1e-4)


def test_extended_ops_edge_cases():
    ops = create_extended_ops()

    assert ops.sqrt(-4) == pytest.approx(2.0, rel=1e-4)
    assert ops.sqrt(0) == 0.0
    assert math.isinf(ops.ln(0))
    assert math.isinf(ops.ln(-4))
