import math

import pytest

from fluxem import create_extended_ops, create_unified_model
from fluxem.backend import get_backend
from fluxem.core.base import log_encode_value, log_decode_value, get_domain_tags
from fluxem.core import get_domain_tag_name
from fluxem.domains.music.atonal import row_matrix
from fluxem.domains.music.atonal import AtonalSetEncoder
from fluxem.domains.chemistry.molecules import Formula


def test_unified_model_compute_basic_ops():
    model = create_unified_model()

    assert model.compute("1234 + 5678") == pytest.approx(6912.0, rel=1e-4)
    assert model.compute("250 * 4") == pytest.approx(1000.0, rel=1e-4)
    assert model.compute("1000 / 8") == pytest.approx(125.0, rel=1e-4)
    assert model.compute("3 ** 4") == pytest.approx(81.0, rel=1e-4)


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


def test_log_encode_decode_tiny_nonzero_roundtrip():
    # This value is far below exp(-99) (~1e-43) but far above EPSILON (1e-300).
    x = 1e-200
    sign, log_mag = log_encode_value(x)
    y = log_decode_value(sign, log_mag)
    assert y == pytest.approx(x, rel=1e-12)


def test_music_atonal_row_matrix_smoke():
    m = row_matrix(list(range(12)))
    assert m.shape == (12, 12)


def test_music_domain_tags_are_distinct():
    enc = AtonalSetEncoder()
    emb = enc.encode([0, 4, 7])
    assert get_domain_tag_name(emb) == "music_atonal"


def test_chemistry_formula_parse_parentheses():
    f = Formula.parse("Ca(OH)2")
    assert f.composition.get("Ca") == 1
    assert f.composition.get("O") == 2
    assert f.composition.get("H") == 2


def test_domain_tags_unique():
    backend = get_backend()
    tags = get_domain_tags()
    names = list(tags.keys())
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            result = backend.allclose(tags[name], tags[other], atol=0.1)
            # Handle both bool (NumPy) and array (JAX/MLX) return types
            is_match = result.item() if hasattr(result, 'item') else result
            if is_match:
                pytest.fail(f"Domain tags collide: {name} vs {other}")
