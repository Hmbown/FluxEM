"""Tests for music domain encoders."""

from fluxem.domains.music import ChordEncoder


def test_chord_encoder_encode_major():
    encoder = ChordEncoder()
    emb = encoder.encode("C", quality="major")
    assert emb is not None
