"""Tests for propositional logic encodings."""

from fluxem.domains.logic import PropositionalEncoder, PropFormula


def test_tautology_detection():
    p = PropFormula.atom("p")
    formula = p | ~p
    encoder = PropositionalEncoder()
    emb = encoder.encode(formula)
    assert encoder.is_tautology(emb)
