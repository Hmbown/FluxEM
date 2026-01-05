"""
Tests for MultiDomainTokenizer quantity detection.
"""

from fluxem.integration.tokenizer import MultiDomainTokenizer, DomainType


def test_quantity_does_not_consume_plain_words():
    tokenizer = MultiDomainTokenizer()
    tokens = tokenizer.tokenize("hello 123 world")
    assert all(token.domain != DomainType.QUANTITY for token in tokens)
    assert len(tokens) == 1
    assert tokens[0].domain == DomainType.TEXT


def test_quantity_detects_units():
    tokenizer = MultiDomainTokenizer()
    tokens = tokenizer.tokenize("speed is 10 m/s")
    domains = [token.domain for token in tokens]
    assert DomainType.QUANTITY in domains
