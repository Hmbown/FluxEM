"""
Unified encoder that routes to domain-specific encoders and handles
cross-domain composition.
"""

from typing import Any, Dict, Optional, Type

from ..backend import get_backend
from .base import (
    BaseEncoder,
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    get_domain_tag_name,
    check_domain,
)


class UnifiedEncoder:
    """
    Unified encoder that manages multiple domain encoders.

    Provides:
    - Automatic domain detection and routing
    - Cross-domain composition operations
    - Batch encoding/decoding
    """

    def __init__(self):
        self._encoders: Dict[str, BaseEncoder] = {}

    def register(self, domain_name: str, encoder: BaseEncoder) -> None:
        """
        Register a domain encoder.

        Args:
            domain_name: Name matching a key in DOMAIN_TAGS
            encoder: Encoder instance implementing BaseEncoder protocol
        """
        if domain_name not in DOMAIN_TAGS:
            raise ValueError(f"Unknown domain: {domain_name}")
        self._encoders[domain_name] = encoder

    def get_encoder(self, domain_name: str) -> Optional[BaseEncoder]:
        """Get the encoder for a specific domain."""
        return self._encoders.get(domain_name)

    def encode(self, value: Any, domain: str) -> Any:
        """
        Encode a value using the specified domain encoder.

        Args:
            value: Value to encode
            domain: Domain name

        Returns:
            128-dim embedding
        """
        encoder = self._encoders.get(domain)
        if encoder is None:
            raise ValueError(f"No encoder registered for domain: {domain}")
        return encoder.encode(value)

    def decode(self, emb: Any, domain: Optional[str] = None) -> Any:
        """
        Decode an embedding back to its domain value.

        Args:
            emb: 128-dim embedding
            domain: Optional domain name. If not provided, inferred from tag.

        Returns:
            Domain-specific value
        """
        if domain is None:
            domain = get_domain_tag_name(emb)

        encoder = self._encoders.get(domain)
        if encoder is None:
            raise ValueError(f"No encoder registered for domain: {domain}")
        return encoder.decode(emb)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is valid for its tagged domain."""
        domain = get_domain_tag_name(emb)
        encoder = self._encoders.get(domain)
        if encoder is None:
            return False
        return encoder.is_valid(emb)

    def same_domain(self, emb1: Any, emb2: Any) -> bool:
        """Check if two embeddings are from the same domain."""
        return get_domain_tag_name(emb1) == get_domain_tag_name(emb2)

    @property
    def registered_domains(self) -> list:
        """List of registered domain names."""
        return list(self._encoders.keys())


# =============================================================================
# Cross-Domain Composition Functions
# =============================================================================


def compose_embeddings(emb1: Any, emb2: Any, operation: str) -> Any:
    """
    Compose two embeddings from potentially different domains.

    This is used when domains interact, e.g., chemistry using math
    for stoichiometry calculations.

    Args:
        emb1: First embedding
        emb2: Second embedding
        operation: Composition operation name

    Returns:
        Composed embedding
    """
    backend = get_backend()

    # Store composition info in the cross-domain section
    result = backend.zeros(EMBEDDING_DIM)

    # Copy primary embedding (first 72 dims)
    for i in range(72):
        result = backend.at_add(result, i, float(emb1[i]))

    # Store composition metadata
    result = backend.at_add(result, 96, 1.0)  # Mark as composed

    # Store second domain tag (8 dims)
    for i in range(8):
        result = backend.at_add(result, 97 + i, float(emb2[i]))

    return result


def extract_numeric_component(emb: Any) -> tuple:
    """
    Extract numeric value from any domain embedding that has magnitude.

    Returns (sign, log_magnitude) if present, (0, 0) otherwise.
    """
    domain = get_domain_tag_name(emb)

    # Physics quantities have magnitude at specific offset
    if domain.startswith("phys_"):
        return (float(emb[16]), float(emb[17]))

    # Math domains encode magnitude similarly
    if domain.startswith("math_"):
        return (float(emb[8]), float(emb[9]))

    return (0.0, 0.0)
