"""
FluxEM Core: Base encoders and domain infrastructure.

This module provides the foundation for all domain encoders:
- BaseEncoder protocol that all encoders must implement
- Domain tags for identifying embedding types
- Unified encoder for cross-domain operations
- Helper functions for embedding manipulation
"""

from .base import (
    # Constants
    EMBEDDING_DIM,
    DOMAIN_OFFSET,
    DOMAIN_SIZE,
    SPECIFIC_OFFSET,
    SPECIFIC_SIZE,
    SHARED_OFFSET,
    SHARED_SIZE,
    COMPOSITION_OFFSET,
    COMPOSITION_SIZE,
    EPSILON,
    # Domain tags
    DOMAIN_TAGS,
    get_domain_tags,
    # Protocol
    BaseEncoder,
    EncodingError,
    # Helper functions
    create_embedding,
    set_domain_tag,
    get_domain_tag_name,
    check_domain,
    get_specific_slice,
    get_shared_slice,
    get_composition_slice,
    # Log encoding utilities
    log_encode_value,
    log_decode_value,
)
from .unified import (
    UnifiedEncoder,
    compose_embeddings,
    extract_numeric_component,
)

__all__ = [
    # Constants
    "EMBEDDING_DIM",
    "DOMAIN_OFFSET",
    "DOMAIN_SIZE",
    "SPECIFIC_OFFSET",
    "SPECIFIC_SIZE",
    "SHARED_OFFSET",
    "SHARED_SIZE",
    "COMPOSITION_OFFSET",
    "COMPOSITION_SIZE",
    "EPSILON",
    # Domain tags
    "DOMAIN_TAGS",
    "get_domain_tags",
    # Protocol
    "BaseEncoder",
    "EncodingError",
    # Helper functions
    "create_embedding",
    "set_domain_tag",
    "get_domain_tag_name",
    "check_domain",
    "get_specific_slice",
    "get_shared_slice",
    "get_composition_slice",
    # Log encoding utilities
    "log_encode_value",
    "log_decode_value",
    # Unified encoder
    "UnifiedEncoder",
    "compose_embeddings",
    "extract_numeric_component",
]
