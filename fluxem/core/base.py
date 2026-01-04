"""
Base encoder protocol and domain tags for FluxEM.

All domain encoders follow this protocol to ensure consistent interfaces
and enable cross-domain composition.
"""

from typing import Any, Dict, Protocol, runtime_checkable, Optional
from dataclasses import dataclass
import math

from ..backend import get_backend

# =============================================================================
# Constants
# =============================================================================

EMBEDDING_DIM = 128  # Total embedding dimension

# Embedding layout:
# dims 0-7:    Domain tag (one-hot identifier)
# dims 8-71:   Domain-specific encoding (64 dims)
# dims 72-95:  Shared semantic features (24 dims)
# dims 96-127: Cross-domain composition (32 dims)

DOMAIN_OFFSET = 0
DOMAIN_SIZE = 8
SPECIFIC_OFFSET = 8
SPECIFIC_SIZE = 64
SHARED_OFFSET = 72
SHARED_SIZE = 24
COMPOSITION_OFFSET = 96
COMPOSITION_SIZE = 32

# Epsilon for numerical stability (must handle very small physical constants like h ~ 1e-34)
EPSILON = 1e-300

# =============================================================================
# Domain Tags (Lazy Initialization)
# =============================================================================

# Raw domain tag values - these are framework-agnostic
_DOMAIN_TAG_VALUES: Dict[str, list] = {
    # Physics domains
    "phys_quantity": [1, 0, 0, 0, 0, 0, 0, 0],
    "phys_constant": [1, 0, 0, 0, 0, 0, 0, 1],
    "phys_unit": [1, 0, 0, 0, 0, 0, 1, 0],
    # Chemistry domains
    "chem_element": [0, 1, 0, 0, 0, 0, 0, 0],
    "chem_molecule": [0, 1, 0, 0, 0, 0, 0, 1],
    "chem_reaction": [0, 1, 0, 0, 0, 0, 1, 0],
    "chem_bond": [0, 1, 0, 0, 0, 0, 1, 1],
    # Logic domains
    "logic_prop": [0, 0, 1, 0, 0, 0, 0, 0],
    "logic_pred": [0, 0, 1, 0, 0, 0, 0, 1],
    "logic_type": [0, 0, 1, 0, 0, 0, 1, 0],
    # Math domains
    "math_real": [0, 0, 0, 1, 0, 0, 0, 0],
    "math_complex": [0, 0, 0, 1, 0, 0, 0, 1],
    "math_rational": [0, 0, 0, 1, 0, 0, 1, 0],
    "math_polynomial": [0, 0, 0, 1, 0, 0, 1, 1],
    "math_vector": [0, 0, 0, 1, 0, 1, 0, 0],
    "math_matrix": [0, 0, 0, 1, 0, 1, 0, 1],
    # Biology domains
    "bio_dna": [0, 0, 0, 0, 1, 0, 0, 0],
    "bio_rna": [0, 0, 0, 0, 1, 0, 0, 1],
    "bio_protein": [0, 0, 0, 0, 1, 0, 1, 0],
    "bio_gene": [0, 0, 0, 0, 1, 0, 1, 1],
    "bio_pathway": [0, 0, 0, 0, 1, 1, 0, 0],
    "bio_taxonomy": [0, 0, 0, 0, 1, 1, 0, 1],
    # Music domains
    "music_pitch": [0, 0, 0, 0, 0, 1, 0, 0],
    "music_chord": [0, 0, 0, 0, 0, 1, 0, 1],
    "music_scale": [0, 0, 0, 0, 0, 1, 1, 0],
    "music_rhythm": [0, 0, 0, 0, 0, 1, 1, 1],
    "music_atonal": [0, 0, 0, 0, 0, 1, 1, 0.5],  # Different from scale
    # Set theory domains
    "set_finite": [0, 0, 1, 0, 0, 0, 1, 1],
    "set_relation": [0, 0, 1, 1, 0, 0, 0, 0],
    "set_function": [0, 0, 1, 1, 0, 0, 0, 1],
    # Graph theory domains
    "graph_directed": [1, 1, 0, 0, 0, 0, 0, 0],
    "graph_undirected": [1, 1, 0, 0, 0, 0, 0, 1],
    "graph_weighted": [1, 1, 0, 0, 0, 0, 1, 0],
    "graph_tree": [1, 1, 0, 0, 0, 0, 1, 1],
    "graph_dag": [1, 1, 0, 0, 0, 1, 0, 0],
    # Geometry domains
    "geom_point2d": [1, 0, 1, 0, 0, 0, 0, 0],
    "geom_point3d": [1, 0, 1, 0, 0, 0, 0, 1],
    "geom_vector2d": [1, 0, 1, 0, 0, 0, 1, 0],
    "geom_vector3d": [1, 0, 1, 0, 0, 0, 1, 1],
    "geom_transform2d": [1, 0, 1, 0, 0, 1, 0, 0],
    "geom_transform3d": [1, 0, 1, 0, 0, 1, 0, 1],
    "geom_triangle": [1, 0, 1, 0, 0, 1, 1, 0],
    "geom_rectangle": [1, 0, 1, 0, 0, 1, 1, 1],
    "geom_circle": [1, 0, 1, 0, 1, 0, 0, 0],
    "geom_polygon": [1, 0, 1, 0, 1, 0, 0, 1],
    # Number theory domains
    "num_integer": [1, 0, 0, 1, 0, 0, 0, 0],
    "num_prime": [1, 0, 0, 1, 0, 0, 0, 1],
    "num_modular": [1, 0, 0, 1, 0, 0, 1, 0],
    "num_rational": [1, 0, 0, 1, 0, 0, 1, 1],
}

# Cached domain tags (lazy initialized)
_DOMAIN_TAGS: Optional[Dict[str, Any]] = None
_DOMAIN_TAGS_BACKEND_NAME: Optional[str] = None


def get_domain_tags() -> Dict[str, Any]:
    """
    Get domain tags as backend arrays (lazy initialization).

    This defers array creation until first use, allowing the backend
    to be set before any arrays are created.

    Returns:
        Dictionary mapping domain names to backend arrays.
    """
    global _DOMAIN_TAGS
    global _DOMAIN_TAGS_BACKEND_NAME
    backend = get_backend()
    if _DOMAIN_TAGS is None or _DOMAIN_TAGS_BACKEND_NAME != getattr(backend, "name", None):
        _DOMAIN_TAGS = {
            name: backend.array(values)
            for name, values in _DOMAIN_TAG_VALUES.items()
        }
        _DOMAIN_TAGS_BACKEND_NAME = getattr(backend, "name", None)
    return _DOMAIN_TAGS


# For backward compatibility, expose DOMAIN_TAGS as a property-like access
# Note: This will trigger lazy initialization on first access
class _DomainTagsProxy:
    """Proxy class that lazily initializes domain tags on access."""

    def __getitem__(self, key: str) -> Any:
        return get_domain_tags()[key]

    def __contains__(self, key: str) -> bool:
        return key in _DOMAIN_TAG_VALUES

    def __iter__(self):
        return iter(_DOMAIN_TAG_VALUES)

    def keys(self):
        return _DOMAIN_TAG_VALUES.keys()

    def values(self):
        return get_domain_tags().values()

    def items(self):
        return get_domain_tags().items()

    def get(self, key: str, default: Any = None) -> Any:
        tags = get_domain_tags()
        return tags.get(key, default)


DOMAIN_TAGS = _DomainTagsProxy()


# =============================================================================
# Base Encoder Protocol
# =============================================================================


@runtime_checkable
class BaseEncoder(Protocol):
    """
    Protocol that all domain encoders must follow.

    Each encoder transforms domain-specific values into 128-dimensional
    embeddings where algebraic operations become geometric transformations.
    """

    @property
    def domain_tag(self) -> Any:
        """8-dimensional one-hot domain identifier."""
        ...

    @property
    def domain_name(self) -> str:
        """Human-readable domain name."""
        ...

    def encode(self, value: Any) -> Any:
        """
        Encode a domain value to a 128-dim embedding.

        Args:
            value: Domain-specific value to encode

        Returns:
            128-dimensional array
        """
        ...

    def decode(self, emb: Any) -> Any:
        """
        Decode a 128-dim embedding back to domain value.

        Args:
            emb: 128-dimensional embedding

        Returns:
            Domain-specific value
        """
        ...

    def is_valid(self, emb: Any) -> bool:
        """
        Check if an embedding is valid for this domain.

        Args:
            emb: Embedding to validate

        Returns:
            True if the embedding has the correct domain tag and structure
        """
        ...


# =============================================================================
# Helper Functions
# =============================================================================


def create_embedding() -> Any:
    """Create a zero-initialized 128-dim embedding."""
    backend = get_backend()
    return backend.zeros(EMBEDDING_DIM)


def set_domain_tag(emb: Any, tag_name: str) -> Any:
    """Set the domain tag on an embedding."""
    backend = get_backend()

    if tag_name not in _DOMAIN_TAG_VALUES:
        raise ValueError(f"Unknown domain tag: {tag_name}")

    tag = get_domain_tags()[tag_name]
    result = emb

    # Set tag values using immutable updates
    for i in range(DOMAIN_SIZE):
        result = backend.at_add(result, i, float(tag[i]) - float(result[i]))

    return result


def get_domain_tag_name(emb: Any) -> str:
    """Extract the domain tag name from an embedding."""
    backend = get_backend()
    tag_slice = emb[DOMAIN_OFFSET:DOMAIN_OFFSET + DOMAIN_SIZE]

    for name, tag in get_domain_tags().items():
        result = backend.allclose(tag_slice, tag, atol=0.1)
        # Handle both bool (NumPy) and array (JAX/MLX) return types
        is_match = result.item() if hasattr(result, 'item') else result
        if is_match:
            return name
    return "unknown"


def check_domain(emb: Any, expected: str) -> bool:
    """Check if embedding has the expected domain tag."""
    return get_domain_tag_name(emb) == expected


def get_specific_slice(emb: Any) -> Any:
    """Get the domain-specific portion of the embedding (dims 8-71)."""
    return emb[SPECIFIC_OFFSET:SPECIFIC_OFFSET + SPECIFIC_SIZE]


def get_shared_slice(emb: Any) -> Any:
    """Get the shared semantic features (dims 72-95)."""
    return emb[SHARED_OFFSET:SHARED_OFFSET + SHARED_SIZE]


def get_composition_slice(emb: Any) -> Any:
    """Get the cross-domain composition features (dims 96-127)."""
    return emb[COMPOSITION_OFFSET:COMPOSITION_OFFSET + COMPOSITION_SIZE]


# =============================================================================
# Log Encoding Utilities (shared across domains)
# =============================================================================


def log_encode_value(value: float) -> tuple:
    """
    Encode a real number using log-magnitude representation.

    Returns:
        (sign, log_magnitude) tuple where:
        - sign is +1, -1, or 0
        - log_magnitude is log(|value|) or -inf sentinel for zero
    """
    if abs(value) < EPSILON:
        return (0.0, -100.0)  # Zero sentinel

    sign = 1.0 if value > 0 else -1.0
    log_mag = math.log(abs(value))
    return (sign, log_mag)


def log_decode_value(sign: float, log_mag: float) -> float:
    """
    Decode a log-magnitude representation back to a real number.

    Args:
        sign: +1, -1, or 0
        log_mag: log(|value|) or zero sentinel (-100)

    Returns:
        The decoded real number
    """
    if abs(sign) < 0.5:
        return 0.0

    magnitude = math.exp(log_mag)
    return sign * magnitude


@dataclass
class EncodingError:
    """Represents an encoding/decoding error with context."""

    domain: str
    operation: str
    message: str
    value: Any = None
