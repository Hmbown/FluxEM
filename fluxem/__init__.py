"""
FluxEM.

Deterministic encoders that map structured domain objects to fixed-dimensional
embeddings with closed-form operations. The package provides arithmetic
encoders (linear/logarithmic) and domain-specific encoders.

Notes
-----
Backends:

- NumPy
- JAX
- MLX

For the mathematical specification and precision notes, see:

- ``docs/FORMAL_DEFINITION.md``
- ``docs/ERROR_MODEL.md``

Examples
--------
>>> from fluxem import create_unified_model
>>> model = create_unified_model()
>>> model.compute("1847*392")
724024.0

>>> from fluxem import create_extended_ops
>>> ops = create_extended_ops()
>>> ops.sqrt(16)
4.0
>>> ops.power(2, 16)
65536.0

>>> from fluxem.backend import set_backend, BackendType
>>> set_backend(BackendType.JAX)  # or MLX, NUMPY
"""

# Backend abstraction layer
from .backend import (
    get_backend,
    set_backend,
    BackendType,
)

# Arithmetic module
from .arithmetic import (
    # Linear encoder (addition, subtraction)
    NumberEncoder,
    parse_arithmetic_expression,
    verify_linear_property,
    # Logarithmic encoder (multiplication, division)
    LogarithmicNumberEncoder,
    verify_multiplication_theorem,
    verify_division_theorem,
    # Unified model (all four operations)
    UnifiedArithmeticModel,
    create_unified_model,
    evaluate_all_operations_ood,
    # Extended operations (powers, roots, exp, ln)
    ExtendedOps,
    create_extended_ops,
)

# Core infrastructure
from .core import (
    # Constants
    EMBEDDING_DIM,
    # Domain tags
    DOMAIN_TAGS,
    get_domain_tags,
    # Protocol
    BaseEncoder,
    # Unified encoder (cross-domain)
    UnifiedEncoder,
    # Helper functions
    create_embedding,
    set_domain_tag,
    get_domain_tag_name,
    check_domain,
)

# Integration layer
from .integration.tokenizer import MultiDomainTokenizer, DomainType
from .integration.pipeline import TrainingPipeline, DomainEncoderRegistry

__version__ = "1.0.0"

__all__ = [
    # Backend
    "get_backend",
    "set_backend",
    "BackendType",
    # Linear encoder (addition, subtraction)
    "NumberEncoder",
    "parse_arithmetic_expression",
    "verify_linear_property",
    # Logarithmic encoder (multiplication, division)
    "LogarithmicNumberEncoder",
    "verify_multiplication_theorem",
    "verify_division_theorem",
    # Unified model (all four operations)
    "UnifiedArithmeticModel",
    "create_unified_model",
    "evaluate_all_operations_ood",
    # Extended operations (powers, roots, exp, ln)
    "ExtendedOps",
    "create_extended_ops",
    # Core infrastructure
    "EMBEDDING_DIM",
    "DOMAIN_TAGS",
    "get_domain_tags",
    "BaseEncoder",
    "UnifiedEncoder",
    "create_embedding",
    "set_domain_tag",
    "get_domain_tag_name",
    "check_domain",
    # Integration layer
    "MultiDomainTokenizer",
    "DomainType",
    "TrainingPipeline",
    "DomainEncoderRegistry",
]
