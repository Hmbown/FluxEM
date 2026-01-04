"""
FluxEM: Algebraic embeddings for exact neural computation.

Structured embeddings where algebraic operations become geometric transformations:
- Addition becomes vector addition
- Multiplication becomes addition in log-space
- Systematic generalization via algebraic structure (parameter-free)

Supports 11 scientific domains:
- Physics: quantities, constants, units
- Chemistry: elements, molecules, reactions, bonds
- Biology: DNA, RNA, proteins, genes, pathways, taxonomy
- Math: reals, complex, rationals, polynomials, vectors, matrices
- Logic: propositional, predicate, type theory
- Music: pitch, chords, scales, rhythm
- Geometry: points, vectors, transforms, shapes
- Graphs: directed, undirected, weighted, trees, DAGs
- Sets: finite sets, relations, functions
- Number Theory: integers, primes, modular arithmetic

Example Usage
-------------
>>> from fluxem import create_unified_model
>>> model = create_unified_model()
>>> model.compute("1847*392")
724024.0
>>> model.compute("123456+789")
124245.0

Extended operations:
>>> from fluxem import create_extended_ops
>>> ops = create_extended_ops()
>>> ops.sqrt(16)
4.0
>>> ops.power(2, 16)
65536.0

Backend selection:
>>> from fluxem.backend import set_backend, BackendType
>>> set_backend(BackendType.JAX)  # or MLX, NUMPY

How It Works
------------
Linear embeddings (addition/subtraction):
    embed(a) + embed(b) = embed(a + b)

Log embeddings (multiplication/division):
    log_embed(a) + log_embed(b) = log_embed(a * b)

Arithmetic operations map to geometric operations in embedding space.
See docs/FORMAL_DEFINITION.md for mathematical specification.
See docs/ERROR_MODEL.md for precision notes.
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
