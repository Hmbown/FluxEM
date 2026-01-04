"""
Set Theory Module for FluxEM-Domains.

Provides exact algebraic embeddings for set-theoretic objects:
- Finite sets with exact membership, subset, union, intersection operations
- Relations (binary relations as sets of pairs)
- Functions (as special relations with uniqueness)

All operations are EXACT by construction - no approximation.
"""

from .sets import (
    FiniteSet,
    SetEncoder,
    SetType,
)
from .relations import (
    Relation,
    RelationEncoder,
    RelationType,
)
from .functions import (
    Function,
    FunctionEncoder,
    FunctionType,
)

__all__ = [
    # Sets
    "FiniteSet",
    "SetEncoder",
    "SetType",
    # Relations
    "Relation",
    "RelationEncoder",
    "RelationType",
    # Functions
    "Function",
    "FunctionEncoder",
    "FunctionType",
]
