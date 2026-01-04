"""Logic domain: Propositional logic, predicate logic, and type theory."""

from .propositional import PropositionalEncoder, PropFormula
from .predicates import PredicateEncoder, FOLFormula, Term
from .types import TypeEncoder, Type, TypeKind

__all__ = [
    # Propositional logic
    "PropositionalEncoder",
    "PropFormula",
    # First-order logic (predicates)
    "PredicateEncoder",
    "FOLFormula",
    "Term",
    # Type theory
    "TypeEncoder",
    "Type",
    "TypeKind",
]
