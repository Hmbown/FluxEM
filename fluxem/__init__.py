"""
FluxEM: Algebraic embeddings for numeric computation.

Separate rule execution from rule discovery. When the rule system is known
(or cheaply recoverable), bake it into the representation and stop wasting
model capacity relearning algebra/grammar.

Modules
-------
- arithmetic: Homomorphic embeddings for +, -, *, / with IEEE-754 precision bounds
- compositional: Oracle baseline for SCAN (demonstrates rule execution vs discovery)

Example Usage
-------------
Arithmetic:
    >>> from fluxem import create_unified_model
    >>> model = create_unified_model()
    >>> model.compute("42+58=")
    100.0
    >>> model.compute("6*7=")
    42.0

Compositional (oracle baseline):
    >>> from fluxem import AlgebraicSCANSolver
    >>> solver = AlgebraicSCANSolver()
    >>> solver.solve("jump twice")
    'I_JUMP I_JUMP'

Extended operations:
    >>> from fluxem import create_extended_ops
    >>> ops = create_extended_ops()
    >>> ops.sqrt(16)
    4.0

What This Is
------------
FluxEM is a deterministic numeric module for hybrid systems:
- Algebraic embeddings with guaranteed homomorphism properties
- A drop-in numeric primitive, not a complete reasoning system

The SCAN oracle demonstrates the same principle for composition:
once rules are known, execution is trivial. The hard problem is rule discovery.

See docs/FORMAL_DEFINITION.md for mathematical specification.
See docs/ERROR_MODEL.md for precision guarantees.
See docs/SCAN_BASELINE.md for the compositional baseline framing.
"""

# Arithmetic exports (backward compatible)
from .arithmetic import (
    NumberEncoder,
    parse_arithmetic_expression,
    verify_linear_property,
    LogarithmicNumberEncoder,
    verify_multiplication_theorem,
    verify_division_theorem,
    UnifiedArithmeticModel,
    create_unified_model,
    evaluate_all_operations_ood,
    ExtendedOps,
    create_extended_ops,
)

# Compositional exports
from .compositional import (
    AlgebraicSCANSolver,
    evaluate_accuracy,
    load_scan_split,
    load_scan_file,
    get_split_stats,
)

__version__ = "0.2.0"

__all__ = [
    # Arithmetic - Linear encoder (addition, subtraction)
    "NumberEncoder",
    "parse_arithmetic_expression",
    "verify_linear_property",
    # Arithmetic - Logarithmic encoder (multiplication, division)
    "LogarithmicNumberEncoder",
    "verify_multiplication_theorem",
    "verify_division_theorem",
    # Arithmetic - Unified model (all four operations)
    "UnifiedArithmeticModel",
    "create_unified_model",
    "evaluate_all_operations_ood",
    # Arithmetic - Extended operations (powers, roots, exp, ln)
    "ExtendedOps",
    "create_extended_ops",
    # Compositional - SCAN solver
    "AlgebraicSCANSolver",
    "evaluate_accuracy",
    "load_scan_split",
    "load_scan_file",
    "get_split_stats",
]
