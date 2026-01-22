"""Logic domain - propositional logic, tautology checking.

This module provides deterministic logic computations.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def parse_formula(formula: str) -> str:
    """Normalize a propositional formula string."""
    formula = formula.lower().strip()
    # Standardize operators
    formula = formula.replace("&&", " and ")
    formula = formula.replace("||", " or ")
    formula = formula.replace("!", " not ")
    formula = formula.replace("~", " not ")
    formula = formula.replace("∧", " and ")
    formula = formula.replace("∨", " or ")
    formula = formula.replace("¬", " not ")
    formula = formula.replace("->", " implies ")
    formula = formula.replace("→", " implies ")
    formula = formula.replace("<->", " iff ")
    formula = formula.replace("↔", " iff ")
    # Clean up whitespace
    formula = re.sub(r'\s+', ' ', formula)
    return formula


def get_variables(formula: str) -> Set[str]:
    """Extract propositional variables from a formula."""
    formula = parse_formula(formula)
    # Remove operators and parentheses
    cleaned = re.sub(r'\b(and|or|not|implies|iff|true|false)\b', ' ', formula)
    cleaned = re.sub(r'[()]', ' ', cleaned)
    # Extract remaining tokens as variables
    tokens = cleaned.split()
    return {t for t in tokens if t and t.isalpha()}


def evaluate_formula(formula: str, assignment: Dict[str, bool]) -> bool:
    """Evaluate a propositional formula under a truth assignment.

    Args:
        formula: Propositional formula string
        assignment: Dict mapping variable names to True/False

    Returns:
        Truth value of the formula
    """
    formula = parse_formula(formula)

    # Replace variables with their values
    for var, val in assignment.items():
        formula = re.sub(rf'\b{var}\b', str(val), formula)

    # Replace operators with Python equivalents
    formula = formula.replace(" and ", " and ")
    formula = formula.replace(" or ", " or ")
    formula = formula.replace(" not ", " not ")
    formula = formula.replace(" implies ", " <= ")  # A implies B = not A or B
    formula = formula.replace(" iff ", " == ")

    # Handle implies specially: convert "A implies B" to "(not A or B)"
    while " <= " in formula:
        # Find the pattern and replace
        formula = re.sub(r'(\w+)\s*<=\s*(\w+)', r'(not \1 or \2)', formula)

    try:
        return eval(formula, {"__builtins__": {}}, {"True": True, "False": False, "not": lambda x: not x})
    except:
        raise ValueError(f"Cannot evaluate formula: {formula}")


def is_tautology(formula: str) -> bool:
    """Check if a formula is a tautology (true under all assignments).

    Args:
        formula: Propositional formula string

    Returns:
        True if formula is a tautology
    """
    variables = get_variables(formula)
    if not variables:
        # No variables - try to evaluate directly
        try:
            return evaluate_formula(formula, {})
        except:
            return False

    var_list = sorted(variables)
    n = len(var_list)

    # Try all 2^n assignments
    for i in range(2 ** n):
        assignment = {var_list[j]: bool((i >> j) & 1) for j in range(n)}
        try:
            if not evaluate_formula(formula, assignment):
                return False
        except:
            return False
    return True


def is_contradiction(formula: str) -> bool:
    """Check if a formula is a contradiction (false under all assignments)."""
    return is_tautology(f"not ({formula})")


def is_satisfiable(formula: str) -> bool:
    """Check if a formula is satisfiable (true under some assignment)."""
    return not is_contradiction(formula)


def is_equivalent(formula1: str, formula2: str) -> bool:
    """Check if two formulas are logically equivalent."""
    return is_tautology(f"({formula1}) iff ({formula2})")


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_formula_arg(args) -> str:
    if isinstance(args, dict):
        return str(args.get("formula", args.get("expression", list(args.values())[0])))
    return str(args)


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register logic tools in the registry."""

    registry.register(ToolSpec(
        name="logic_tautology",
        function=lambda args: is_tautology(_parse_formula_arg(args)),
        description="Checks if a propositional formula is a tautology (always true).",
        parameters={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "description": "Propositional formula (e.g., 'p or not p')"}
            },
            "required": ["formula"]
        },
        returns="Boolean: True if tautology",
        examples=[
            {"input": {"formula": "p or not p"}, "output": True},
            {"input": {"formula": "p and not p"}, "output": False},
            {"input": {"formula": "(p implies q) or (q implies p)"}, "output": True},
        ],
        domain="logic",
        tags=["tautology", "propositional", "truth"],
    ))
