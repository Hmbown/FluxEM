"""Chemistry domain - molecular weights, formulas, reactions.

This module provides deterministic chemistry computations.
"""

import re
from typing import Any, Dict, List, Optional, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Atomic Data
# =============================================================================

ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.003,
    "Li": 6.941, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974, "S": 32.065, "Cl": 35.453, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.96,
    "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.96, "Tc": 98.0, "Ru": 101.07,
    "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60,
    "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Pm": 145.0, "Sm": 150.36,
    "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0, "Rn": 222.0,
}

# Common compound formulas
COMPOUND_FORMULAS = {
    "water": "H2O",
    "glucose": "C6H12O6",
    "ethanol": "C2H5OH",
    "methane": "CH4",
    "ammonia": "NH3",
    "carbon dioxide": "CO2",
    "sulfuric acid": "H2SO4",
    "hydrochloric acid": "HCl",
    "sodium chloride": "NaCl",
    "salt": "NaCl",
    "table salt": "NaCl",
    "sucrose": "C12H22O11",
    "acetic acid": "CH3COOH",
    "vinegar": "CH3COOH",
    "aspirin": "C9H8O4",
    "caffeine": "C8H10N4O2",
    "benzene": "C6H6",
    "acetone": "C3H6O",
    "nitric acid": "HNO3",
    "phosphoric acid": "H3PO4",
    "sodium hydroxide": "NaOH",
    "potassium hydroxide": "KOH",
    "calcium carbonate": "CaCO3",
    "hydrogen peroxide": "H2O2",
    "ozone": "O3",
}

# Common balanced reactions
BALANCED_REACTIONS = {
    "H2 + O2 -> H2O": "2 H2 + 1 O2 -> 2 H2O",
    "H2 + O2 = H2O": "2 H2 + 1 O2 -> 2 H2O",
    "C + O2 -> CO2": "1 C + 1 O2 -> 1 CO2",
    "N2 + H2 -> NH3": "1 N2 + 3 H2 -> 2 NH3",
    "CH4 + O2 -> CO2 + H2O": "1 CH4 + 2 O2 -> 1 CO2 + 2 H2O",
    "Fe + O2 -> Fe2O3": "4 Fe + 3 O2 -> 2 Fe2O3",
    "Na + Cl2 -> NaCl": "2 Na + 1 Cl2 -> 2 NaCl",
    "Mg + O2 -> MgO": "2 Mg + 1 O2 -> 2 MgO",
    "Al + O2 -> Al2O3": "4 Al + 3 O2 -> 2 Al2O3",
    "NaOH + HCl -> NaCl + H2O": "1 NaOH + 1 HCl -> 1 NaCl + 1 H2O",
    "Ca + H2O -> Ca(OH)2 + H2": "1 Ca + 2 H2O -> 1 Ca(OH)2 + 1 H2",
    "HCl + NaOH -> NaCl + H2O": "1 HCl + 1 NaOH -> 1 NaCl + 1 H2O",
    "CaCO3 -> CaO + CO2": "1 CaCO3 -> 1 CaO + 1 CO2",
}


# =============================================================================
# Core Functions
# =============================================================================

def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a molecular formula into element counts.

    Args:
        formula: Molecular formula string (e.g., 'H2O', 'C6H12O6', 'Ca(OH)2')

    Returns:
        Dictionary mapping elements to counts
    """
    elements = {}

    # Handle parentheses
    while '(' in formula:
        match = re.search(r'\(([^()]+)\)(\d*)', formula)
        if match:
            group = match.group(1)
            multiplier = int(match.group(2)) if match.group(2) else 1
            group_elements = parse_formula(group)
            expanded = ''.join(f"{el}{count * multiplier}" for el, count in group_elements.items())
            formula = formula[:match.start()] + expanded + formula[match.end():]
        else:
            break

    # Parse elements
    pattern = r'([A-Z][a-z]?)(\d*)'
    for match in re.finditer(pattern, formula):
        element = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        elements[element] = elements.get(element, 0) + count

    return elements


def molecular_weight(formula: str) -> float:
    """Calculate molecular weight from formula.

    Args:
        formula: Molecular formula string

    Returns:
        Molecular weight in g/mol
    """
    elements = parse_formula(formula)
    weight = 0.0
    for element, count in elements.items():
        if element in ATOMIC_WEIGHTS:
            weight += ATOMIC_WEIGHTS[element] * count
        else:
            raise ValueError(f"Unknown element: {element}")
    return round(weight, 3)


def lookup_formula(name: str) -> str:
    """Look up molecular formula by compound name.

    Args:
        name: Compound name (e.g., 'glucose', 'water')

    Returns:
        Molecular formula string
    """
    name_lower = name.lower().strip()
    if name_lower in COMPOUND_FORMULAS:
        return COMPOUND_FORMULAS[name_lower]
    raise ValueError(f"Unknown compound: {name}")


def balance_simple(reaction: str) -> str:
    """Balance a simple chemical reaction using pattern matching.

    Args:
        reaction: Reaction string (e.g., 'H2 + O2 -> H2O')

    Returns:
        Balanced reaction string
    """
    # Normalize reaction string
    reaction_clean = reaction.strip()
    reaction_clean = re.sub(r'\s+', ' ', reaction_clean)
    reaction_clean = reaction_clean.replace('=', '->')
    reaction_clean = reaction_clean.replace('→', '->')

    for pattern, balanced in BALANCED_REACTIONS.items():
        pattern_norm = pattern.replace(' ', '').replace('->', '->')
        reaction_norm = reaction_clean.replace(' ', '').replace('->', '->')
        if pattern_norm.lower() == reaction_norm.lower():
            return f"Balanced: {balanced}"

    return f"Cannot balance (not in database): {reaction}"


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_mw_arg(args):
    if isinstance(args, dict):
        return molecular_weight(args.get("formula", list(args.values())[0]))
    return molecular_weight(str(args))


def _parse_formula_arg(args):
    if isinstance(args, dict):
        return lookup_formula(args.get("name", args.get("compound", list(args.values())[0])))
    return lookup_formula(str(args))


def _parse_balance_arg(args):
    if isinstance(args, dict):
        return balance_simple(args.get("reaction", args.get("equation", list(args.values())[0])))
    return balance_simple(str(args))


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register chemistry tools in the registry."""

    registry.register(ToolSpec(
        name="chemistry_molecule",
        function=_parse_mw_arg,
        description="Calculates molecular weight from a molecular formula.",
        parameters={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "description": "Molecular formula (e.g., 'H2O', 'C6H12O6')"}
            },
            "required": ["formula"]
        },
        returns="Molecular weight in g/mol as float",
        examples=[
            {"input": {"formula": "H2O"}, "output": 18.015},
            {"input": {"formula": "C6H12O6"}, "output": 180.156},
        ],
        domain="chemistry",
        tags=["molecular", "weight", "formula"],
    ))

    registry.register(ToolSpec(
        name="chemistry_formula",
        function=_parse_formula_arg,
        description="Looks up molecular formula by compound name.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Compound name (e.g., 'glucose', 'water')"}
            },
            "required": ["name"]
        },
        returns="Molecular formula string",
        examples=[
            {"input": {"name": "glucose"}, "output": "C6H12O6"},
            {"input": {"name": "water"}, "output": "H2O"},
        ],
        domain="chemistry",
        tags=["formula", "compound", "lookup"],
    ))

    registry.register(ToolSpec(
        name="chemistry_balance_simple",
        function=_parse_balance_arg,
        description="Balances a simple chemical reaction using pattern matching.",
        parameters={
            "type": "object",
            "properties": {
                "reaction": {"type": "string", "description": "Reaction string (e.g., 'H2 + O2 -> H2O')"}
            },
            "required": ["reaction"]
        },
        returns="Balanced reaction string with coefficients",
        examples=[
            {"input": {"reaction": "H2 + O2 -> H2O"}, "output": "Balanced: 2 H2 + 1 O2 -> 2 H2O"},
        ],
        domain="chemistry",
        tags=["balance", "reaction", "equation"],
    ))
