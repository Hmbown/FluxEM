"""Physics domain - units, dimensions, conversions.

This module provides deterministic physics computations for units and dimensions.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# SI Base Dimensions
# =============================================================================

SI_DIMENSIONS = {
    "L": "length",
    "M": "mass",
    "T": "time",
    "I": "electric current",
    "Θ": "temperature",
    "N": "amount of substance",
    "J": "luminous intensity",
}

# Unit to SI dimension mapping
UNIT_DIMENSIONS = {
    # Length
    "m": {"L": 1},
    "meter": {"L": 1},
    "meters": {"L": 1},
    "km": {"L": 1},
    "kilometer": {"L": 1},
    "kilometers": {"L": 1},
    "cm": {"L": 1},
    "mm": {"L": 1},
    "ft": {"L": 1},
    "feet": {"L": 1},
    "foot": {"L": 1},
    "in": {"L": 1},
    "inch": {"L": 1},
    "inches": {"L": 1},
    "mi": {"L": 1},
    "mile": {"L": 1},
    "miles": {"L": 1},
    "yd": {"L": 1},
    "yard": {"L": 1},
    "yards": {"L": 1},
    # Mass
    "kg": {"M": 1},
    "kilogram": {"M": 1},
    "kilograms": {"M": 1},
    "g": {"M": 1},
    "gram": {"M": 1},
    "grams": {"M": 1},
    "mg": {"M": 1},
    "lb": {"M": 1},
    "pound": {"M": 1},
    "pounds": {"M": 1},
    "oz": {"M": 1},
    "ounce": {"M": 1},
    "ounces": {"M": 1},
    # Time
    "s": {"T": 1},
    "sec": {"T": 1},
    "second": {"T": 1},
    "seconds": {"T": 1},
    "min": {"T": 1},
    "minute": {"T": 1},
    "minutes": {"T": 1},
    "h": {"T": 1},
    "hr": {"T": 1},
    "hour": {"T": 1},
    "hours": {"T": 1},
    "day": {"T": 1},
    "days": {"T": 1},
    # Velocity
    "m/s": {"L": 1, "T": -1},
    "km/h": {"L": 1, "T": -1},
    "mph": {"L": 1, "T": -1},
    "ft/s": {"L": 1, "T": -1},
    # Acceleration
    "m/s^2": {"L": 1, "T": -2},
    "m/s2": {"L": 1, "T": -2},
    # Force
    "N": {"M": 1, "L": 1, "T": -2},
    "newton": {"M": 1, "L": 1, "T": -2},
    "newtons": {"M": 1, "L": 1, "T": -2},
    "kN": {"M": 1, "L": 1, "T": -2},
    "lbf": {"M": 1, "L": 1, "T": -2},
    # Energy
    "J": {"M": 1, "L": 2, "T": -2},
    "joule": {"M": 1, "L": 2, "T": -2},
    "joules": {"M": 1, "L": 2, "T": -2},
    "kJ": {"M": 1, "L": 2, "T": -2},
    "cal": {"M": 1, "L": 2, "T": -2},
    "kcal": {"M": 1, "L": 2, "T": -2},
    "eV": {"M": 1, "L": 2, "T": -2},
    "kWh": {"M": 1, "L": 2, "T": -2},
    # Power
    "W": {"M": 1, "L": 2, "T": -3},
    "watt": {"M": 1, "L": 2, "T": -3},
    "watts": {"M": 1, "L": 2, "T": -3},
    "kW": {"M": 1, "L": 2, "T": -3},
    "MW": {"M": 1, "L": 2, "T": -3},
    "hp": {"M": 1, "L": 2, "T": -3},
    # Pressure
    "Pa": {"M": 1, "L": -1, "T": -2},
    "pascal": {"M": 1, "L": -1, "T": -2},
    "kPa": {"M": 1, "L": -1, "T": -2},
    "MPa": {"M": 1, "L": -1, "T": -2},
    "bar": {"M": 1, "L": -1, "T": -2},
    "atm": {"M": 1, "L": -1, "T": -2},
    "psi": {"M": 1, "L": -1, "T": -2},
    # Frequency
    "Hz": {"T": -1},
    "hertz": {"T": -1},
    "kHz": {"T": -1},
    "MHz": {"T": -1},
    "GHz": {"T": -1},
    # Electric
    "A": {"I": 1},
    "ampere": {"I": 1},
    "V": {"M": 1, "L": 2, "T": -3, "I": -1},
    "volt": {"M": 1, "L": 2, "T": -3, "I": -1},
    "Ω": {"M": 1, "L": 2, "T": -3, "I": -2},
    "ohm": {"M": 1, "L": 2, "T": -3, "I": -2},
    "C": {"T": 1, "I": 1},
    "coulomb": {"T": 1, "I": 1},
    "F": {"M": -1, "L": -2, "T": 4, "I": 2},
    "farad": {"M": -1, "L": -2, "T": 4, "I": 2},
    # Temperature
    "K": {"Θ": 1},
    "kelvin": {"Θ": 1},
    # Area/Volume
    "m^2": {"L": 2},
    "m2": {"L": 2},
    "m^3": {"L": 3},
    "m3": {"L": 3},
    "L": {"L": 3},
    "liter": {"L": 3},
    "liters": {"L": 3},
    "mL": {"L": 3},
    "gal": {"L": 3},
    "gallon": {"L": 3},
    "gallons": {"L": 3},
}

# Conversion factors to SI base units
CONVERSION_TO_SI = {
    # Length (to meters)
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "mm": 0.001,
    "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    # Mass (to kg)
    "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gram": 0.001, "grams": 0.001,
    "mg": 1e-6,
    "lb": 0.453592, "pound": 0.453592, "pounds": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
    # Time (to seconds)
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "day": 86400.0, "days": 86400.0,
    # Velocity
    "m/s": 1.0,
    "km/h": 1/3.6,
    "mph": 0.44704,
    "ft/s": 0.3048,
    # Force (to Newtons)
    "N": 1.0, "newton": 1.0, "newtons": 1.0,
    "kN": 1000.0,
    "lbf": 4.44822,
    # Energy (to Joules)
    "J": 1.0, "joule": 1.0, "joules": 1.0,
    "kJ": 1000.0,
    "cal": 4.184,
    "kcal": 4184.0,
    "eV": 1.602e-19,
    "kWh": 3.6e6,
    # Power (to Watts)
    "W": 1.0, "watt": 1.0, "watts": 1.0,
    "kW": 1000.0,
    "MW": 1e6,
    "hp": 745.7,
    # Pressure (to Pascals)
    "Pa": 1.0, "pascal": 1.0,
    "kPa": 1000.0,
    "MPa": 1e6,
    "bar": 1e5,
    "atm": 101325.0,
    "psi": 6894.76,
    # Frequency (to Hz)
    "Hz": 1.0, "hertz": 1.0,
    "kHz": 1000.0,
    "MHz": 1e6,
    "GHz": 1e9,
    # Volume (to m^3)
    "m^3": 1.0, "m3": 1.0,
    "L": 0.001, "liter": 0.001, "liters": 0.001,
    "mL": 1e-6,
    "gal": 0.003785, "gallon": 0.003785, "gallons": 0.003785,
    # Area
    "m^2": 1.0, "m2": 1.0,
}


def get_dimensions(unit_str: str) -> Dict[str, int]:
    """Get SI dimensions for a unit string."""
    unit = unit_str.strip().lower()
    # Handle common aliases
    unit = unit.replace("meter", "m").replace("kilogram", "kg").replace("second", "s")

    dims = UNIT_DIMENSIONS.get(unit) or UNIT_DIMENSIONS.get(unit_str)
    if dims:
        return {k: v for k, v in dims.items() if v != 0}

    # Try parsing compound units
    dims = _parse_compound_unit(unit_str)
    return {k: v for k, v in dims.items() if v != 0}


def _parse_compound_unit(unit_str: str) -> Dict[str, int]:
    """Parse compound units like m/s^2, kg*m/s."""
    result = {}
    # Split by / and *
    parts = re.split(r'[/\*]', unit_str)
    operators = re.findall(r'[/\*]', unit_str)

    for i, part in enumerate(parts):
        exp_match = re.match(r'(\w+)\^?(-?\d+)?', part.strip())
        if exp_match:
            base_unit = exp_match.group(1)
            exp = int(exp_match.group(2) or 1)

            if i > 0 and i - 1 < len(operators) and operators[i - 1] == '/':
                exp = -exp

            base_dims = UNIT_DIMENSIONS.get(base_unit, {})
            for dim, val in base_dims.items():
                result[dim] = result.get(dim, 0) + val * exp

    return result


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between compatible physical units."""
    from_dims = get_dimensions(from_unit)
    to_dims = get_dimensions(to_unit)

    # Check dimension compatibility
    if from_dims != to_dims:
        raise ValueError(f"Incompatible dimensions: {from_unit} ({from_dims}) vs {to_unit} ({to_dims})")

    from_factor = CONVERSION_TO_SI.get(from_unit, 1.0)
    to_factor = CONVERSION_TO_SI.get(to_unit, 1.0)

    return value * from_factor / to_factor


def can_add_units(unit1: str, unit2: str) -> bool:
    """Check if two units have compatible dimensions for addition."""
    dims1 = get_dimensions(unit1)
    dims2 = get_dimensions(unit2)
    return dims1 == dims2


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_dimensions_arg(args):
    if isinstance(args, dict):
        return get_dimensions(args.get("unit", args.get("unit_str", str(list(args.values())[0]))))
    return get_dimensions(str(args))


def _parse_convert_arg(args):
    if isinstance(args, dict):
        value = float(args.get("value", 1.0))
        from_unit = args.get("from_unit", args.get("from", args.get("source")))
        to_unit = args.get("to_unit", args.get("to", args.get("target")))
        return convert_units(value, from_unit, to_unit)

    if isinstance(args, str):
        # Parse "5 km to meters" format
        match = re.match(r'(-?\d+\.?\d*)\s*(\S+)\s+to\s+(\S+)', args, re.IGNORECASE)
        if match:
            return convert_units(float(match.group(1)), match.group(2), match.group(3))

    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return convert_units(float(args[0]), str(args[1]), str(args[2]))

    raise ValueError(f"Cannot parse conversion arguments: {args}")


def _parse_can_add_arg(args):
    if isinstance(args, dict):
        unit1 = args.get("unit1", args.get("a"))
        unit2 = args.get("unit2", args.get("b"))
        return can_add_units(unit1, unit2)

    if isinstance(args, str):
        # Parse "5 m and 3 km" or "m and km" format
        match = re.search(r'(\d*\.?\d*)\s*(\S+)\s+and\s+(\d*\.?\d*)\s*(\S+)', args, re.IGNORECASE)
        if match:
            unit1 = match.group(2)
            unit2 = match.group(4)
            return can_add_units(unit1, unit2)

    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return can_add_units(str(args[0]), str(args[1]))

    raise ValueError(f"Cannot parse can_add arguments: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register physics tools in the registry."""

    registry.register(ToolSpec(
        name="physics_dimensions",
        function=_parse_dimensions_arg,
        description="Extracts SI base dimensions from a unit string.",
        parameters={
            "type": "object",
            "properties": {
                "unit": {"type": "string", "description": "Unit string (e.g., 'm/s', 'kg*m/s^2', 'N')"}
            },
            "required": ["unit"]
        },
        returns="Dictionary of SI exponents (e.g., {'L': 1, 'T': -1})",
        examples=[
            {"input": {"unit": "m/s"}, "output": {"L": 1, "T": -1}},
            {"input": {"unit": "N"}, "output": {"M": 1, "L": 1, "T": -2}},
        ],
        domain="physics",
        tags=["dimensions", "units", "si"],
    ))

    registry.register(ToolSpec(
        name="physics_convert",
        function=_parse_convert_arg,
        description="Converts between compatible physical units.",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit": {"type": "string", "description": "Target unit"},
            },
            "required": ["value", "from_unit", "to_unit"]
        },
        returns="Converted value as float",
        examples=[
            {"input": {"value": 5, "from_unit": "km", "to_unit": "m"}, "output": 5000.0},
            {"input": {"value": 100, "from_unit": "mph", "to_unit": "km/h"}, "output": 160.934},
        ],
        domain="physics",
        tags=["convert", "units", "measurement"],
    ))

    registry.register(ToolSpec(
        name="physics_can_add",
        function=_parse_can_add_arg,
        description="Checks if two physical quantities can be added (same dimensions).",
        parameters={
            "type": "object",
            "properties": {
                "unit1": {"type": "string", "description": "First unit"},
                "unit2": {"type": "string", "description": "Second unit"},
            },
            "required": ["unit1", "unit2"]
        },
        returns="Boolean (True if units are compatible for addition)",
        examples=[
            {"input": {"unit1": "m", "unit2": "km"}, "output": True},
            {"input": {"unit1": "kg", "unit2": "m"}, "output": False},
        ],
        domain="physics",
        tags=["dimensions", "compatibility", "units"],
    ))
