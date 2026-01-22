"""Cooking domain - recipe scaling, unit conversions, nutrition.

This module provides deterministic cooking computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants - Conversion factors
# =============================================================================

# Volume conversions to milliliters
VOLUME_TO_ML = {
    "cup": 236.588,
    "cups": 236.588,
    "tbsp": 14.787,
    "tablespoon": 14.787,
    "tablespoons": 14.787,
    "tsp": 4.929,
    "teaspoon": 4.929,
    "teaspoons": 4.929,
    "ml": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "l": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "fl_oz": 29.574,
    "fluid_ounce": 29.574,
    "pint": 473.176,
    "quart": 946.353,
    "gallon": 3785.41,
}

# Common ingredient densities (g/ml)
INGREDIENT_DENSITY = {
    "flour": 0.53,
    "all_purpose_flour": 0.53,
    "sugar": 0.85,
    "granulated_sugar": 0.85,
    "brown_sugar": 0.93,
    "powdered_sugar": 0.56,
    "butter": 0.91,
    "oil": 0.92,
    "vegetable_oil": 0.92,
    "olive_oil": 0.92,
    "milk": 1.03,
    "water": 1.0,
    "honey": 1.42,
    "maple_syrup": 1.37,
    "salt": 1.22,
    "cocoa_powder": 0.52,
    "rice": 0.85,
    "oats": 0.41,
}


# =============================================================================
# Core Functions
# =============================================================================

def scale_recipe(original_servings: int, desired_servings: int,
                 ingredient_amount: float) -> float:
    """Scale an ingredient amount for different serving size.

    Args:
        original_servings: Original recipe serving count
        desired_servings: Desired serving count
        ingredient_amount: Original ingredient amount

    Returns:
        Scaled ingredient amount
    """
    if original_servings <= 0:
        raise ValueError("Original servings must be positive")
    return ingredient_amount * (desired_servings / original_servings)


def cup_to_gram(cups: float, ingredient: str) -> float:
    """Convert cups to grams for a specific ingredient.

    Args:
        cups: Number of cups
        ingredient: Ingredient name (e.g., 'flour', 'sugar')

    Returns:
        Weight in grams
    """
    ingredient_lower = ingredient.lower().replace(" ", "_")
    if ingredient_lower not in INGREDIENT_DENSITY:
        raise ValueError(f"Unknown ingredient: {ingredient}. Known: {list(INGREDIENT_DENSITY.keys())}")

    density = INGREDIENT_DENSITY[ingredient_lower]
    volume_ml = cups * VOLUME_TO_ML["cup"]
    return volume_ml * density


def gram_to_cup(grams: float, ingredient: str) -> float:
    """Convert grams to cups for a specific ingredient.

    Args:
        grams: Weight in grams
        ingredient: Ingredient name

    Returns:
        Volume in cups
    """
    ingredient_lower = ingredient.lower().replace(" ", "_")
    if ingredient_lower not in INGREDIENT_DENSITY:
        raise ValueError(f"Unknown ingredient: {ingredient}")

    density = INGREDIENT_DENSITY[ingredient_lower]
    volume_ml = grams / density
    return volume_ml / VOLUME_TO_ML["cup"]


def temperature_convert(temp: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between Fahrenheit, Celsius, and Kelvin.

    Args:
        temp: Temperature value
        from_unit: Source unit ('F', 'C', or 'K')
        to_unit: Target unit ('F', 'C', or 'K')

    Returns:
        Converted temperature
    """
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()

    # Convert to Celsius first
    if from_unit == 'F':
        celsius = (temp - 32) * 5 / 9
    elif from_unit == 'K':
        celsius = temp - 273.15
    elif from_unit == 'C':
        celsius = temp
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Convert from Celsius to target
    if to_unit == 'F':
        return celsius * 9 / 5 + 32
    elif to_unit == 'K':
        return celsius + 273.15
    elif to_unit == 'C':
        return celsius
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def baking_time_adjust(original_time: float, original_temp: float,
                       new_temp: float) -> float:
    """Approximate baking time adjustment when changing temperature.

    Rule of thumb: 10°F increase → reduce time by ~5-10%

    Args:
        original_time: Original baking time (minutes)
        original_temp: Original temperature (°F)
        new_temp: New temperature (°F)

    Returns:
        Adjusted baking time (minutes)
    """
    temp_diff = new_temp - original_temp
    # Approximate: 10°F change = 7% time change
    time_factor = 1 - (temp_diff / 10) * 0.07
    return original_time * max(0.5, min(2.0, time_factor))


def yield_adjust(pan_diameter_orig: float, pan_diameter_new: float,
                 recipe_amount: float) -> float:
    """Adjust recipe for different pan size (by area).

    Args:
        pan_diameter_orig: Original pan diameter
        pan_diameter_new: New pan diameter
        recipe_amount: Original recipe amount

    Returns:
        Adjusted recipe amount
    """
    area_ratio = (pan_diameter_new / pan_diameter_orig) ** 2
    return recipe_amount * area_ratio


def protein_per_serving(total_protein: float, servings: int) -> float:
    """Calculate protein per serving.

    Args:
        total_protein: Total protein in recipe (grams)
        servings: Number of servings

    Returns:
        Protein per serving (grams)
    """
    if servings <= 0:
        raise ValueError("Servings must be positive")
    return total_protein / servings


def volume_convert(amount: float, from_unit: str, to_unit: str) -> float:
    """Convert between volume units.

    Args:
        amount: Volume amount
        from_unit: Source unit (cup, tbsp, tsp, ml, l, fl_oz, etc.)
        to_unit: Target unit

    Returns:
        Converted volume
    """
    from_unit = from_unit.lower().replace(" ", "_")
    to_unit = to_unit.lower().replace(" ", "_")

    if from_unit not in VOLUME_TO_ML:
        raise ValueError(f"Unknown volume unit: {from_unit}")
    if to_unit not in VOLUME_TO_ML:
        raise ValueError(f"Unknown volume unit: {to_unit}")

    ml = amount * VOLUME_TO_ML[from_unit]
    return ml / VOLUME_TO_ML[to_unit]


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_scale_recipe(args) -> Tuple[int, int, float]:
    if isinstance(args, dict):
        orig = int(args.get("original_servings", args.get("original")))
        desired = int(args.get("desired_servings", args.get("desired")))
        amount = float(args.get("ingredient_amount", args.get("amount")))
        return orig, desired, amount
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return int(args[0]), int(args[1]), float(args[2])
    raise ValueError(f"Cannot parse scale_recipe args: {args}")


def _parse_cup_to_gram(args) -> Tuple[float, str]:
    if isinstance(args, dict):
        cups = float(args.get("cups", args.get("amount")))
        ingredient = str(args.get("ingredient"))
        return cups, ingredient
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), str(args[1])
    raise ValueError(f"Cannot parse cup_to_gram args: {args}")


def _parse_temp_convert(args) -> Tuple[float, str, str]:
    if isinstance(args, dict):
        temp = float(args.get("temp", args.get("temperature")))
        from_u = str(args.get("from_unit", args.get("from")))
        to_u = str(args.get("to_unit", args.get("to")))
        return temp, from_u, to_u
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), str(args[1]), str(args[2])
    raise ValueError(f"Cannot parse temp_convert args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register cooking tools in the registry."""

    registry.register(ToolSpec(
        name="cooking_scale_recipe",
        function=lambda args: scale_recipe(*_parse_scale_recipe(args)),
        description="Scales an ingredient amount when changing recipe serving size.",
        parameters={
            "type": "object",
            "properties": {
                "original_servings": {"type": "integer", "description": "Original serving count"},
                "desired_servings": {"type": "integer", "description": "Desired serving count"},
                "ingredient_amount": {"type": "number", "description": "Original ingredient amount"},
            },
            "required": ["original_servings", "desired_servings", "ingredient_amount"]
        },
        returns="Scaled ingredient amount",
        examples=[
            {"input": {"original_servings": 4, "desired_servings": 6, "ingredient_amount": 2}, "output": 3.0},
        ],
        domain="cooking",
        tags=["scale", "recipe", "servings"],
    ))

    registry.register(ToolSpec(
        name="cooking_cup_to_gram",
        function=lambda args: cup_to_gram(*_parse_cup_to_gram(args)),
        description="Converts cups to grams for a specific ingredient.",
        parameters={
            "type": "object",
            "properties": {
                "cups": {"type": "number", "description": "Number of cups"},
                "ingredient": {"type": "string", "description": "Ingredient name (flour, sugar, butter, etc.)"},
            },
            "required": ["cups", "ingredient"]
        },
        returns="Weight in grams",
        examples=[
            {"input": {"cups": 1, "ingredient": "flour"}, "output": 125.4},
            {"input": {"cups": 1, "ingredient": "sugar"}, "output": 201.1},
        ],
        domain="cooking",
        tags=["conversion", "cup", "gram", "weight"],
    ))

    registry.register(ToolSpec(
        name="cooking_temperature_convert",
        function=lambda args: temperature_convert(*_parse_temp_convert(args)),
        description="Converts temperature between Fahrenheit, Celsius, and Kelvin.",
        parameters={
            "type": "object",
            "properties": {
                "temp": {"type": "number", "description": "Temperature value"},
                "from_unit": {"type": "string", "description": "Source unit (F, C, or K)"},
                "to_unit": {"type": "string", "description": "Target unit (F, C, or K)"},
            },
            "required": ["temp", "from_unit", "to_unit"]
        },
        returns="Converted temperature",
        examples=[
            {"input": {"temp": 350, "from_unit": "F", "to_unit": "C"}, "output": 176.67},
            {"input": {"temp": 180, "from_unit": "C", "to_unit": "F"}, "output": 356.0},
        ],
        domain="cooking",
        tags=["temperature", "conversion", "fahrenheit", "celsius"],
    ))

    registry.register(ToolSpec(
        name="cooking_baking_time_adjust",
        function=lambda args: baking_time_adjust(
            float(args.get("original_time", args.get("time")) if isinstance(args, dict) else args[0]),
            float(args.get("original_temp", args.get("orig_temp")) if isinstance(args, dict) else args[1]),
            float(args.get("new_temp") if isinstance(args, dict) else args[2])
        ),
        description="Adjusts baking time when changing oven temperature.",
        parameters={
            "type": "object",
            "properties": {
                "original_time": {"type": "number", "description": "Original baking time (minutes)"},
                "original_temp": {"type": "number", "description": "Original temperature (°F)"},
                "new_temp": {"type": "number", "description": "New temperature (°F)"},
            },
            "required": ["original_time", "original_temp", "new_temp"]
        },
        returns="Adjusted baking time (minutes)",
        examples=[
            {"input": {"original_time": 30, "original_temp": 350, "new_temp": 375}, "output": 27.375},
        ],
        domain="cooking",
        tags=["baking", "time", "temperature"],
    ))

    registry.register(ToolSpec(
        name="cooking_yield_adjust",
        function=lambda args: yield_adjust(
            float(args.get("pan_diameter_orig", args.get("orig_diameter")) if isinstance(args, dict) else args[0]),
            float(args.get("pan_diameter_new", args.get("new_diameter")) if isinstance(args, dict) else args[1]),
            float(args.get("recipe_amount", args.get("amount")) if isinstance(args, dict) else args[2])
        ),
        description="Adjusts recipe amount for different pan sizes (by area ratio).",
        parameters={
            "type": "object",
            "properties": {
                "pan_diameter_orig": {"type": "number", "description": "Original pan diameter"},
                "pan_diameter_new": {"type": "number", "description": "New pan diameter"},
                "recipe_amount": {"type": "number", "description": "Original recipe amount"},
            },
            "required": ["pan_diameter_orig", "pan_diameter_new", "recipe_amount"]
        },
        returns="Adjusted recipe amount",
        examples=[
            {"input": {"pan_diameter_orig": 8, "pan_diameter_new": 10, "recipe_amount": 1}, "output": 1.5625},
        ],
        domain="cooking",
        tags=["pan", "size", "yield"],
    ))

    registry.register(ToolSpec(
        name="cooking_protein_per_serving",
        function=lambda args: protein_per_serving(
            float(args.get("total_protein", args.get("protein")) if isinstance(args, dict) else args[0]),
            int(args.get("servings") if isinstance(args, dict) else args[1])
        ),
        description="Calculates protein per serving from total recipe protein.",
        parameters={
            "type": "object",
            "properties": {
                "total_protein": {"type": "number", "description": "Total protein in recipe (grams)"},
                "servings": {"type": "integer", "description": "Number of servings"},
            },
            "required": ["total_protein", "servings"]
        },
        returns="Protein per serving (grams)",
        examples=[
            {"input": {"total_protein": 60, "servings": 4}, "output": 15.0},
        ],
        domain="cooking",
        tags=["protein", "nutrition", "serving"],
    ))
