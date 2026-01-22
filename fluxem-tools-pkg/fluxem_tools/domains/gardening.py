"""Gardening domain - soil, plants, watering calculations.

This module provides deterministic gardening/horticulture computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def soil_volume_bed(length: float, width: float, depth: float) -> float:
    """Calculate soil volume needed for a raised bed.

    Args:
        length: Bed length
        width: Bed width
        depth: Soil depth

    Returns:
        Volume in same cubic units as input
    """
    return length * width * depth


def soil_volume_pot(diameter: float, height: float) -> float:
    """Calculate soil volume for a cylindrical pot.

    Args:
        diameter: Pot diameter
        height: Pot height/depth

    Returns:
        Volume (π * r² * h)
    """
    radius = diameter / 2
    return math.pi * (radius ** 2) * height


def cubic_feet_to_gallons(cubic_feet: float) -> float:
    """Convert cubic feet to US gallons (for soil)."""
    return cubic_feet * 7.48052


def cubic_feet_to_quarts(cubic_feet: float) -> float:
    """Convert cubic feet to quarts (for potting soil bags)."""
    return cubic_feet * 29.9221


def seed_spacing_grid(bed_length: float, bed_width: float,
                      spacing: float) -> int:
    """Calculate number of plants in a grid layout.

    Args:
        bed_length: Bed length
        bed_width: Bed width
        spacing: Distance between plants

    Returns:
        Number of plants that fit
    """
    plants_length = int(bed_length / spacing) + 1
    plants_width = int(bed_width / spacing) + 1
    return plants_length * plants_width


def seed_spacing_rows(bed_length: float, bed_width: float,
                      in_row_spacing: float, row_spacing: float) -> int:
    """Calculate number of plants in row layout.

    Args:
        bed_length: Bed length
        bed_width: Bed width
        in_row_spacing: Spacing between plants in a row
        row_spacing: Spacing between rows

    Returns:
        Number of plants that fit
    """
    num_rows = int(bed_width / row_spacing) + 1
    plants_per_row = int(bed_length / in_row_spacing) + 1
    return num_rows * plants_per_row


def water_needs(area_sqft: float, inches_per_week: float = 1.0) -> float:
    """Calculate weekly water needs.

    Args:
        area_sqft: Garden area in square feet
        inches_per_week: Water needs (default 1 inch/week)

    Returns:
        Gallons of water needed per week
    """
    # 1 inch of water over 1 sq ft = 0.623 gallons
    return area_sqft * inches_per_week * 0.623


def fertilizer_amount(area_sqft: float, rate_per_1000sqft: float) -> float:
    """Calculate fertilizer amount needed.

    Args:
        area_sqft: Area to fertilize
        rate_per_1000sqft: Application rate per 1000 sq ft

    Returns:
        Amount of fertilizer needed
    """
    return (area_sqft / 1000) * rate_per_1000sqft


def npk_ratio(n: float, p: float, k: float) -> str:
    """Simplify NPK ratio to common form.

    Args:
        n: Nitrogen percentage
        p: Phosphorus percentage
        k: Potassium percentage

    Returns:
        Simplified ratio string (e.g., "1-1-1")
    """
    min_val = min(n, p, k) if min(n, p, k) > 0 else 1
    return f"{int(n/min_val)}-{int(p/min_val)}-{int(k/min_val)}"


def compost_ratio(greens_volume: float, browns_volume: float) -> float:
    """Calculate carbon to nitrogen ratio approximation.

    Ideal C:N is 25-30:1. Greens ~20:1, Browns ~60:1.

    Args:
        greens_volume: Volume of green materials
        browns_volume: Volume of brown materials

    Returns:
        Approximate C:N ratio
    """
    # Approximation based on typical materials
    greens_cn = 20
    browns_cn = 60

    total = greens_volume + browns_volume
    if total == 0:
        return 0

    weighted = (greens_volume * greens_cn + browns_volume * browns_cn) / total
    return round(weighted, 1)


def days_to_maturity_adjusted(base_days: float, temp_factor: float = 1.0) -> float:
    """Adjust days to maturity based on growing conditions.

    Args:
        base_days: Base days to maturity (from seed packet)
        temp_factor: Temperature factor (< 1 for warm, > 1 for cool)

    Returns:
        Adjusted days to maturity
    """
    return base_days * temp_factor


def sun_hours_needed(plant_type: str) -> Tuple[int, int]:
    """Get sun hours needed for plant type.

    Args:
        plant_type: Type of plant (full_sun, partial_sun, partial_shade, full_shade)

    Returns:
        (min_hours, max_hours) of direct sun
    """
    sun_requirements = {
        "full_sun": (6, 8),
        "partial_sun": (4, 6),
        "partial_shade": (2, 4),
        "full_shade": (0, 2),
    }

    plant_type_lower = plant_type.lower().replace(" ", "_")
    if plant_type_lower not in sun_requirements:
        raise ValueError(f"Unknown plant type: {plant_type}")

    return sun_requirements[plant_type_lower]


def mulch_volume(area_sqft: float, depth_inches: float) -> float:
    """Calculate mulch volume needed.

    Args:
        area_sqft: Area to cover
        depth_inches: Mulch depth

    Returns:
        Volume in cubic feet
    """
    return area_sqft * (depth_inches / 12)


def mulch_bags(cubic_feet: float, bag_size_cf: float = 2.0) -> int:
    """Calculate number of mulch bags needed.

    Args:
        cubic_feet: Total volume needed
        bag_size_cf: Bag size in cubic feet (default 2)

    Returns:
        Number of bags needed
    """
    return math.ceil(cubic_feet / bag_size_cf)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_bed_volume(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        l = float(args.get("length", args.get("l")))
        w = float(args.get("width", args.get("w")))
        d = float(args.get("depth", args.get("d")))
        return l, w, d
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse bed_volume args: {args}")


def _parse_seed_spacing(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        l = float(args.get("bed_length", args.get("length")))
        w = float(args.get("bed_width", args.get("width")))
        s = float(args.get("spacing"))
        return l, w, s
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse seed_spacing args: {args}")


def _parse_water(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        area = float(args.get("area_sqft", args.get("area")))
        inches = float(args.get("inches_per_week", args.get("inches", 1.0)))
        return area, inches
    if isinstance(args, (list, tuple)):
        return float(args[0]), float(args[1]) if len(args) > 1 else 1.0
    raise ValueError(f"Cannot parse water args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register gardening tools in the registry."""

    registry.register(ToolSpec(
        name="garden_soil_volume",
        function=lambda args: soil_volume_bed(*_parse_bed_volume(args)),
        description="Calculates soil volume needed for a raised bed.",
        parameters={
            "type": "object",
            "properties": {
                "length": {"type": "number", "description": "Bed length"},
                "width": {"type": "number", "description": "Bed width"},
                "depth": {"type": "number", "description": "Soil depth"},
            },
            "required": ["length", "width", "depth"]
        },
        returns="Volume in cubic units",
        examples=[
            {"input": {"length": 4, "width": 8, "depth": 1}, "output": 32.0},
        ],
        domain="gardening",
        tags=["soil", "volume", "raised bed"],
    ))

    registry.register(ToolSpec(
        name="garden_water_schedule",
        function=lambda args: water_needs(*_parse_water(args)),
        description="Calculates weekly water needs in gallons.",
        parameters={
            "type": "object",
            "properties": {
                "area_sqft": {"type": "number", "description": "Garden area (sq ft)"},
                "inches_per_week": {"type": "number", "description": "Water needed (default 1 inch)"},
            },
            "required": ["area_sqft"]
        },
        returns="Gallons of water needed per week",
        examples=[
            {"input": {"area_sqft": 100, "inches_per_week": 1}, "output": 62.3},
        ],
        domain="gardening",
        tags=["water", "irrigation", "schedule"],
    ))

    registry.register(ToolSpec(
        name="garden_seed_spacing",
        function=lambda args: seed_spacing_grid(*_parse_seed_spacing(args)),
        description="Calculates number of plants that fit in a grid layout.",
        parameters={
            "type": "object",
            "properties": {
                "bed_length": {"type": "number", "description": "Bed length"},
                "bed_width": {"type": "number", "description": "Bed width"},
                "spacing": {"type": "number", "description": "Distance between plants"},
            },
            "required": ["bed_length", "bed_width", "spacing"]
        },
        returns="Number of plants that fit",
        examples=[
            {"input": {"bed_length": 4, "bed_width": 8, "spacing": 1}, "output": 45},
        ],
        domain="gardening",
        tags=["spacing", "plants", "layout"],
    ))

    registry.register(ToolSpec(
        name="garden_fertilizer_ratio",
        function=lambda args: fertilizer_amount(
            float(args.get("area_sqft", args.get("area")) if isinstance(args, dict) else args[0]),
            float(args.get("rate_per_1000sqft", args.get("rate")) if isinstance(args, dict) else args[1])
        ),
        description="Calculates fertilizer amount needed for an area.",
        parameters={
            "type": "object",
            "properties": {
                "area_sqft": {"type": "number", "description": "Area to fertilize"},
                "rate_per_1000sqft": {"type": "number", "description": "Rate per 1000 sq ft"},
            },
            "required": ["area_sqft", "rate_per_1000sqft"]
        },
        returns="Amount of fertilizer needed",
        examples=[
            {"input": {"area_sqft": 500, "rate_per_1000sqft": 4}, "output": 2.0},
        ],
        domain="gardening",
        tags=["fertilizer", "rate", "application"],
    ))

    registry.register(ToolSpec(
        name="garden_mulch_volume",
        function=lambda args: mulch_bags(
            mulch_volume(
                float(args.get("area_sqft", args.get("area")) if isinstance(args, dict) else args[0]),
                float(args.get("depth_inches", args.get("depth")) if isinstance(args, dict) else args[1])
            ),
            float(args.get("bag_size", 2.0) if isinstance(args, dict) else (args[2] if len(args) > 2 else 2.0))
        ),
        description="Calculates number of mulch bags needed.",
        parameters={
            "type": "object",
            "properties": {
                "area_sqft": {"type": "number", "description": "Area to cover"},
                "depth_inches": {"type": "number", "description": "Mulch depth (inches)"},
                "bag_size": {"type": "number", "description": "Bag size in cubic feet (default 2)"},
            },
            "required": ["area_sqft", "depth_inches"]
        },
        returns="Number of bags needed",
        examples=[
            {"input": {"area_sqft": 100, "depth_inches": 3}, "output": 13},
        ],
        domain="gardening",
        tags=["mulch", "bags", "landscaping"],
    ))

    registry.register(ToolSpec(
        name="garden_compost_ratio",
        function=lambda args: compost_ratio(
            float(args.get("greens_volume", args.get("greens")) if isinstance(args, dict) else args[0]),
            float(args.get("browns_volume", args.get("browns")) if isinstance(args, dict) else args[1])
        ),
        description="Calculates approximate carbon-to-nitrogen ratio for compost.",
        parameters={
            "type": "object",
            "properties": {
                "greens_volume": {"type": "number", "description": "Volume of green materials"},
                "browns_volume": {"type": "number", "description": "Volume of brown materials"},
            },
            "required": ["greens_volume", "browns_volume"]
        },
        returns="Approximate C:N ratio",
        examples=[
            {"input": {"greens_volume": 1, "browns_volume": 2}, "output": 46.7},
        ],
        domain="gardening",
        tags=["compost", "ratio", "carbon"],
    ))
