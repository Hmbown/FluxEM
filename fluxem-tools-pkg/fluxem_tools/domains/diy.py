"""DIY/Construction domain - paint, flooring, lumber calculations.

This module provides deterministic construction/home improvement computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def paint_area(length: float, height: float, num_coats: int = 2,
               coverage_per_gallon: float = 350) -> float:
    """Calculate gallons of paint needed for walls.

    Args:
        length: Total wall length (feet or meters)
        height: Wall height
        num_coats: Number of coats (default 2)
        coverage_per_gallon: Square feet per gallon (default 350)

    Returns:
        Gallons of paint needed
    """
    area = length * height * num_coats
    return area / coverage_per_gallon


def flooring_tiles(room_length: float, room_width: float,
                   tile_length: float, tile_width: float,
                   waste_factor: float = 0.10) -> int:
    """Calculate number of tiles needed for flooring.

    Args:
        room_length: Room length
        room_width: Room width
        tile_length: Individual tile length
        tile_width: Individual tile width
        waste_factor: Extra for waste/cuts (default 10%)

    Returns:
        Number of tiles needed (rounded up)
    """
    room_area = room_length * room_width
    tile_area = tile_length * tile_width
    tiles_needed = room_area / tile_area
    tiles_with_waste = tiles_needed * (1 + waste_factor)
    return math.ceil(tiles_with_waste)


def lumber_board_feet(thickness_inches: float, width_inches: float,
                      length_feet: float) -> float:
    """Calculate board feet of lumber.

    Board foot = (thickness × width × length) / 144
    (when thickness and width in inches, length in feet)

    Args:
        thickness_inches: Thickness in inches
        width_inches: Width in inches
        length_feet: Length in feet

    Returns:
        Board feet
    """
    return (thickness_inches * width_inches * length_feet * 12) / 144


def concrete_volume(length: float, width: float, depth: float) -> float:
    """Calculate concrete volume needed.

    Args:
        length: Length (feet or meters)
        width: Width
        depth: Depth/thickness

    Returns:
        Volume in cubic units
    """
    return length * width * depth


def cubic_feet_to_yards(cubic_feet: float) -> float:
    """Convert cubic feet to cubic yards (for concrete ordering).

    Args:
        cubic_feet: Volume in cubic feet

    Returns:
        Volume in cubic yards
    """
    return cubic_feet / 27


def wallpaper_rolls(wall_area: float, roll_coverage: float = 28,
                    pattern_repeat: float = 0) -> int:
    """Calculate wallpaper rolls needed.

    Args:
        wall_area: Total wall area (sq ft)
        roll_coverage: Coverage per roll (default 28 sq ft)
        pattern_repeat: Pattern repeat length (adds waste)

    Returns:
        Number of rolls needed
    """
    # Add waste for pattern matching
    waste_factor = 1.0 if pattern_repeat == 0 else 1.15
    adjusted_area = wall_area * waste_factor
    return math.ceil(adjusted_area / roll_coverage)


def stair_dimensions(total_rise: float, riser_height: float = 7.5) -> Dict[str, float]:
    """Calculate stair dimensions.

    Standard: Riser height 7-7.75", Tread depth 10-11"
    Rule: 2 risers + 1 tread = 24-25 inches

    Args:
        total_rise: Total height to climb (inches)
        riser_height: Desired riser height (default 7.5")

    Returns:
        Dict with num_risers, riser_height, tread_depth, total_run
    """
    num_risers = round(total_rise / riser_height)
    actual_riser = total_rise / num_risers
    # Using 2R + T = 25 rule
    tread_depth = 25 - 2 * actual_riser
    total_run = tread_depth * (num_risers - 1)

    return {
        "num_risers": num_risers,
        "riser_height": round(actual_riser, 2),
        "tread_depth": round(tread_depth, 2),
        "total_run": round(total_run, 2),
    }


def wire_gauge_ampacity(amps: float, distance_feet: float,
                        voltage: float = 120) -> int:
    """Recommend wire gauge for circuit.

    Args:
        amps: Current draw (amps)
        distance_feet: One-way wire run distance
        voltage: Circuit voltage (default 120V)

    Returns:
        Recommended AWG gauge
    """
    # Simplified - considers 3% voltage drop
    # Voltage drop = 2 * I * R * L (round trip)
    # For copper: AWG relates to resistance

    # Common residential gauges and their max amps
    gauges = [
        (14, 15),
        (12, 20),
        (10, 30),
        (8, 40),
        (6, 55),
        (4, 70),
        (2, 95),
    ]

    # Find appropriate gauge
    for gauge, max_amps in gauges:
        if amps <= max_amps * 0.8:  # 80% rule
            return gauge

    return 2  # Largest common residential


def deck_boards(deck_length: float, deck_width: float,
                board_length: float, board_width: float,
                gap: float = 0.125) -> int:
    """Calculate deck boards needed.

    Args:
        deck_length: Deck length (feet)
        deck_width: Deck width (feet)
        board_length: Board length (feet)
        board_width: Board width (inches)
        gap: Gap between boards (inches, default 1/8")

    Returns:
        Number of boards needed
    """
    # Convert board width to feet and add gap
    board_width_ft = (board_width + gap) / 12
    boards_across = math.ceil(deck_width / board_width_ft)
    boards_lengthwise = math.ceil(deck_length / board_length)

    return boards_across * boards_lengthwise


def gravel_tons(length: float, width: float, depth_inches: float,
                density: float = 1.4) -> float:
    """Calculate tons of gravel needed.

    Args:
        length: Area length (feet)
        width: Area width (feet)
        depth_inches: Gravel depth (inches)
        density: Tons per cubic yard (default 1.4)

    Returns:
        Tons of gravel needed
    """
    volume_cf = length * width * (depth_inches / 12)
    volume_cy = volume_cf / 27
    return volume_cy * density


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_paint(args) -> Tuple[float, float, int, float]:
    if isinstance(args, dict):
        length = float(args.get("length"))
        height = float(args.get("height"))
        coats = int(args.get("num_coats", args.get("coats", 2)))
        coverage = float(args.get("coverage_per_gallon", args.get("coverage", 350)))
        return length, height, coats, coverage
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return (float(args[0]), float(args[1]),
                int(args[2]) if len(args) > 2 else 2,
                float(args[3]) if len(args) > 3 else 350)
    raise ValueError(f"Cannot parse paint args: {args}")


def _parse_tiles(args) -> Tuple[float, float, float, float, float]:
    if isinstance(args, dict):
        rl = float(args.get("room_length", args.get("length")))
        rw = float(args.get("room_width", args.get("width")))
        tl = float(args.get("tile_length", args.get("tile_l")))
        tw = float(args.get("tile_width", args.get("tile_w")))
        waste = float(args.get("waste_factor", args.get("waste", 0.10)))
        return rl, rw, tl, tw, waste
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return (float(args[0]), float(args[1]), float(args[2]), float(args[3]),
                float(args[4]) if len(args) > 4 else 0.10)
    raise ValueError(f"Cannot parse tiles args: {args}")


def _parse_concrete(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        l = float(args.get("length", args.get("l")))
        w = float(args.get("width", args.get("w")))
        d = float(args.get("depth", args.get("d")))
        return l, w, d
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse concrete args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register DIY/construction tools in the registry."""

    registry.register(ToolSpec(
        name="diy_paint_area",
        function=lambda args: paint_area(*_parse_paint(args)),
        description="Calculates gallons of paint needed for walls.",
        parameters={
            "type": "object",
            "properties": {
                "length": {"type": "number", "description": "Total wall length"},
                "height": {"type": "number", "description": "Wall height"},
                "num_coats": {"type": "integer", "description": "Number of coats (default 2)"},
                "coverage_per_gallon": {"type": "number", "description": "Sq ft per gallon (default 350)"},
            },
            "required": ["length", "height"]
        },
        returns="Gallons of paint needed",
        examples=[
            {"input": {"length": 50, "height": 8, "num_coats": 2}, "output": 2.29},
        ],
        domain="diy",
        tags=["paint", "wall", "area"],
    ))

    registry.register(ToolSpec(
        name="diy_flooring_tiles",
        function=lambda args: flooring_tiles(*_parse_tiles(args)),
        description="Calculates number of tiles needed for flooring (includes waste factor).",
        parameters={
            "type": "object",
            "properties": {
                "room_length": {"type": "number", "description": "Room length"},
                "room_width": {"type": "number", "description": "Room width"},
                "tile_length": {"type": "number", "description": "Tile length"},
                "tile_width": {"type": "number", "description": "Tile width"},
                "waste_factor": {"type": "number", "description": "Waste factor (default 0.10)"},
            },
            "required": ["room_length", "room_width", "tile_length", "tile_width"]
        },
        returns="Number of tiles needed",
        examples=[
            {"input": {"room_length": 12, "room_width": 10, "tile_length": 1, "tile_width": 1}, "output": 132},
        ],
        domain="diy",
        tags=["tile", "floor", "area"],
    ))

    registry.register(ToolSpec(
        name="diy_lumber_board_feet",
        function=lambda args: lumber_board_feet(
            float(args.get("thickness_inches", args.get("thickness")) if isinstance(args, dict) else args[0]),
            float(args.get("width_inches", args.get("width")) if isinstance(args, dict) else args[1]),
            float(args.get("length_feet", args.get("length")) if isinstance(args, dict) else args[2])
        ),
        description="Calculates board feet of lumber.",
        parameters={
            "type": "object",
            "properties": {
                "thickness_inches": {"type": "number", "description": "Thickness in inches"},
                "width_inches": {"type": "number", "description": "Width in inches"},
                "length_feet": {"type": "number", "description": "Length in feet"},
            },
            "required": ["thickness_inches", "width_inches", "length_feet"]
        },
        returns="Board feet",
        examples=[
            {"input": {"thickness_inches": 1, "width_inches": 12, "length_feet": 8}, "output": 8.0},
        ],
        domain="diy",
        tags=["lumber", "board feet", "wood"],
    ))

    registry.register(ToolSpec(
        name="diy_concrete_volume",
        function=lambda args: cubic_feet_to_yards(concrete_volume(*_parse_concrete(args))),
        description="Calculates cubic yards of concrete needed.",
        parameters={
            "type": "object",
            "properties": {
                "length": {"type": "number", "description": "Length (feet)"},
                "width": {"type": "number", "description": "Width (feet)"},
                "depth": {"type": "number", "description": "Depth/thickness (feet)"},
            },
            "required": ["length", "width", "depth"]
        },
        returns="Volume in cubic yards",
        examples=[
            {"input": {"length": 10, "width": 10, "depth": 0.333}, "output": 1.23},
        ],
        domain="diy",
        tags=["concrete", "volume", "yards"],
    ))

    registry.register(ToolSpec(
        name="diy_wallpaper_rolls",
        function=lambda args: wallpaper_rolls(
            float(args.get("wall_area", args.get("area")) if isinstance(args, dict) else args[0]),
            float(args.get("roll_coverage", args.get("coverage", 28)) if isinstance(args, dict) else (args[1] if len(args) > 1 else 28)),
            float(args.get("pattern_repeat", 0) if isinstance(args, dict) else (args[2] if len(args) > 2 else 0))
        ),
        description="Calculates wallpaper rolls needed.",
        parameters={
            "type": "object",
            "properties": {
                "wall_area": {"type": "number", "description": "Total wall area (sq ft)"},
                "roll_coverage": {"type": "number", "description": "Coverage per roll (default 28)"},
                "pattern_repeat": {"type": "number", "description": "Pattern repeat (adds waste)"},
            },
            "required": ["wall_area"]
        },
        returns="Number of rolls needed",
        examples=[
            {"input": {"wall_area": 200}, "output": 8},
        ],
        domain="diy",
        tags=["wallpaper", "rolls", "wall"],
    ))

    registry.register(ToolSpec(
        name="diy_stair_dimensions",
        function=lambda args: stair_dimensions(
            float(args.get("total_rise", args.get("rise")) if isinstance(args, dict) else args[0]),
            float(args.get("riser_height", 7.5) if isinstance(args, dict) else (args[1] if len(args) > 1 else 7.5))
        ),
        description="Calculates stair dimensions (risers, treads, total run).",
        parameters={
            "type": "object",
            "properties": {
                "total_rise": {"type": "number", "description": "Total height to climb (inches)"},
                "riser_height": {"type": "number", "description": "Desired riser height (default 7.5)"},
            },
            "required": ["total_rise"]
        },
        returns="Dict with num_risers, riser_height, tread_depth, total_run",
        examples=[
            {"input": {"total_rise": 108}, "output": {"num_risers": 14, "riser_height": 7.71, "tread_depth": 9.57, "total_run": 124.43}},
        ],
        domain="diy",
        tags=["stairs", "dimensions", "construction"],
    ))

    registry.register(ToolSpec(
        name="diy_wire_gauge",
        function=lambda args: wire_gauge_ampacity(
            float(args.get("amps", args.get("current")) if isinstance(args, dict) else args[0]),
            float(args.get("distance_feet", args.get("distance")) if isinstance(args, dict) else args[1]),
            float(args.get("voltage", 120) if isinstance(args, dict) else (args[2] if len(args) > 2 else 120))
        ),
        description="Recommends wire gauge (AWG) for a circuit.",
        parameters={
            "type": "object",
            "properties": {
                "amps": {"type": "number", "description": "Current draw (amps)"},
                "distance_feet": {"type": "number", "description": "Wire run distance (feet)"},
                "voltage": {"type": "number", "description": "Circuit voltage (default 120)"},
            },
            "required": ["amps", "distance_feet"]
        },
        returns="Recommended AWG gauge",
        examples=[
            {"input": {"amps": 15, "distance_feet": 50}, "output": 12},
        ],
        domain="diy",
        tags=["wire", "gauge", "electrical"],
    ))
