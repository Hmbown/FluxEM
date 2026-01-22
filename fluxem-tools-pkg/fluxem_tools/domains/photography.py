"""Photography domain - exposure, depth of field, focal length.

This module provides deterministic photography computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def exposure_value(aperture: float, shutter_speed: float, iso: int = 100) -> float:
    """Calculate Exposure Value (EV).

    EV = log2(N²/t) at ISO 100
    EVᵢₛₒ = EV₁₀₀ + log2(ISO/100)

    Args:
        aperture: F-number (e.g., 2.8, 5.6)
        shutter_speed: Shutter speed in seconds (e.g., 0.001 for 1/1000s)
        iso: ISO sensitivity (default 100)

    Returns:
        Exposure Value
    """
    ev100 = math.log2((aperture ** 2) / shutter_speed)
    return ev100 - math.log2(iso / 100)


def equivalent_exposure(aperture: float, shutter: float, iso: int,
                        new_aperture: float = None, new_shutter: float = None,
                        new_iso: int = None) -> Dict[str, float]:
    """Calculate equivalent exposure settings.

    Maintains the same exposure while changing one parameter.

    Args:
        aperture: Current f-number
        shutter: Current shutter speed (seconds)
        iso: Current ISO
        new_aperture: New f-number (optional)
        new_shutter: New shutter (optional)
        new_iso: New ISO (optional)

    Returns:
        Dict with calculated equivalent settings
    """
    # Current exposure value
    current_ev = exposure_value(aperture, shutter, iso)

    result = {}

    if new_aperture is not None and new_shutter is None and new_iso is None:
        # Calculate new shutter for same EV
        # EV = log2(N²/t) - log2(ISO/100)
        # t = N² / 2^(EV + log2(ISO/100))
        ev_adjusted = current_ev + math.log2(iso / 100)
        new_shutter = (new_aperture ** 2) / (2 ** ev_adjusted)
        result = {"aperture": new_aperture, "shutter": new_shutter, "iso": iso}

    elif new_shutter is not None and new_aperture is None and new_iso is None:
        # Calculate new aperture for same EV
        ev_adjusted = current_ev + math.log2(iso / 100)
        new_aperture = math.sqrt(new_shutter * (2 ** ev_adjusted))
        result = {"aperture": round(new_aperture, 1), "shutter": new_shutter, "iso": iso}

    elif new_iso is not None and new_aperture is None and new_shutter is None:
        # Calculate new shutter for same EV at new ISO
        iso_diff = math.log2(new_iso / iso)
        new_shutter = shutter * (2 ** iso_diff)
        result = {"aperture": aperture, "shutter": new_shutter, "iso": new_iso}

    else:
        result = {"aperture": aperture, "shutter": shutter, "iso": iso}

    return result


def depth_of_field(focal_length: float, aperture: float, distance: float,
                   coc: float = 0.03) -> Dict[str, float]:
    """Calculate depth of field.

    Args:
        focal_length: Focal length in mm
        aperture: F-number
        distance: Subject distance in mm
        coc: Circle of confusion in mm (default 0.03 for full frame)

    Returns:
        Dict with near_limit, far_limit, total_dof (all in mm)
    """
    # Hyperfocal distance
    h = (focal_length ** 2) / (aperture * coc) + focal_length

    # Near and far limits
    near = (distance * (h - focal_length)) / (h + distance - 2 * focal_length)
    far = (distance * (h - focal_length)) / (h - distance)

    if far < 0:
        far = float('inf')

    return {
        "near_limit": round(near, 1),
        "far_limit": round(far, 1) if far != float('inf') else float('inf'),
        "total_dof": round(far - near, 1) if far != float('inf') else float('inf'),
    }


def hyperfocal_distance(focal_length: float, aperture: float,
                        coc: float = 0.03) -> float:
    """Calculate hyperfocal distance.

    When focused at hyperfocal distance, everything from half that
    distance to infinity is acceptably sharp.

    Args:
        focal_length: Focal length in mm
        aperture: F-number
        coc: Circle of confusion in mm (default 0.03)

    Returns:
        Hyperfocal distance in mm
    """
    return (focal_length ** 2) / (aperture * coc) + focal_length


def focal_length_equivalent(focal_length: float, crop_factor: float) -> float:
    """Calculate 35mm equivalent focal length.

    Args:
        focal_length: Actual focal length in mm
        crop_factor: Sensor crop factor (1.0 for full frame, 1.5 for APS-C, etc.)

    Returns:
        Equivalent focal length in mm
    """
    return focal_length * crop_factor


def flash_guide_number(distance: float, aperture: float) -> float:
    """Calculate flash guide number.

    GN = Distance × Aperture

    Args:
        distance: Flash-to-subject distance (meters)
        aperture: F-number used

    Returns:
        Guide number (in meters at ISO 100)
    """
    return distance * aperture


def flash_distance(guide_number: float, aperture: float) -> float:
    """Calculate maximum flash distance.

    Distance = GN / Aperture

    Args:
        guide_number: Flash guide number
        aperture: F-number

    Returns:
        Maximum distance in meters
    """
    return guide_number / aperture


def aspect_ratio_crop(original_width: int, original_height: int,
                      target_ratio: str) -> Tuple[int, int]:
    """Calculate crop dimensions for target aspect ratio.

    Args:
        original_width: Original image width
        original_height: Original image height
        target_ratio: Target ratio as string (e.g., "16:9", "4:3", "1:1")

    Returns:
        (cropped_width, cropped_height)
    """
    parts = target_ratio.split(":")
    target_w = float(parts[0])
    target_h = float(parts[1])
    target = target_w / target_h

    original_ratio = original_width / original_height

    if original_ratio > target:
        # Original is wider - crop width
        new_width = int(original_height * target)
        return (new_width, original_height)
    else:
        # Original is taller - crop height
        new_height = int(original_width / target)
        return (original_width, new_height)


def stops_difference(value1: float, value2: float, is_aperture: bool = True) -> float:
    """Calculate difference in stops between two values.

    Args:
        value1: First value (aperture or shutter)
        value2: Second value
        is_aperture: True if comparing f-numbers, False for shutter speeds

    Returns:
        Difference in stops
    """
    if is_aperture:
        # Aperture: each stop is sqrt(2) ratio
        return 2 * math.log2(value2 / value1)
    else:
        # Shutter: each stop is 2x ratio
        return math.log2(value2 / value1)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_exposure(args) -> Tuple[float, float, int]:
    if isinstance(args, dict):
        aperture = float(args.get("aperture", args.get("f")))
        shutter = float(args.get("shutter_speed", args.get("shutter")))
        iso = int(args.get("iso", 100))
        return aperture, shutter, iso
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1]), int(args[2]) if len(args) > 2 else 100
    raise ValueError(f"Cannot parse exposure args: {args}")


def _parse_dof(args) -> Tuple[float, float, float, float]:
    if isinstance(args, dict):
        fl = float(args.get("focal_length", args.get("fl")))
        ap = float(args.get("aperture", args.get("f")))
        dist = float(args.get("distance", args.get("d")))
        coc = float(args.get("coc", 0.03))
        return fl, ap, dist, coc
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2]), float(args[3]) if len(args) > 3 else 0.03
    raise ValueError(f"Cannot parse dof args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register photography tools in the registry."""

    registry.register(ToolSpec(
        name="photo_exposure_value",
        function=lambda args: exposure_value(*_parse_exposure(args)),
        description="Calculates Exposure Value (EV) from aperture, shutter, and ISO.",
        parameters={
            "type": "object",
            "properties": {
                "aperture": {"type": "number", "description": "F-number (e.g., 2.8)"},
                "shutter_speed": {"type": "number", "description": "Shutter in seconds (e.g., 0.001)"},
                "iso": {"type": "integer", "description": "ISO sensitivity (default 100)"},
            },
            "required": ["aperture", "shutter_speed"]
        },
        returns="Exposure Value",
        examples=[
            {"input": {"aperture": 8, "shutter_speed": 0.004, "iso": 100}, "output": 14.0},
        ],
        domain="photography",
        tags=["exposure", "ev", "light"],
    ))

    registry.register(ToolSpec(
        name="photo_depth_of_field",
        function=lambda args: depth_of_field(*_parse_dof(args)),
        description="Calculates depth of field (near limit, far limit, total).",
        parameters={
            "type": "object",
            "properties": {
                "focal_length": {"type": "number", "description": "Focal length in mm"},
                "aperture": {"type": "number", "description": "F-number"},
                "distance": {"type": "number", "description": "Subject distance in mm"},
                "coc": {"type": "number", "description": "Circle of confusion (default 0.03)"},
            },
            "required": ["focal_length", "aperture", "distance"]
        },
        returns="Dict with near_limit, far_limit, total_dof in mm",
        examples=[
            {"input": {"focal_length": 50, "aperture": 2.8, "distance": 3000}, "output": {"near_limit": 2720, "far_limit": 3340, "total_dof": 620}},
        ],
        domain="photography",
        tags=["dof", "focus", "aperture"],
    ))

    registry.register(ToolSpec(
        name="photo_focal_length_equiv",
        function=lambda args: focal_length_equivalent(
            float(args.get("focal_length", args.get("fl")) if isinstance(args, dict) else args[0]),
            float(args.get("crop_factor", args.get("crop")) if isinstance(args, dict) else args[1])
        ),
        description="Calculates 35mm equivalent focal length for crop sensors.",
        parameters={
            "type": "object",
            "properties": {
                "focal_length": {"type": "number", "description": "Actual focal length in mm"},
                "crop_factor": {"type": "number", "description": "Sensor crop factor (1.5 for APS-C)"},
            },
            "required": ["focal_length", "crop_factor"]
        },
        returns="Equivalent focal length in mm",
        examples=[
            {"input": {"focal_length": 35, "crop_factor": 1.5}, "output": 52.5},
        ],
        domain="photography",
        tags=["focal length", "crop", "equivalent"],
    ))

    registry.register(ToolSpec(
        name="photo_flash_guide_number",
        function=lambda args: flash_guide_number(
            float(args.get("distance", args.get("d")) if isinstance(args, dict) else args[0]),
            float(args.get("aperture", args.get("f")) if isinstance(args, dict) else args[1])
        ),
        description="Calculates flash guide number (GN = Distance × Aperture).",
        parameters={
            "type": "object",
            "properties": {
                "distance": {"type": "number", "description": "Flash distance (meters)"},
                "aperture": {"type": "number", "description": "F-number used"},
            },
            "required": ["distance", "aperture"]
        },
        returns="Guide number (meters at ISO 100)",
        examples=[
            {"input": {"distance": 5, "aperture": 5.6}, "output": 28.0},
        ],
        domain="photography",
        tags=["flash", "guide number", "exposure"],
    ))

    registry.register(ToolSpec(
        name="photo_hyperfocal",
        function=lambda args: hyperfocal_distance(
            float(args.get("focal_length", args.get("fl")) if isinstance(args, dict) else args[0]),
            float(args.get("aperture", args.get("f")) if isinstance(args, dict) else args[1]),
            float(args.get("coc", 0.03) if isinstance(args, dict) else (args[2] if len(args) > 2 else 0.03))
        ),
        description="Calculates hyperfocal distance for maximum depth of field.",
        parameters={
            "type": "object",
            "properties": {
                "focal_length": {"type": "number", "description": "Focal length in mm"},
                "aperture": {"type": "number", "description": "F-number"},
                "coc": {"type": "number", "description": "Circle of confusion (default 0.03)"},
            },
            "required": ["focal_length", "aperture"]
        },
        returns="Hyperfocal distance in mm",
        examples=[
            {"input": {"focal_length": 24, "aperture": 11}, "output": 1769.5},
        ],
        domain="photography",
        tags=["hyperfocal", "focus", "landscape"],
    ))

    registry.register(ToolSpec(
        name="photo_aspect_ratio_crop",
        function=lambda args: aspect_ratio_crop(
            int(args.get("original_width", args.get("width")) if isinstance(args, dict) else args[0]),
            int(args.get("original_height", args.get("height")) if isinstance(args, dict) else args[1]),
            str(args.get("target_ratio", args.get("ratio")) if isinstance(args, dict) else args[2])
        ),
        description="Calculates crop dimensions for target aspect ratio.",
        parameters={
            "type": "object",
            "properties": {
                "original_width": {"type": "integer", "description": "Original image width"},
                "original_height": {"type": "integer", "description": "Original image height"},
                "target_ratio": {"type": "string", "description": "Target ratio (e.g., '16:9', '1:1')"},
            },
            "required": ["original_width", "original_height", "target_ratio"]
        },
        returns="(cropped_width, cropped_height)",
        examples=[
            {"input": {"original_width": 6000, "original_height": 4000, "target_ratio": "16:9"}, "output": [6000, 3375]},
        ],
        domain="photography",
        tags=["aspect ratio", "crop", "dimensions"],
    ))
