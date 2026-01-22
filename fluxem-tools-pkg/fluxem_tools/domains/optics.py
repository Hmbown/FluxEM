"""Optics domain - lenses, light, refraction.

This module provides deterministic optics computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

C = 299792458  # Speed of light in vacuum (m/s)
N_AIR = 1.0003  # Refractive index of air
N_WATER = 1.333  # Refractive index of water
N_GLASS = 1.52  # Typical crown glass


# =============================================================================
# Core Functions
# =============================================================================

def thin_lens_equation(object_dist: float, image_dist: float = None,
                       focal_length: float = None) -> float:
    """Calculate using thin lens equation: 1/f = 1/do + 1/di

    Provide any two values to calculate the third.

    Args:
        object_dist: Distance from object to lens (positive if on same side as light source)
        image_dist: Distance from lens to image (positive if on opposite side from object)
        focal_length: Focal length of lens

    Returns:
        The missing value (focal_length, image_dist, or object_dist)
    """
    if focal_length is None:
        if object_dist is None or image_dist is None:
            raise ValueError("Need at least two values")
        return 1 / (1/object_dist + 1/image_dist)
    elif image_dist is None:
        if object_dist is None or focal_length is None:
            raise ValueError("Need at least two values")
        return 1 / (1/focal_length - 1/object_dist)
    else:  # object_dist is None
        if image_dist is None or focal_length is None:
            raise ValueError("Need at least two values")
        return 1 / (1/focal_length - 1/image_dist)


def magnification(image_dist: float, object_dist: float) -> float:
    """Calculate linear magnification.

    M = -di/do (negative indicates inverted image)

    Args:
        image_dist: Image distance from lens
        object_dist: Object distance from lens

    Returns:
        Magnification (negative = inverted)
    """
    if object_dist == 0:
        raise ValueError("Object distance cannot be zero")
    return -image_dist / object_dist


def snells_law(n1: float, angle1: float, n2: float) -> float:
    """Calculate refracted angle using Snell's law.

    n1 * sin(θ1) = n2 * sin(θ2)

    Args:
        n1: Refractive index of first medium
        angle1: Incident angle (radians)
        n2: Refractive index of second medium

    Returns:
        Refracted angle in radians (or NaN if total internal reflection)
    """
    sin_theta2 = (n1 / n2) * math.sin(angle1)
    if abs(sin_theta2) > 1:
        return float('nan')  # Total internal reflection
    return math.asin(sin_theta2)


def snells_law_degrees(n1: float, angle1_deg: float, n2: float) -> float:
    """Snell's law with angles in degrees."""
    result_rad = snells_law(n1, math.radians(angle1_deg), n2)
    return math.degrees(result_rad) if not math.isnan(result_rad) else float('nan')


def critical_angle(n1: float, n2: float) -> float:
    """Calculate critical angle for total internal reflection.

    θc = arcsin(n2/n1), where n1 > n2

    Args:
        n1: Refractive index of denser medium (light source side)
        n2: Refractive index of less dense medium

    Returns:
        Critical angle in radians
    """
    if n1 <= n2:
        raise ValueError("n1 must be greater than n2 for total internal reflection")
    return math.asin(n2 / n1)


def critical_angle_degrees(n1: float, n2: float) -> float:
    """Critical angle in degrees."""
    return math.degrees(critical_angle(n1, n2))


def diffraction_grating_angle(order: int, wavelength: float, grating_spacing: float) -> float:
    """Calculate diffraction angle from a grating.

    d * sin(θ) = m * λ

    Args:
        order: Diffraction order (integer, can be negative)
        wavelength: Wavelength of light (m)
        grating_spacing: Distance between grating lines (m)

    Returns:
        Diffraction angle in radians
    """
    sin_theta = (order * wavelength) / grating_spacing
    if abs(sin_theta) > 1:
        return float('nan')  # Order not visible
    return math.asin(sin_theta)


def wavelength_to_frequency(wavelength: float) -> float:
    """Convert wavelength to frequency.

    f = c / λ

    Args:
        wavelength: Wavelength (m)

    Returns:
        Frequency in Hz
    """
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive")
    return C / wavelength


def frequency_to_wavelength(frequency: float) -> float:
    """Convert frequency to wavelength.

    λ = c / f

    Args:
        frequency: Frequency (Hz)

    Returns:
        Wavelength in meters
    """
    if frequency <= 0:
        raise ValueError("Frequency must be positive")
    return C / frequency


def lens_power(focal_length: float) -> float:
    """Calculate lens power in diopters.

    P = 1/f (where f is in meters)

    Args:
        focal_length: Focal length (m)

    Returns:
        Power in diopters (D)
    """
    if focal_length == 0:
        raise ValueError("Focal length cannot be zero")
    return 1 / focal_length


def brewster_angle(n1: float, n2: float) -> float:
    """Calculate Brewster's angle for polarization.

    θB = arctan(n2/n1)

    Args:
        n1: Refractive index of first medium
        n2: Refractive index of second medium

    Returns:
        Brewster's angle in radians
    """
    return math.atan(n2 / n1)


def numerical_aperture(n: float, half_angle: float) -> float:
    """Calculate numerical aperture of an optical system.

    NA = n * sin(θ)

    Args:
        n: Refractive index of medium
        half_angle: Half-angle of maximum cone of light (radians)

    Returns:
        Numerical aperture
    """
    return n * math.sin(half_angle)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_thin_lens(args) -> Dict[str, float]:
    if isinstance(args, dict):
        return {
            "object_dist": float(args.get("object_dist", args.get("do"))) if "object_dist" in args or "do" in args else None,
            "image_dist": float(args.get("image_dist", args.get("di"))) if "image_dist" in args or "di" in args else None,
            "focal_length": float(args.get("focal_length", args.get("f"))) if "focal_length" in args or "f" in args else None,
        }
    raise ValueError(f"Cannot parse thin lens args: {args}")


def _parse_snell(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        n1 = float(args.get("n1", args.get("n_incident", N_AIR)))
        angle = float(args.get("angle", args.get("theta1", args.get("angle1"))))
        n2 = float(args.get("n2", args.get("n_refracted")))
        return n1, angle, n2
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse snell args: {args}")


def _parse_magnification(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        di = float(args.get("image_dist", args.get("di")))
        do = float(args.get("object_dist", args.get("do")))
        return di, do
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse magnification args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register optics tools in the registry."""

    registry.register(ToolSpec(
        name="optics_focal_length",
        function=lambda args: thin_lens_equation(**_parse_thin_lens(args)),
        description="Calculates using thin lens equation (1/f = 1/do + 1/di). Provide two values.",
        parameters={
            "type": "object",
            "properties": {
                "object_dist": {"type": "number", "description": "Object distance from lens (m)"},
                "image_dist": {"type": "number", "description": "Image distance from lens (m)"},
                "focal_length": {"type": "number", "description": "Focal length (m)"},
            },
        },
        returns="The missing value from the thin lens equation",
        examples=[
            {"input": {"object_dist": 0.3, "image_dist": 0.6}, "output": 0.2},
        ],
        domain="optics",
        tags=["lens", "focal", "thin lens"],
    ))

    registry.register(ToolSpec(
        name="optics_magnification",
        function=lambda args: magnification(*_parse_magnification(args)),
        description="Calculates linear magnification (M = -di/do).",
        parameters={
            "type": "object",
            "properties": {
                "image_dist": {"type": "number", "description": "Image distance (m)"},
                "object_dist": {"type": "number", "description": "Object distance (m)"},
            },
            "required": ["image_dist", "object_dist"]
        },
        returns="Magnification (negative = inverted)",
        examples=[
            {"input": {"image_dist": 0.6, "object_dist": 0.3}, "output": -2.0},
        ],
        domain="optics",
        tags=["magnification", "lens", "image"],
    ))

    registry.register(ToolSpec(
        name="optics_snells_law",
        function=lambda args: snells_law_degrees(*_parse_snell(args)),
        description="Calculates refracted angle using Snell's law (angles in degrees).",
        parameters={
            "type": "object",
            "properties": {
                "n1": {"type": "number", "description": "Refractive index of first medium"},
                "angle": {"type": "number", "description": "Incident angle (degrees)"},
                "n2": {"type": "number", "description": "Refractive index of second medium"},
            },
            "required": ["n1", "angle", "n2"]
        },
        returns="Refracted angle in degrees (NaN if total internal reflection)",
        examples=[
            {"input": {"n1": 1.0, "angle": 30, "n2": 1.5}, "output": 19.47},
        ],
        domain="optics",
        tags=["snell", "refraction", "angle"],
    ))

    def _parse_critical_angle(args):
        if isinstance(args, dict):
            n1 = args.get("n1")
            n2 = args.get("n2")
            if n1 is None or n2 is None:
                raise ValueError("Required: n1 and n2")
            return critical_angle_degrees(float(n1), float(n2))
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            return critical_angle_degrees(float(args[0]), float(args[1]))
        raise ValueError(f"Cannot parse critical_angle args: {args}")

    registry.register(ToolSpec(
        name="optics_critical_angle",
        function=_parse_critical_angle,
        description="Calculates critical angle for total internal reflection (degrees).",
        parameters={
            "type": "object",
            "properties": {
                "n1": {"type": "number", "description": "Refractive index of denser medium"},
                "n2": {"type": "number", "description": "Refractive index of less dense medium"},
            },
            "required": ["n1", "n2"]
        },
        returns="Critical angle in degrees",
        examples=[
            {"input": {"n1": 1.5, "n2": 1.0}, "output": 41.81},
        ],
        domain="optics",
        tags=["critical angle", "reflection", "total internal"],
    ))

    def _parse_diffraction_grating(args):
        if isinstance(args, dict):
            m = args.get("order", args.get("m"))
            wl = args.get("wavelength", args.get("lambda"))
            sp = args.get("spacing", args.get("d"))
            if m is None or wl is None or sp is None:
                raise ValueError("Required: order, wavelength, and spacing")
            return math.degrees(diffraction_grating_angle(int(m), float(wl), float(sp)))
        if isinstance(args, (list, tuple)) and len(args) >= 3:
            return math.degrees(diffraction_grating_angle(int(args[0]), float(args[1]), float(args[2])))
        raise ValueError(f"Cannot parse diffraction_grating args: {args}")

    registry.register(ToolSpec(
        name="optics_diffraction_grating",
        function=_parse_diffraction_grating,
        description="Calculates diffraction angle from a grating (d*sin(θ) = m*λ).",
        parameters={
            "type": "object",
            "properties": {
                "order": {"type": "integer", "description": "Diffraction order"},
                "wavelength": {"type": "number", "description": "Wavelength (m)"},
                "spacing": {"type": "number", "description": "Grating line spacing (m)"},
            },
            "required": ["order", "wavelength", "spacing"]
        },
        returns="Diffraction angle in degrees",
        examples=[
            {"input": {"order": 1, "wavelength": 500e-9, "spacing": 1e-6}, "output": 30.0},
        ],
        domain="optics",
        tags=["diffraction", "grating", "wavelength"],
    ))

    registry.register(ToolSpec(
        name="optics_lens_power",
        function=lambda args: lens_power(
            float(args.get("focal_length", args.get("f", args)) if isinstance(args, dict) else args)
        ),
        description="Calculates lens power in diopters (P = 1/f).",
        parameters={
            "type": "object",
            "properties": {
                "focal_length": {"type": "number", "description": "Focal length (m)"},
            },
            "required": ["focal_length"]
        },
        returns="Power in diopters",
        examples=[
            {"input": {"focal_length": 0.5}, "output": 2.0},
        ],
        domain="optics",
        tags=["lens", "power", "diopter"],
    ))
