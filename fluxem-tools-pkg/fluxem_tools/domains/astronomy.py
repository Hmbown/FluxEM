"""Astronomy domain - orbital mechanics, celestial calculations.

This module provides deterministic astronomy computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

G = 6.67430e-11  # Gravitational constant (m³/(kg·s²))
AU = 1.495978707e11  # Astronomical Unit (m)
PARSEC = 3.085677581e16  # Parsec (m)
SOLAR_MASS = 1.989e30  # kg
EARTH_MASS = 5.972e24  # kg
C = 299792458  # Speed of light (m/s)

# Julian date reference
J2000 = 2451545.0  # January 1, 2000, 12:00 TT


# =============================================================================
# Core Functions
# =============================================================================

def orbital_period(semi_major_axis: float, central_mass: float) -> float:
    """Calculate orbital period using Kepler's third law.

    T = 2π * sqrt(a³ / (G * M))

    Args:
        semi_major_axis: Semi-major axis of orbit (m)
        central_mass: Mass of central body (kg)

    Returns:
        Orbital period in seconds
    """
    if semi_major_axis <= 0 or central_mass <= 0:
        raise ValueError("Semi-major axis and mass must be positive")
    return 2 * math.pi * math.sqrt((semi_major_axis ** 3) / (G * central_mass))


def escape_velocity(mass: float, radius: float) -> float:
    """Calculate escape velocity from a body.

    v_esc = sqrt(2 * G * M / r)

    Args:
        mass: Mass of body (kg)
        radius: Distance from center (m)

    Returns:
        Escape velocity in m/s
    """
    if mass <= 0 or radius <= 0:
        raise ValueError("Mass and radius must be positive")
    return math.sqrt(2 * G * mass / radius)


def orbital_velocity(mass: float, radius: float) -> float:
    """Calculate circular orbital velocity.

    v = sqrt(G * M / r)

    Args:
        mass: Mass of central body (kg)
        radius: Orbital radius (m)

    Returns:
        Orbital velocity in m/s
    """
    if mass <= 0 or radius <= 0:
        raise ValueError("Mass and radius must be positive")
    return math.sqrt(G * mass / radius)


def parallax_distance(parallax_arcsec: float) -> float:
    """Calculate distance from parallax angle.

    d (pc) = 1 / p (arcsec)

    Args:
        parallax_arcsec: Parallax angle in arcseconds

    Returns:
        Distance in parsecs
    """
    if parallax_arcsec <= 0:
        raise ValueError("Parallax must be positive")
    return 1.0 / parallax_arcsec


def angular_diameter(physical_diameter: float, distance: float) -> float:
    """Calculate angular diameter of an object.

    θ = 2 * arctan(d / (2 * D)) ≈ d / D (for small angles)

    Args:
        physical_diameter: Physical diameter (same units as distance)
        distance: Distance to object

    Returns:
        Angular diameter in radians
    """
    if distance <= 0:
        raise ValueError("Distance must be positive")
    return 2 * math.atan(physical_diameter / (2 * distance))


def angular_diameter_degrees(physical_diameter: float, distance: float) -> float:
    """Calculate angular diameter in degrees."""
    return math.degrees(angular_diameter(physical_diameter, distance))


def schwarzschild_radius(mass: float) -> float:
    """Calculate Schwarzschild radius (event horizon) of a black hole.

    r_s = 2 * G * M / c²

    Args:
        mass: Mass (kg)

    Returns:
        Schwarzschild radius in meters
    """
    if mass <= 0:
        raise ValueError("Mass must be positive")
    return 2 * G * mass / (C ** 2)


def moon_phase(julian_day: float) -> float:
    """Calculate moon phase from Julian day.

    Returns phase as fraction (0 = new moon, 0.5 = full moon).

    Args:
        julian_day: Julian day number

    Returns:
        Phase as fraction (0 to 1)
    """
    # Synodic month = 29.53059 days
    synodic_month = 29.53059
    # Known new moon: January 6, 2000 (JD 2451550.1)
    known_new_moon = 2451550.1

    days_since = julian_day - known_new_moon
    phase = (days_since % synodic_month) / synodic_month
    return phase


def julian_day(year: int, month: int, day: float) -> float:
    """Calculate Julian day number from calendar date.

    Uses the algorithm for Gregorian calendar dates.

    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)
        day: Day (can include fractional day)

    Returns:
        Julian day number
    """
    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + A // 4

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return jd


def gravitational_time_dilation(mass: float, radius: float) -> float:
    """Calculate gravitational time dilation factor.

    t_dilated / t_proper = sqrt(1 - r_s/r)

    Returns the factor by which time passes slower near a massive body.

    Args:
        mass: Mass of body (kg)
        radius: Distance from center (m)

    Returns:
        Time dilation factor (< 1 means time passes slower)
    """
    rs = schwarzschild_radius(mass)
    if radius <= rs:
        raise ValueError("Radius must be greater than Schwarzschild radius")
    return math.sqrt(1 - rs / radius)


def hill_sphere_radius(m_planet: float, m_star: float, semi_major_axis: float,
                       eccentricity: float = 0) -> float:
    """Calculate Hill sphere radius (region of gravitational dominance).

    r_H ≈ a * (1-e) * (m_planet / (3 * m_star))^(1/3)

    Args:
        m_planet: Planet mass (kg)
        m_star: Star mass (kg)
        semi_major_axis: Orbital semi-major axis (m)
        eccentricity: Orbital eccentricity (0 to 1)

    Returns:
        Hill sphere radius in meters
    """
    if eccentricity >= 1 or eccentricity < 0:
        raise ValueError("Eccentricity must be in [0, 1)")
    return semi_major_axis * (1 - eccentricity) * ((m_planet / (3 * m_star)) ** (1/3))


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_orbital_period(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        a = float(args.get("semi_major_axis", args.get("a")))
        m = float(args.get("central_mass", args.get("mass", args.get("M"))))
        return a, m
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse orbital period args: {args}")


def _parse_escape_velocity(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        m = float(args.get("mass", args.get("M")))
        r = float(args.get("radius", args.get("r")))
        return m, r
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse escape velocity args: {args}")


def _parse_julian_day(args) -> Tuple[int, int, float]:
    if isinstance(args, dict):
        y = int(args.get("year", args.get("y")))
        m = int(args.get("month", args.get("m")))
        d = float(args.get("day", args.get("d")))
        return y, m, d
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return int(args[0]), int(args[1]), float(args[2])
    raise ValueError(f"Cannot parse julian day args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register astronomy tools in the registry."""

    registry.register(ToolSpec(
        name="astronomy_orbital_period",
        function=lambda args: orbital_period(*_parse_orbital_period(args)),
        description="Calculates orbital period using Kepler's third law.",
        parameters={
            "type": "object",
            "properties": {
                "semi_major_axis": {"type": "number", "description": "Semi-major axis (m)"},
                "central_mass": {"type": "number", "description": "Mass of central body (kg)"},
            },
            "required": ["semi_major_axis", "central_mass"]
        },
        returns="Orbital period in seconds",
        examples=[
            {"input": {"semi_major_axis": 1.496e11, "central_mass": 1.989e30}, "output": 31558118.4},
        ],
        domain="astronomy",
        tags=["orbit", "period", "kepler"],
    ))

    registry.register(ToolSpec(
        name="astronomy_escape_velocity",
        function=lambda args: escape_velocity(*_parse_escape_velocity(args)),
        description="Calculates escape velocity from a celestial body.",
        parameters={
            "type": "object",
            "properties": {
                "mass": {"type": "number", "description": "Mass of body (kg)"},
                "radius": {"type": "number", "description": "Distance from center (m)"},
            },
            "required": ["mass", "radius"]
        },
        returns="Escape velocity in m/s",
        examples=[
            {"input": {"mass": 5.972e24, "radius": 6.371e6}, "output": 11186.0},
        ],
        domain="astronomy",
        tags=["escape", "velocity"],
    ))

    registry.register(ToolSpec(
        name="astronomy_parallax_distance",
        function=lambda args: parallax_distance(
            float(args.get("parallax", args) if isinstance(args, dict) else args)
        ),
        description="Calculates distance from parallax angle (d = 1/p).",
        parameters={
            "type": "object",
            "properties": {
                "parallax": {"type": "number", "description": "Parallax angle (arcseconds)"},
            },
            "required": ["parallax"]
        },
        returns="Distance in parsecs",
        examples=[
            {"input": {"parallax": 0.1}, "output": 10.0},
        ],
        domain="astronomy",
        tags=["parallax", "distance", "parsec"],
    ))

    registry.register(ToolSpec(
        name="astronomy_moon_phase",
        function=lambda args: moon_phase(
            float(args.get("julian_day", args.get("jd", args)) if isinstance(args, dict) else args)
        ),
        description="Calculates moon phase from Julian day (0=new, 0.5=full).",
        parameters={
            "type": "object",
            "properties": {
                "julian_day": {"type": "number", "description": "Julian day number"},
            },
            "required": ["julian_day"]
        },
        returns="Phase as fraction (0 to 1)",
        examples=[
            {"input": {"julian_day": 2460000}, "output": 0.75},
        ],
        domain="astronomy",
        tags=["moon", "phase", "lunar"],
    ))

    registry.register(ToolSpec(
        name="astronomy_julian_day",
        function=lambda args: julian_day(*_parse_julian_day(args)),
        description="Converts calendar date to Julian day number.",
        parameters={
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Year"},
                "month": {"type": "integer", "description": "Month (1-12)"},
                "day": {"type": "number", "description": "Day (can be fractional)"},
            },
            "required": ["year", "month", "day"]
        },
        returns="Julian day number",
        examples=[
            {"input": {"year": 2000, "month": 1, "day": 1.5}, "output": 2451545.0},
        ],
        domain="astronomy",
        tags=["julian", "date", "calendar"],
    ))

    registry.register(ToolSpec(
        name="astronomy_angular_diameter",
        function=lambda args: angular_diameter_degrees(
            float(args.get("diameter", args.get("d")) if isinstance(args, dict) else args[0]),
            float(args.get("distance", args.get("D")) if isinstance(args, dict) else args[1])
        ),
        description="Calculates angular diameter of an object in degrees.",
        parameters={
            "type": "object",
            "properties": {
                "diameter": {"type": "number", "description": "Physical diameter"},
                "distance": {"type": "number", "description": "Distance to object"},
            },
            "required": ["diameter", "distance"]
        },
        returns="Angular diameter in degrees",
        examples=[
            {"input": {"diameter": 3474000, "distance": 384400000}, "output": 0.518},
        ],
        domain="astronomy",
        tags=["angular", "diameter", "size"],
    ))
