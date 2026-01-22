"""Fluid dynamics domain - flow, pressure, viscosity.

This module provides deterministic fluid dynamics computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

RHO_WATER = 1000  # kg/m³ - density of water at 20°C
RHO_AIR = 1.225  # kg/m³ - density of air at sea level
G = 9.80665  # m/s² - gravitational acceleration


# =============================================================================
# Core Functions
# =============================================================================

def reynolds_number(velocity: float, length: float, kinematic_viscosity: float) -> float:
    """Calculate Reynolds number (dimensionless flow regime indicator).

    Re = v * L / ν

    Args:
        velocity: Flow velocity (m/s)
        length: Characteristic length (m) - e.g., pipe diameter
        kinematic_viscosity: Kinematic viscosity (m²/s)

    Returns:
        Reynolds number (dimensionless)
    """
    if kinematic_viscosity <= 0:
        raise ValueError("Kinematic viscosity must be positive")
    return velocity * length / kinematic_viscosity


def is_turbulent(reynolds: float) -> bool:
    """Determine if flow is turbulent based on Reynolds number.

    Transition typically occurs around Re = 2300-4000 for pipe flow.

    Args:
        reynolds: Reynolds number

    Returns:
        True if turbulent (Re > 4000)
    """
    return reynolds > 4000


def bernoulli_velocity(pressure_diff: float, density: float) -> float:
    """Calculate flow velocity from pressure difference (Bernoulli).

    v = sqrt(2 * ΔP / ρ)

    Args:
        pressure_diff: Pressure difference (Pa)
        density: Fluid density (kg/m³)

    Returns:
        Velocity in m/s
    """
    if pressure_diff < 0:
        pressure_diff = abs(pressure_diff)  # Direction doesn't matter for magnitude
    if density <= 0:
        raise ValueError("Density must be positive")
    return math.sqrt(2 * pressure_diff / density)


def drag_force(cd: float, density: float, velocity: float, area: float) -> float:
    """Calculate drag force.

    F_d = ½ * C_d * ρ * v² * A

    Args:
        cd: Drag coefficient (dimensionless)
        density: Fluid density (kg/m³)
        velocity: Flow velocity (m/s)
        area: Reference area (m²)

    Returns:
        Drag force in Newtons
    """
    return 0.5 * cd * density * (velocity ** 2) * area


def flow_rate_pipe(velocity: float, diameter: float) -> float:
    """Calculate volumetric flow rate through a circular pipe.

    Q = v * π * (d/2)²

    Args:
        velocity: Flow velocity (m/s)
        diameter: Pipe diameter (m)

    Returns:
        Volumetric flow rate (m³/s)
    """
    radius = diameter / 2
    return velocity * math.pi * (radius ** 2)


def pressure_depth(density: float, depth: float, surface_pressure: float = 101325) -> float:
    """Calculate pressure at depth in a fluid.

    P = P₀ + ρgh

    Args:
        density: Fluid density (kg/m³)
        depth: Depth below surface (m)
        surface_pressure: Pressure at surface (Pa), default atmospheric

    Returns:
        Pressure in Pascals
    """
    return surface_pressure + density * G * depth


def buoyancy_force(fluid_density: float, displaced_volume: float) -> float:
    """Calculate buoyancy force (Archimedes' principle).

    F_b = ρ_fluid * V * g

    Args:
        fluid_density: Density of fluid (kg/m³)
        displaced_volume: Volume of fluid displaced (m³)

    Returns:
        Buoyancy force in Newtons
    """
    return fluid_density * displaced_volume * G


def terminal_velocity(mass: float, drag_coeff: float, density: float, area: float) -> float:
    """Calculate terminal velocity of a falling object.

    v_t = sqrt(2 * m * g / (ρ * C_d * A))

    Args:
        mass: Object mass (kg)
        drag_coeff: Drag coefficient
        density: Fluid density (kg/m³)
        area: Cross-sectional area (m²)

    Returns:
        Terminal velocity in m/s
    """
    if drag_coeff <= 0 or density <= 0 or area <= 0:
        raise ValueError("Drag coefficient, density, and area must be positive")
    return math.sqrt(2 * mass * G / (density * drag_coeff * area))


def hagen_poiseuille_flow(pressure_diff: float, radius: float,
                          length: float, viscosity: float) -> float:
    """Calculate volumetric flow rate through a pipe (laminar flow).

    Q = (π * r⁴ * ΔP) / (8 * μ * L)

    Args:
        pressure_diff: Pressure difference (Pa)
        radius: Pipe radius (m)
        length: Pipe length (m)
        viscosity: Dynamic viscosity (Pa·s)

    Returns:
        Volumetric flow rate (m³/s)
    """
    if viscosity <= 0 or length <= 0:
        raise ValueError("Viscosity and length must be positive")
    return (math.pi * (radius ** 4) * pressure_diff) / (8 * viscosity * length)


def stokes_drag(viscosity: float, radius: float, velocity: float) -> float:
    """Calculate Stokes drag on a sphere (low Reynolds number).

    F = 6 * π * μ * r * v

    Args:
        viscosity: Dynamic viscosity (Pa·s)
        radius: Sphere radius (m)
        velocity: Velocity (m/s)

    Returns:
        Drag force in Newtons
    """
    return 6 * math.pi * viscosity * radius * velocity


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_reynolds(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        v = float(args.get("velocity", args.get("v")))
        L = float(args.get("length", args.get("L", args.get("diameter"))))
        nu = float(args.get("kinematic_viscosity", args.get("nu")))
        return v, L, nu
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse reynolds args: {args}")


def _parse_drag(args) -> Tuple[float, float, float, float]:
    if isinstance(args, dict):
        cd = float(args.get("cd", args.get("drag_coeff")))
        rho = float(args.get("density", args.get("rho")))
        v = float(args.get("velocity", args.get("v")))
        a = float(args.get("area", args.get("A")))
        return cd, rho, v, a
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return float(args[0]), float(args[1]), float(args[2]), float(args[3])
    raise ValueError(f"Cannot parse drag args: {args}")


def _parse_pressure_depth(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        rho = float(args.get("density", args.get("rho", RHO_WATER)))
        depth = float(args.get("depth", args.get("h")))
        p0 = float(args.get("surface_pressure", args.get("P0", 101325)))
        return rho, depth, p0
    if isinstance(args, (list, tuple)):
        rho = float(args[0]) if len(args) > 0 else RHO_WATER
        depth = float(args[1]) if len(args) > 1 else 0
        p0 = float(args[2]) if len(args) > 2 else 101325
        return rho, depth, p0
    raise ValueError(f"Cannot parse pressure_depth args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register fluid dynamics tools in the registry."""

    registry.register(ToolSpec(
        name="fluid_reynolds_number",
        function=lambda args: reynolds_number(*_parse_reynolds(args)),
        description="Calculates Reynolds number (Re = vL/ν) to determine flow regime.",
        parameters={
            "type": "object",
            "properties": {
                "velocity": {"type": "number", "description": "Flow velocity (m/s)"},
                "length": {"type": "number", "description": "Characteristic length (m)"},
                "kinematic_viscosity": {"type": "number", "description": "Kinematic viscosity (m²/s)"},
            },
            "required": ["velocity", "length", "kinematic_viscosity"]
        },
        returns="Reynolds number (dimensionless)",
        examples=[
            {"input": {"velocity": 2, "length": 0.05, "kinematic_viscosity": 1e-6}, "output": 100000.0},
        ],
        domain="fluid_dynamics",
        tags=["reynolds", "flow", "turbulence"],
    ))

    def _parse_bernoulli(args):
        if isinstance(args, dict):
            dp = args.get("pressure_diff", args.get("dP"))
            rho = args.get("density", args.get("rho"))
            if dp is None or rho is None:
                raise ValueError("Required: pressure_diff and density")
            return bernoulli_velocity(float(dp), float(rho))
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            return bernoulli_velocity(float(args[0]), float(args[1]))
        raise ValueError(f"Cannot parse bernoulli args: {args}")

    registry.register(ToolSpec(
        name="fluid_bernoulli_velocity",
        function=_parse_bernoulli,
        description="Calculates flow velocity from pressure difference using Bernoulli.",
        parameters={
            "type": "object",
            "properties": {
                "pressure_diff": {"type": "number", "description": "Pressure difference (Pa)"},
                "density": {"type": "number", "description": "Fluid density (kg/m³)"},
            },
            "required": ["pressure_diff", "density"]
        },
        returns="Velocity in m/s",
        examples=[
            {"input": {"pressure_diff": 500, "density": 1000}, "output": 1.0},
        ],
        domain="fluid_dynamics",
        tags=["bernoulli", "velocity", "pressure"],
    ))

    registry.register(ToolSpec(
        name="fluid_drag_force",
        function=lambda args: drag_force(*_parse_drag(args)),
        description="Calculates drag force (F = ½CdρV²A).",
        parameters={
            "type": "object",
            "properties": {
                "cd": {"type": "number", "description": "Drag coefficient"},
                "density": {"type": "number", "description": "Fluid density (kg/m³)"},
                "velocity": {"type": "number", "description": "Flow velocity (m/s)"},
                "area": {"type": "number", "description": "Reference area (m²)"},
            },
            "required": ["cd", "density", "velocity", "area"]
        },
        returns="Drag force in Newtons",
        examples=[
            {"input": {"cd": 0.5, "density": 1.225, "velocity": 10, "area": 1}, "output": 30.625},
        ],
        domain="fluid_dynamics",
        tags=["drag", "force", "aerodynamics"],
    ))

    def _parse_flow_rate(args):
        if isinstance(args, dict):
            v = args.get("velocity", args.get("v"))
            d = args.get("diameter", args.get("d"))
            if v is None or d is None:
                raise ValueError("Required: velocity and diameter")
            return flow_rate_pipe(float(v), float(d))
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            return flow_rate_pipe(float(args[0]), float(args[1]))
        raise ValueError(f"Cannot parse flow_rate args: {args}")

    registry.register(ToolSpec(
        name="fluid_flow_rate",
        function=_parse_flow_rate,
        description="Calculates volumetric flow rate through a circular pipe.",
        parameters={
            "type": "object",
            "properties": {
                "velocity": {"type": "number", "description": "Flow velocity (m/s)"},
                "diameter": {"type": "number", "description": "Pipe diameter (m)"},
            },
            "required": ["velocity", "diameter"]
        },
        returns="Volumetric flow rate (m³/s)",
        examples=[
            {"input": {"velocity": 2, "diameter": 0.1}, "output": 0.01571},
        ],
        domain="fluid_dynamics",
        tags=["flow rate", "pipe", "volume"],
    ))

    registry.register(ToolSpec(
        name="fluid_pressure_depth",
        function=lambda args: pressure_depth(*_parse_pressure_depth(args)),
        description="Calculates pressure at depth in a fluid (P = P₀ + ρgh).",
        parameters={
            "type": "object",
            "properties": {
                "density": {"type": "number", "description": "Fluid density (kg/m³), default water"},
                "depth": {"type": "number", "description": "Depth below surface (m)"},
                "surface_pressure": {"type": "number", "description": "Surface pressure (Pa), default 101325"},
            },
            "required": ["depth"]
        },
        returns="Pressure in Pascals",
        examples=[
            {"input": {"density": 1000, "depth": 10}, "output": 199391.65},
        ],
        domain="fluid_dynamics",
        tags=["pressure", "depth", "hydrostatic"],
    ))

    def _parse_buoyancy(args):
        if isinstance(args, dict):
            rho = args.get("fluid_density", args.get("density", args.get("rho")))
            vol = args.get("displaced_volume", args.get("volume", args.get("V")))
            if rho is None or vol is None:
                raise ValueError("Required: fluid_density and displaced_volume")
            return buoyancy_force(float(rho), float(vol))
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            return buoyancy_force(float(args[0]), float(args[1]))
        raise ValueError(f"Cannot parse buoyancy args: {args}")

    registry.register(ToolSpec(
        name="fluid_buoyancy",
        function=_parse_buoyancy,
        description="Calculates buoyancy force (Archimedes' principle).",
        parameters={
            "type": "object",
            "properties": {
                "fluid_density": {"type": "number", "description": "Fluid density (kg/m³)"},
                "displaced_volume": {"type": "number", "description": "Displaced volume (m³)"},
            },
            "required": ["fluid_density", "displaced_volume"]
        },
        returns="Buoyancy force in Newtons",
        examples=[
            {"input": {"fluid_density": 1000, "displaced_volume": 0.1}, "output": 980.665},
        ],
        domain="fluid_dynamics",
        tags=["buoyancy", "archimedes", "float"],
    ))
