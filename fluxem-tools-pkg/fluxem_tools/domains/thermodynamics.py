"""Thermodynamics domain - heat transfer, gas laws, efficiency.

This module provides deterministic thermodynamics computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

R_GAS = 8.314462  # J/(mol·K) - Universal gas constant
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)


# =============================================================================
# Core Functions
# =============================================================================

def heat_transfer_conduction(k: float, area: float, temp_diff: float, thickness: float) -> float:
    """Calculate heat transfer rate by conduction (Fourier's law).

    Q = k * A * ΔT / d

    Args:
        k: Thermal conductivity (W/(m·K))
        area: Cross-sectional area (m²)
        temp_diff: Temperature difference (K or °C)
        thickness: Material thickness (m)

    Returns:
        Heat transfer rate in Watts
    """
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    return k * area * temp_diff / thickness


def ideal_gas_pressure(n: float, temp: float, volume: float) -> float:
    """Calculate pressure using ideal gas law.

    PV = nRT → P = nRT/V

    Args:
        n: Amount of substance (mol)
        temp: Temperature (K)
        volume: Volume (m³)

    Returns:
        Pressure in Pascals
    """
    if volume <= 0:
        raise ValueError("Volume must be positive")
    if temp <= 0:
        raise ValueError("Temperature must be positive (Kelvin)")
    return n * R_GAS * temp / volume


def ideal_gas_volume(n: float, temp: float, pressure: float) -> float:
    """Calculate volume using ideal gas law.

    PV = nRT → V = nRT/P

    Args:
        n: Amount of substance (mol)
        temp: Temperature (K)
        pressure: Pressure (Pa)

    Returns:
        Volume in m³
    """
    if pressure <= 0:
        raise ValueError("Pressure must be positive")
    if temp <= 0:
        raise ValueError("Temperature must be positive (Kelvin)")
    return n * R_GAS * temp / pressure


def carnot_efficiency(t_hot: float, t_cold: float) -> float:
    """Calculate Carnot efficiency (maximum theoretical efficiency).

    η = 1 - T_cold/T_hot

    Args:
        t_hot: Hot reservoir temperature (K)
        t_cold: Cold reservoir temperature (K)

    Returns:
        Efficiency as fraction (0 to 1)
    """
    if t_hot <= 0 or t_cold <= 0:
        raise ValueError("Temperatures must be positive (Kelvin)")
    if t_cold >= t_hot:
        raise ValueError("Hot temperature must exceed cold temperature")
    return 1 - t_cold / t_hot


def entropy_change_isothermal(n: float, v_final: float, v_initial: float) -> float:
    """Calculate entropy change for isothermal process.

    ΔS = nR * ln(V_f/V_i)

    Args:
        n: Amount of substance (mol)
        v_final: Final volume (m³)
        v_initial: Initial volume (m³)

    Returns:
        Entropy change in J/K
    """
    if v_final <= 0 or v_initial <= 0:
        raise ValueError("Volumes must be positive")
    return n * R_GAS * math.log(v_final / v_initial)


def work_isothermal(n: float, temp: float, v_final: float, v_initial: float) -> float:
    """Calculate work done in isothermal expansion/compression.

    W = nRT * ln(V_f/V_i)

    Args:
        n: Amount of substance (mol)
        temp: Temperature (K)
        v_final: Final volume (m³)
        v_initial: Initial volume (m³)

    Returns:
        Work in Joules (positive = work done BY system)
    """
    if v_final <= 0 or v_initial <= 0:
        raise ValueError("Volumes must be positive")
    if temp <= 0:
        raise ValueError("Temperature must be positive (Kelvin)")
    return n * R_GAS * temp * math.log(v_final / v_initial)


def specific_heat_energy(mass: float, specific_heat: float, temp_change: float) -> float:
    """Calculate heat energy using specific heat capacity.

    Q = m * c * ΔT

    Args:
        mass: Mass (kg)
        specific_heat: Specific heat capacity (J/(kg·K))
        temp_change: Temperature change (K or °C)

    Returns:
        Heat energy in Joules
    """
    return mass * specific_heat * temp_change


def stefan_boltzmann_power(emissivity: float, area: float, temp: float) -> float:
    """Calculate radiated power using Stefan-Boltzmann law.

    P = ε * σ * A * T⁴

    Args:
        emissivity: Surface emissivity (0 to 1)
        area: Surface area (m²)
        temp: Temperature (K)

    Returns:
        Radiated power in Watts
    """
    if temp <= 0:
        raise ValueError("Temperature must be positive (Kelvin)")
    if not 0 <= emissivity <= 1:
        raise ValueError("Emissivity must be between 0 and 1")
    return emissivity * STEFAN_BOLTZMANN * area * (temp ** 4)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_conduction_args(args) -> Tuple[float, float, float, float]:
    if isinstance(args, dict):
        k = float(args.get("k", args.get("conductivity")))
        area = float(args.get("area", args.get("A")))
        temp_diff = float(args.get("temp_diff", args.get("delta_t", args.get("dt"))))
        thickness = float(args.get("thickness", args.get("d")))
        return k, area, temp_diff, thickness
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return float(args[0]), float(args[1]), float(args[2]), float(args[3])
    raise ValueError(f"Cannot parse conduction args: {args}")


def _parse_ideal_gas_p(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        n = float(args.get("n", args.get("moles")))
        temp = float(args.get("temp", args.get("T", args.get("temperature"))))
        volume = float(args.get("volume", args.get("V")))
        return n, temp, volume
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse ideal gas args: {args}")


def _parse_ideal_gas_v(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        n = float(args.get("n", args.get("moles")))
        temp = float(args.get("temp", args.get("T", args.get("temperature"))))
        pressure = float(args.get("pressure", args.get("P")))
        return n, temp, pressure
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse ideal gas args: {args}")


def _parse_carnot_args(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        t_hot = float(args.get("t_hot", args.get("T_h", args.get("hot"))))
        t_cold = float(args.get("t_cold", args.get("T_c", args.get("cold"))))
        return t_hot, t_cold
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse Carnot args: {args}")


def _parse_specific_heat_args(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        mass = float(args.get("mass", args.get("m")))
        c = float(args.get("specific_heat", args.get("c")))
        dt = float(args.get("temp_change", args.get("delta_t", args.get("dt"))))
        return mass, c, dt
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse specific heat args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register thermodynamics tools in the registry."""

    registry.register(ToolSpec(
        name="thermo_heat_conduction",
        function=lambda args: heat_transfer_conduction(*_parse_conduction_args(args)),
        description="Calculates heat transfer rate by conduction using Fourier's law.",
        parameters={
            "type": "object",
            "properties": {
                "k": {"type": "number", "description": "Thermal conductivity (W/(m·K))"},
                "area": {"type": "number", "description": "Cross-sectional area (m²)"},
                "temp_diff": {"type": "number", "description": "Temperature difference (K)"},
                "thickness": {"type": "number", "description": "Material thickness (m)"},
            },
            "required": ["k", "area", "temp_diff", "thickness"]
        },
        returns="Heat transfer rate in Watts",
        examples=[
            {"input": {"k": 205, "area": 0.01, "temp_diff": 100, "thickness": 0.05}, "output": 4100.0},
        ],
        domain="thermodynamics",
        tags=["heat", "conduction", "fourier"],
    ))

    registry.register(ToolSpec(
        name="thermo_ideal_gas_pressure",
        function=lambda args: ideal_gas_pressure(*_parse_ideal_gas_p(args)),
        description="Calculates gas pressure using the ideal gas law (PV = nRT).",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "number", "description": "Amount of substance (mol)"},
                "temp": {"type": "number", "description": "Temperature (K)"},
                "volume": {"type": "number", "description": "Volume (m³)"},
            },
            "required": ["n", "temp", "volume"]
        },
        returns="Pressure in Pascals",
        examples=[
            {"input": {"n": 1, "temp": 300, "volume": 0.0224}, "output": 111347.8},
        ],
        domain="thermodynamics",
        tags=["gas", "pressure", "ideal"],
    ))

    registry.register(ToolSpec(
        name="thermo_ideal_gas_volume",
        function=lambda args: ideal_gas_volume(*_parse_ideal_gas_v(args)),
        description="Calculates gas volume using the ideal gas law (PV = nRT).",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "number", "description": "Amount of substance (mol)"},
                "temp": {"type": "number", "description": "Temperature (K)"},
                "pressure": {"type": "number", "description": "Pressure (Pa)"},
            },
            "required": ["n", "temp", "pressure"]
        },
        returns="Volume in m³",
        examples=[
            {"input": {"n": 1, "temp": 273.15, "pressure": 101325}, "output": 0.02241},
        ],
        domain="thermodynamics",
        tags=["gas", "volume", "ideal"],
    ))

    registry.register(ToolSpec(
        name="thermo_carnot_efficiency",
        function=lambda args: carnot_efficiency(*_parse_carnot_args(args)),
        description="Calculates maximum theoretical (Carnot) efficiency of a heat engine.",
        parameters={
            "type": "object",
            "properties": {
                "t_hot": {"type": "number", "description": "Hot reservoir temperature (K)"},
                "t_cold": {"type": "number", "description": "Cold reservoir temperature (K)"},
            },
            "required": ["t_hot", "t_cold"]
        },
        returns="Efficiency as fraction (0 to 1)",
        examples=[
            {"input": {"t_hot": 500, "t_cold": 300}, "output": 0.4},
        ],
        domain="thermodynamics",
        tags=["efficiency", "carnot", "engine"],
    ))

    registry.register(ToolSpec(
        name="thermo_specific_heat_energy",
        function=lambda args: specific_heat_energy(*_parse_specific_heat_args(args)),
        description="Calculates heat energy using specific heat capacity (Q = mcΔT).",
        parameters={
            "type": "object",
            "properties": {
                "mass": {"type": "number", "description": "Mass (kg)"},
                "specific_heat": {"type": "number", "description": "Specific heat capacity (J/(kg·K))"},
                "temp_change": {"type": "number", "description": "Temperature change (K or °C)"},
            },
            "required": ["mass", "specific_heat", "temp_change"]
        },
        returns="Heat energy in Joules",
        examples=[
            {"input": {"mass": 1, "specific_heat": 4186, "temp_change": 10}, "output": 41860.0},
        ],
        domain="thermodynamics",
        tags=["heat", "specific heat", "energy"],
    ))

    registry.register(ToolSpec(
        name="thermo_stefan_boltzmann",
        function=lambda args: stefan_boltzmann_power(
            float(args.get("emissivity", args.get("e", 1.0)) if isinstance(args, dict) else args[0]),
            float(args.get("area", args.get("A")) if isinstance(args, dict) else args[1]),
            float(args.get("temp", args.get("T")) if isinstance(args, dict) else args[2])
        ),
        description="Calculates radiated power using Stefan-Boltzmann law (P = εσAT⁴).",
        parameters={
            "type": "object",
            "properties": {
                "emissivity": {"type": "number", "description": "Surface emissivity (0 to 1, default=1)"},
                "area": {"type": "number", "description": "Surface area (m²)"},
                "temp": {"type": "number", "description": "Temperature (K)"},
            },
            "required": ["area", "temp"]
        },
        returns="Radiated power in Watts",
        examples=[
            {"input": {"emissivity": 1, "area": 1, "temp": 1000}, "output": 56703.74},
        ],
        domain="thermodynamics",
        tags=["radiation", "blackbody", "stefan-boltzmann"],
    ))

    registry.register(ToolSpec(
        name="thermo_work_isothermal",
        function=lambda args: work_isothermal(
            float(args.get("n", args.get("moles")) if isinstance(args, dict) else args[0]),
            float(args.get("temp", args.get("T")) if isinstance(args, dict) else args[1]),
            float(args.get("v_final", args.get("V_f")) if isinstance(args, dict) else args[2]),
            float(args.get("v_initial", args.get("V_i")) if isinstance(args, dict) else args[3])
        ),
        description="Calculates work done in isothermal expansion/compression.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "number", "description": "Amount of substance (mol)"},
                "temp": {"type": "number", "description": "Temperature (K)"},
                "v_final": {"type": "number", "description": "Final volume (m³)"},
                "v_initial": {"type": "number", "description": "Initial volume (m³)"},
            },
            "required": ["n", "temp", "v_final", "v_initial"]
        },
        returns="Work in Joules (positive = work done BY system)",
        examples=[
            {"input": {"n": 1, "temp": 300, "v_final": 0.2, "v_initial": 0.1}, "output": 1729.13},
        ],
        domain="thermodynamics",
        tags=["work", "isothermal", "expansion"],
    ))
