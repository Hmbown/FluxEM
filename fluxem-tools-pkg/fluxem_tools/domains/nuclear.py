"""Nuclear domain - radioactive decay, radiation, binding energy.

This module provides deterministic nuclear physics computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

C = 299792458  # Speed of light (m/s)
U = 1.66053906660e-27  # Atomic mass unit (kg)
MEV_PER_U = 931.494  # MeV per atomic mass unit
AVOGADRO = 6.02214076e23  # Avogadro's number


# =============================================================================
# Core Functions
# =============================================================================

def radioactive_decay(initial_amount: float, decay_constant: float, time: float) -> float:
    """Calculate remaining amount after radioactive decay.

    N = N₀ * e^(-λt)

    Args:
        initial_amount: Initial amount (atoms, grams, or activity)
        decay_constant: Decay constant λ (1/time)
        time: Elapsed time

    Returns:
        Remaining amount
    """
    return initial_amount * math.exp(-decay_constant * time)


def half_life_from_decay_constant(decay_constant: float) -> float:
    """Calculate half-life from decay constant.

    t½ = ln(2) / λ

    Args:
        decay_constant: Decay constant λ

    Returns:
        Half-life
    """
    if decay_constant <= 0:
        raise ValueError("Decay constant must be positive")
    return math.log(2) / decay_constant


def decay_constant_from_half_life(half_life: float) -> float:
    """Calculate decay constant from half-life.

    λ = ln(2) / t½

    Args:
        half_life: Half-life

    Returns:
        Decay constant
    """
    if half_life <= 0:
        raise ValueError("Half-life must be positive")
    return math.log(2) / half_life


def activity(num_atoms: float, decay_constant: float) -> float:
    """Calculate radioactive activity.

    A = λN (in decays per second, i.e., Becquerels)

    Args:
        num_atoms: Number of radioactive atoms
        decay_constant: Decay constant (1/s)

    Returns:
        Activity in Becquerels
    """
    return decay_constant * num_atoms


def activity_from_mass(mass_grams: float, atomic_mass: float, half_life: float) -> float:
    """Calculate activity from mass of radioactive material.

    A = (λ * m * NA) / M

    Args:
        mass_grams: Mass of sample (g)
        atomic_mass: Atomic mass (g/mol)
        half_life: Half-life (s)

    Returns:
        Activity in Becquerels
    """
    lambda_ = math.log(2) / half_life
    num_atoms = (mass_grams / atomic_mass) * AVOGADRO
    return lambda_ * num_atoms


def binding_energy_per_nucleon(mass_defect_u: float, nucleons: int) -> float:
    """Calculate binding energy per nucleon.

    B/A = Δm * c² / A = Δm * 931.494 MeV / A

    Args:
        mass_defect_u: Mass defect in atomic mass units
        nucleons: Total number of nucleons (A = Z + N)

    Returns:
        Binding energy per nucleon in MeV
    """
    if nucleons <= 0:
        raise ValueError("Number of nucleons must be positive")
    return (mass_defect_u * MEV_PER_U) / nucleons


def mass_defect(Z: int, N: int, atomic_mass_u: float) -> float:
    """Calculate mass defect.

    Δm = Z*m_p + N*m_n - M_atom

    Args:
        Z: Atomic number (protons)
        N: Neutron number
        atomic_mass_u: Actual atomic mass (u)

    Returns:
        Mass defect in atomic mass units
    """
    # Proton mass ≈ 1.007276 u, Neutron mass ≈ 1.008665 u
    m_p = 1.007276
    m_n = 1.008665
    return Z * m_p + N * m_n - atomic_mass_u


def dose_equivalent(absorbed_dose: float, quality_factor: float) -> float:
    """Calculate dose equivalent (biological effect).

    H = D * Q

    Args:
        absorbed_dose: Absorbed dose in Gray (Gy)
        quality_factor: Quality factor (Q) - depends on radiation type
            - X-rays, gamma, beta: Q ≈ 1
            - Neutrons: Q ≈ 5-20
            - Alpha: Q ≈ 20

    Returns:
        Dose equivalent in Sieverts (Sv)
    """
    return absorbed_dose * quality_factor


def inverse_square_intensity(intensity_ref: float, distance_ref: float, distance: float) -> float:
    """Calculate radiation intensity at distance (inverse square law).

    I = I_ref * (d_ref / d)²

    Args:
        intensity_ref: Reference intensity
        distance_ref: Reference distance
        distance: Target distance

    Returns:
        Intensity at target distance
    """
    if distance <= 0 or distance_ref <= 0:
        raise ValueError("Distances must be positive")
    return intensity_ref * (distance_ref / distance) ** 2


def q_value(mass_reactants_u: float, mass_products_u: float) -> float:
    """Calculate Q-value of a nuclear reaction.

    Q = (m_reactants - m_products) * c²

    Args:
        mass_reactants_u: Total mass of reactants (u)
        mass_products_u: Total mass of products (u)

    Returns:
        Q-value in MeV (positive = exothermic)
    """
    return (mass_reactants_u - mass_products_u) * MEV_PER_U


def mean_lifetime(decay_constant: float) -> float:
    """Calculate mean lifetime of radioactive isotope.

    τ = 1/λ

    Args:
        decay_constant: Decay constant

    Returns:
        Mean lifetime
    """
    if decay_constant <= 0:
        raise ValueError("Decay constant must be positive")
    return 1 / decay_constant


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_decay(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        n0 = float(args.get("initial_amount", args.get("N0", args.get("initial"))))
        lambda_ = float(args.get("decay_constant", args.get("lambda")))
        t = float(args.get("time", args.get("t")))
        return n0, lambda_, t
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse decay args: {args}")


def _parse_binding_energy(args) -> Tuple[float, int]:
    if isinstance(args, dict):
        dm = float(args.get("mass_defect", args.get("dm")))
        A = int(args.get("nucleons", args.get("A")))
        return dm, A
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), int(args[1])
    raise ValueError(f"Cannot parse binding energy args: {args}")


def _parse_dose(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        d = float(args.get("absorbed_dose", args.get("dose", args.get("D"))))
        q = float(args.get("quality_factor", args.get("Q", 1)))
        return d, q
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse dose args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register nuclear physics tools in the registry."""

    registry.register(ToolSpec(
        name="nuclear_decay",
        function=lambda args: radioactive_decay(*_parse_decay(args)),
        description="Calculates remaining amount after radioactive decay (N = N₀e^(-λt)).",
        parameters={
            "type": "object",
            "properties": {
                "initial_amount": {"type": "number", "description": "Initial amount"},
                "decay_constant": {"type": "number", "description": "Decay constant λ"},
                "time": {"type": "number", "description": "Elapsed time"},
            },
            "required": ["initial_amount", "decay_constant", "time"]
        },
        returns="Remaining amount",
        examples=[
            {"input": {"initial_amount": 1000, "decay_constant": 0.1, "time": 10}, "output": 367.88},
        ],
        domain="nuclear",
        tags=["decay", "radioactive", "exponential"],
    ))

    registry.register(ToolSpec(
        name="nuclear_half_life",
        function=lambda args: half_life_from_decay_constant(
            float(args.get("decay_constant", args.get("lambda", args)) if isinstance(args, dict) else args)
        ),
        description="Calculates half-life from decay constant (t½ = ln(2)/λ).",
        parameters={
            "type": "object",
            "properties": {
                "decay_constant": {"type": "number", "description": "Decay constant λ"},
            },
            "required": ["decay_constant"]
        },
        returns="Half-life",
        examples=[
            {"input": {"decay_constant": 0.693}, "output": 1.0},
        ],
        domain="nuclear",
        tags=["half-life", "decay"],
    ))

    registry.register(ToolSpec(
        name="nuclear_activity",
        function=lambda args: activity(
            float(args.get("num_atoms", args.get("N", args[0])) if isinstance(args, dict) else args[0]),
            float(args.get("decay_constant", args.get("lambda", args[1])) if isinstance(args, dict) else args[1])
        ),
        description="Calculates radioactive activity (A = λN) in Becquerels.",
        parameters={
            "type": "object",
            "properties": {
                "num_atoms": {"type": "number", "description": "Number of atoms"},
                "decay_constant": {"type": "number", "description": "Decay constant (1/s)"},
            },
            "required": ["num_atoms", "decay_constant"]
        },
        returns="Activity in Becquerels",
        examples=[
            {"input": {"num_atoms": 1e10, "decay_constant": 1e-8}, "output": 100.0},
        ],
        domain="nuclear",
        tags=["activity", "becquerel", "decay"],
    ))

    registry.register(ToolSpec(
        name="nuclear_binding_energy",
        function=lambda args: binding_energy_per_nucleon(*_parse_binding_energy(args)),
        description="Calculates binding energy per nucleon in MeV.",
        parameters={
            "type": "object",
            "properties": {
                "mass_defect": {"type": "number", "description": "Mass defect in atomic mass units"},
                "nucleons": {"type": "integer", "description": "Total nucleons (A = Z + N)"},
            },
            "required": ["mass_defect", "nucleons"]
        },
        returns="Binding energy per nucleon in MeV",
        examples=[
            {"input": {"mass_defect": 0.0304, "nucleons": 4}, "output": 7.07},
        ],
        domain="nuclear",
        tags=["binding energy", "nucleon", "mass defect"],
    ))

    registry.register(ToolSpec(
        name="nuclear_dose_equivalent",
        function=lambda args: dose_equivalent(*_parse_dose(args)),
        description="Calculates dose equivalent (H = D*Q) in Sieverts.",
        parameters={
            "type": "object",
            "properties": {
                "absorbed_dose": {"type": "number", "description": "Absorbed dose in Gray"},
                "quality_factor": {"type": "number", "description": "Quality factor (1 for X/gamma/beta, 20 for alpha)"},
            },
            "required": ["absorbed_dose", "quality_factor"]
        },
        returns="Dose equivalent in Sieverts",
        examples=[
            {"input": {"absorbed_dose": 0.01, "quality_factor": 1}, "output": 0.01},
            {"input": {"absorbed_dose": 0.01, "quality_factor": 20}, "output": 0.2},
        ],
        domain="nuclear",
        tags=["dose", "sievert", "radiation"],
    ))

    registry.register(ToolSpec(
        name="nuclear_inverse_square",
        function=lambda args: inverse_square_intensity(
            float(args.get("intensity_ref", args.get("I_ref", args[0])) if isinstance(args, dict) else args[0]),
            float(args.get("distance_ref", args.get("d_ref", args[1])) if isinstance(args, dict) else args[1]),
            float(args.get("distance", args.get("d", args[2])) if isinstance(args, dict) else args[2])
        ),
        description="Calculates radiation intensity at distance (inverse square law).",
        parameters={
            "type": "object",
            "properties": {
                "intensity_ref": {"type": "number", "description": "Reference intensity"},
                "distance_ref": {"type": "number", "description": "Reference distance"},
                "distance": {"type": "number", "description": "Target distance"},
            },
            "required": ["intensity_ref", "distance_ref", "distance"]
        },
        returns="Intensity at target distance",
        examples=[
            {"input": {"intensity_ref": 100, "distance_ref": 1, "distance": 2}, "output": 25.0},
        ],
        domain="nuclear",
        tags=["intensity", "distance", "inverse square"],
    ))
