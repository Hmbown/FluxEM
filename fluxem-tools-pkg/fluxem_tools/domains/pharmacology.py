"""Pharmacology domain - drug kinetics, dosing calculations.

This module provides deterministic pharmacokinetic computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def half_life_decay(initial_amount: float, half_life: float, time: float) -> float:
    """Calculate remaining amount after radioactive/drug decay.

    A = A₀ * (1/2)^(t/t½)

    Args:
        initial_amount: Initial amount
        half_life: Half-life (same time units as time)
        time: Elapsed time

    Returns:
        Remaining amount
    """
    if half_life <= 0:
        raise ValueError("Half-life must be positive")
    return initial_amount * (0.5 ** (time / half_life))


def elimination_rate_constant(half_life: float) -> float:
    """Calculate elimination rate constant from half-life.

    k = ln(2) / t½

    Args:
        half_life: Half-life

    Returns:
        Elimination rate constant (k)
    """
    if half_life <= 0:
        raise ValueError("Half-life must be positive")
    return math.log(2) / half_life


def dosage_by_weight(dose_per_kg: float, weight: float) -> float:
    """Calculate dosage based on body weight.

    Args:
        dose_per_kg: Dose per kilogram (mg/kg)
        weight: Body weight (kg)

    Returns:
        Total dose (mg)
    """
    if weight <= 0:
        raise ValueError("Weight must be positive")
    return dose_per_kg * weight


def bioavailability_oral(auc_oral: float, auc_iv: float, dose_oral: float, dose_iv: float) -> float:
    """Calculate oral bioavailability (F).

    F = (AUC_oral / AUC_iv) * (Dose_iv / Dose_oral)

    Args:
        auc_oral: Area under curve for oral administration
        auc_iv: Area under curve for IV administration
        dose_oral: Oral dose
        dose_iv: IV dose

    Returns:
        Bioavailability as fraction (0 to 1)
    """
    if auc_iv == 0 or dose_oral == 0:
        raise ValueError("Denominators must be non-zero")
    return (auc_oral / auc_iv) * (dose_iv / dose_oral)


def clearance_rate(dose: float, auc: float) -> float:
    """Calculate drug clearance rate.

    CL = Dose / AUC

    Args:
        dose: Administered dose
        auc: Area under the plasma concentration-time curve

    Returns:
        Clearance rate
    """
    if auc == 0:
        raise ValueError("AUC must be non-zero")
    return dose / auc


def volume_of_distribution(dose: float, concentration: float) -> float:
    """Calculate apparent volume of distribution.

    Vd = Dose / C₀

    Args:
        dose: Administered dose
        concentration: Initial plasma concentration

    Returns:
        Volume of distribution
    """
    if concentration == 0:
        raise ValueError("Concentration must be non-zero")
    return dose / concentration


def steady_state_concentration(dose: float, bioavailability: float,
                               clearance: float, interval: float) -> float:
    """Calculate average steady-state plasma concentration.

    Css = (F * Dose) / (CL * τ)

    Args:
        dose: Dose per interval
        bioavailability: Bioavailability (F, 0 to 1)
        clearance: Clearance rate
        interval: Dosing interval (tau)

    Returns:
        Average steady-state concentration
    """
    if clearance == 0 or interval == 0:
        raise ValueError("Clearance and interval must be non-zero")
    return (bioavailability * dose) / (clearance * interval)


def loading_dose(target_concentration: float, volume_of_dist: float,
                 bioavailability: float = 1.0) -> float:
    """Calculate loading dose to achieve target concentration.

    Loading dose = (C_target * Vd) / F

    Args:
        target_concentration: Target plasma concentration
        volume_of_dist: Volume of distribution
        bioavailability: Bioavailability (default 1.0 for IV)

    Returns:
        Loading dose
    """
    if bioavailability <= 0:
        raise ValueError("Bioavailability must be positive")
    return (target_concentration * volume_of_dist) / bioavailability


def maintenance_dose(clearance: float, target_concentration: float,
                     bioavailability: float, interval: float) -> float:
    """Calculate maintenance dose for steady state.

    Maintenance dose = (CL * Css * τ) / F

    Args:
        clearance: Clearance rate
        target_concentration: Target steady-state concentration
        bioavailability: Bioavailability (F)
        interval: Dosing interval

    Returns:
        Maintenance dose per interval
    """
    if bioavailability <= 0:
        raise ValueError("Bioavailability must be positive")
    return (clearance * target_concentration * interval) / bioavailability


def time_to_steady_state(half_life: float, fraction: float = 0.9) -> float:
    """Calculate time to reach fraction of steady state.

    t = -t½ * log₂(1 - fraction)

    Args:
        half_life: Drug half-life
        fraction: Fraction of steady state (default 0.9 = 90%)

    Returns:
        Time to reach specified fraction of steady state
    """
    if half_life <= 0:
        raise ValueError("Half-life must be positive")
    if fraction <= 0 or fraction >= 1:
        raise ValueError("Fraction must be between 0 and 1")
    return -half_life * math.log2(1 - fraction)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_half_life_decay(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        initial = float(args.get("initial_amount", args.get("initial", args.get("A0"))))
        half_life = float(args.get("half_life", args.get("t_half")))
        time = float(args.get("time", args.get("t")))
        return initial, half_life, time
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse half_life_decay args: {args}")


def _parse_dosage_weight(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        dose_per_kg = float(args.get("dose_per_kg", args.get("dose_kg")))
        weight = float(args.get("weight", args.get("w")))
        return dose_per_kg, weight
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse dosage_weight args: {args}")


def _parse_steady_state(args) -> Tuple[float, float, float, float]:
    if isinstance(args, dict):
        dose = float(args.get("dose", args.get("D")))
        f = float(args.get("bioavailability", args.get("F", 1.0)))
        cl = float(args.get("clearance", args.get("CL")))
        tau = float(args.get("interval", args.get("tau")))
        return dose, f, cl, tau
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return float(args[0]), float(args[1]), float(args[2]), float(args[3])
    raise ValueError(f"Cannot parse steady_state args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register pharmacology tools in the registry."""

    registry.register(ToolSpec(
        name="pharma_half_life_decay",
        function=lambda args: half_life_decay(*_parse_half_life_decay(args)),
        description="Calculates remaining drug amount after exponential decay.",
        parameters={
            "type": "object",
            "properties": {
                "initial_amount": {"type": "number", "description": "Initial amount"},
                "half_life": {"type": "number", "description": "Half-life (same units as time)"},
                "time": {"type": "number", "description": "Elapsed time"},
            },
            "required": ["initial_amount", "half_life", "time"]
        },
        returns="Remaining amount",
        examples=[
            {"input": {"initial_amount": 100, "half_life": 6, "time": 12}, "output": 25.0},
        ],
        domain="pharmacology",
        tags=["decay", "half-life", "kinetics"],
    ))

    registry.register(ToolSpec(
        name="pharma_dosage_by_weight",
        function=lambda args: dosage_by_weight(*_parse_dosage_weight(args)),
        description="Calculates dosage based on body weight.",
        parameters={
            "type": "object",
            "properties": {
                "dose_per_kg": {"type": "number", "description": "Dose per kilogram (mg/kg)"},
                "weight": {"type": "number", "description": "Body weight (kg)"},
            },
            "required": ["dose_per_kg", "weight"]
        },
        returns="Total dose (mg)",
        examples=[
            {"input": {"dose_per_kg": 10, "weight": 70}, "output": 700.0},
        ],
        domain="pharmacology",
        tags=["dosage", "weight", "pediatric"],
    ))

    registry.register(ToolSpec(
        name="pharma_clearance",
        function=lambda args: clearance_rate(
            float(args.get("dose", args[0]) if isinstance(args, dict) else args[0]),
            float(args.get("auc", args[1]) if isinstance(args, dict) else args[1])
        ),
        description="Calculates drug clearance rate (CL = Dose/AUC).",
        parameters={
            "type": "object",
            "properties": {
                "dose": {"type": "number", "description": "Administered dose"},
                "auc": {"type": "number", "description": "Area under curve"},
            },
            "required": ["dose", "auc"]
        },
        returns="Clearance rate",
        examples=[
            {"input": {"dose": 500, "auc": 50}, "output": 10.0},
        ],
        domain="pharmacology",
        tags=["clearance", "kinetics"],
    ))

    registry.register(ToolSpec(
        name="pharma_volume_distribution",
        function=lambda args: volume_of_distribution(
            float(args.get("dose", args[0]) if isinstance(args, dict) else args[0]),
            float(args.get("concentration", args.get("C0", args[1])) if isinstance(args, dict) else args[1])
        ),
        description="Calculates apparent volume of distribution (Vd = Dose/C₀).",
        parameters={
            "type": "object",
            "properties": {
                "dose": {"type": "number", "description": "Administered dose"},
                "concentration": {"type": "number", "description": "Initial plasma concentration"},
            },
            "required": ["dose", "concentration"]
        },
        returns="Volume of distribution",
        examples=[
            {"input": {"dose": 500, "concentration": 10}, "output": 50.0},
        ],
        domain="pharmacology",
        tags=["volume", "distribution", "vd"],
    ))

    registry.register(ToolSpec(
        name="pharma_steady_state",
        function=lambda args: steady_state_concentration(*_parse_steady_state(args)),
        description="Calculates average steady-state plasma concentration.",
        parameters={
            "type": "object",
            "properties": {
                "dose": {"type": "number", "description": "Dose per interval"},
                "bioavailability": {"type": "number", "description": "Bioavailability (F, 0-1)"},
                "clearance": {"type": "number", "description": "Clearance rate"},
                "interval": {"type": "number", "description": "Dosing interval"},
            },
            "required": ["dose", "bioavailability", "clearance", "interval"]
        },
        returns="Average steady-state concentration",
        examples=[
            {"input": {"dose": 500, "bioavailability": 0.8, "clearance": 10, "interval": 8}, "output": 5.0},
        ],
        domain="pharmacology",
        tags=["steady state", "css", "dosing"],
    ))

    registry.register(ToolSpec(
        name="pharma_time_to_steady_state",
        function=lambda args: time_to_steady_state(
            float(args.get("half_life", args.get("t_half", args)) if isinstance(args, dict) else args[0] if isinstance(args, (list, tuple)) else args),
            float(args.get("fraction", 0.9) if isinstance(args, dict) else (args[1] if isinstance(args, (list, tuple)) and len(args) > 1 else 0.9))
        ),
        description="Calculates time to reach fraction of steady state.",
        parameters={
            "type": "object",
            "properties": {
                "half_life": {"type": "number", "description": "Drug half-life"},
                "fraction": {"type": "number", "description": "Fraction of steady state (default 0.9)"},
            },
            "required": ["half_life"]
        },
        returns="Time to reach specified fraction of steady state",
        examples=[
            {"input": {"half_life": 6, "fraction": 0.9}, "output": 19.93},
        ],
        domain="pharmacology",
        tags=["steady state", "time", "kinetics"],
    ))
