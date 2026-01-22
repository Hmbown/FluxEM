"""Acoustics domain - sound, decibels, frequency.

This module provides deterministic acoustics computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

SPEED_OF_SOUND = 343.0  # m/s at 20°C in air
REFERENCE_INTENSITY = 1e-12  # W/m² - threshold of hearing
REFERENCE_PRESSURE = 2e-5  # Pa - reference sound pressure


# =============================================================================
# Core Functions
# =============================================================================

def db_from_intensity(intensity: float) -> float:
    """Convert intensity to decibels.

    dB = 10 * log10(I / I₀)

    Args:
        intensity: Sound intensity (W/m²)

    Returns:
        Sound level in decibels
    """
    if intensity <= 0:
        raise ValueError("Intensity must be positive")
    return 10 * math.log10(intensity / REFERENCE_INTENSITY)


def intensity_from_db(db: float) -> float:
    """Convert decibels to intensity.

    I = I₀ * 10^(dB/10)

    Args:
        db: Sound level in decibels

    Returns:
        Sound intensity (W/m²)
    """
    return REFERENCE_INTENSITY * (10 ** (db / 10))


def db_add(db1: float, db2: float) -> float:
    """Add two sound levels in decibels.

    dB_total = 10 * log10(10^(dB1/10) + 10^(dB2/10))

    Args:
        db1: First sound level (dB)
        db2: Second sound level (dB)

    Returns:
        Combined sound level in decibels
    """
    i1 = 10 ** (db1 / 10)
    i2 = 10 ** (db2 / 10)
    return 10 * math.log10(i1 + i2)


def wavelength(frequency: float, speed: float = SPEED_OF_SOUND) -> float:
    """Calculate wavelength from frequency.

    λ = v / f

    Args:
        frequency: Frequency (Hz)
        speed: Speed of sound (m/s), default 343 m/s

    Returns:
        Wavelength in meters
    """
    if frequency <= 0:
        raise ValueError("Frequency must be positive")
    return speed / frequency


def frequency_from_wavelength(wavelength_m: float, speed: float = SPEED_OF_SOUND) -> float:
    """Calculate frequency from wavelength.

    f = v / λ

    Args:
        wavelength_m: Wavelength (m)
        speed: Speed of sound (m/s), default 343 m/s

    Returns:
        Frequency in Hz
    """
    if wavelength_m <= 0:
        raise ValueError("Wavelength must be positive")
    return speed / wavelength_m


def doppler_frequency(source_freq: float, v_source: float, v_observer: float,
                      approaching: bool = True, speed: float = SPEED_OF_SOUND) -> float:
    """Calculate observed frequency due to Doppler effect.

    f' = f * (v ± v_observer) / (v ∓ v_source)

    Args:
        source_freq: Source frequency (Hz)
        v_source: Speed of source (m/s)
        v_observer: Speed of observer (m/s)
        approaching: True if source and observer approaching each other
        speed: Speed of sound (m/s), default 343 m/s

    Returns:
        Observed frequency in Hz
    """
    if approaching:
        return source_freq * (speed + v_observer) / (speed - v_source)
    else:
        return source_freq * (speed - v_observer) / (speed + v_source)


def spl_distance(spl_ref: float, distance_ref: float, distance: float) -> float:
    """Calculate sound pressure level at a distance (inverse square law).

    SPL₂ = SPL₁ - 20 * log10(d₂/d₁)

    Args:
        spl_ref: Reference SPL (dB) at reference distance
        distance_ref: Reference distance (m)
        distance: Target distance (m)

    Returns:
        SPL at target distance in dB
    """
    if distance <= 0 or distance_ref <= 0:
        raise ValueError("Distances must be positive")
    return spl_ref - 20 * math.log10(distance / distance_ref)


def frequency_ratio_to_cents(ratio: float) -> float:
    """Convert frequency ratio to cents (musical intervals).

    cents = 1200 * log2(ratio)

    Args:
        ratio: Frequency ratio (e.g., 2.0 for octave)

    Returns:
        Interval in cents
    """
    if ratio <= 0:
        raise ValueError("Ratio must be positive")
    return 1200 * math.log2(ratio)


def cents_to_frequency_ratio(cents: float) -> float:
    """Convert cents to frequency ratio.

    ratio = 2^(cents/1200)

    Args:
        cents: Interval in cents

    Returns:
        Frequency ratio
    """
    return 2 ** (cents / 1200)


def resonance_frequency(length: float, mode: int = 1,
                        open_both_ends: bool = True,
                        speed: float = SPEED_OF_SOUND) -> float:
    """Calculate resonance frequency of a tube/pipe.

    For open both ends: f_n = n * v / (2L)
    For closed one end: f_n = (2n-1) * v / (4L)

    Args:
        length: Length of tube (m)
        mode: Harmonic mode number (1 = fundamental)
        open_both_ends: True if open at both ends
        speed: Speed of sound (m/s)

    Returns:
        Resonance frequency in Hz
    """
    if length <= 0:
        raise ValueError("Length must be positive")
    if mode < 1:
        raise ValueError("Mode must be at least 1")

    if open_both_ends:
        return mode * speed / (2 * length)
    else:
        return (2 * mode - 1) * speed / (4 * length)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_db_add(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        db1 = float(args.get("db1", args.get("level1")))
        db2 = float(args.get("db2", args.get("level2")))
        return db1, db2
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse db_add args: {args}")


def _parse_doppler(args) -> Tuple[float, float, float, bool, float]:
    if isinstance(args, dict):
        f = float(args.get("source_freq", args.get("f")))
        v_s = float(args.get("v_source", args.get("vs", 0)))
        v_o = float(args.get("v_observer", args.get("vo", 0)))
        approaching = args.get("approaching", True)
        speed = float(args.get("speed", SPEED_OF_SOUND))
        return f, v_s, v_o, approaching, speed
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        f, v_s, v_o = float(args[0]), float(args[1]), float(args[2])
        approaching = args[3] if len(args) > 3 else True
        speed = float(args[4]) if len(args) > 4 else SPEED_OF_SOUND
        return f, v_s, v_o, approaching, speed
    raise ValueError(f"Cannot parse doppler args: {args}")


def _parse_spl_distance(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        spl_ref = float(args.get("spl_ref", args.get("spl")))
        d_ref = float(args.get("distance_ref", args.get("d_ref")))
        d = float(args.get("distance", args.get("d")))
        return spl_ref, d_ref, d
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse spl_distance args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register acoustics tools in the registry."""

    registry.register(ToolSpec(
        name="acoustics_db_from_intensity",
        function=lambda args: db_from_intensity(
            float(args.get("intensity", args) if isinstance(args, dict) else args)
        ),
        description="Converts sound intensity (W/m²) to decibels.",
        parameters={
            "type": "object",
            "properties": {
                "intensity": {"type": "number", "description": "Sound intensity (W/m²)"},
            },
            "required": ["intensity"]
        },
        returns="Sound level in decibels",
        examples=[
            {"input": {"intensity": 1e-6}, "output": 60.0},
        ],
        domain="acoustics",
        tags=["decibel", "intensity", "spl"],
    ))

    registry.register(ToolSpec(
        name="acoustics_db_add",
        function=lambda args: db_add(*_parse_db_add(args)),
        description="Adds two sound levels in decibels (logarithmic addition).",
        parameters={
            "type": "object",
            "properties": {
                "db1": {"type": "number", "description": "First sound level (dB)"},
                "db2": {"type": "number", "description": "Second sound level (dB)"},
            },
            "required": ["db1", "db2"]
        },
        returns="Combined sound level in decibels",
        examples=[
            {"input": {"db1": 60, "db2": 60}, "output": 63.01},
        ],
        domain="acoustics",
        tags=["decibel", "add", "combine"],
    ))

    registry.register(ToolSpec(
        name="acoustics_wavelength",
        function=lambda args: wavelength(
            float(args.get("frequency", args.get("f")) if isinstance(args, dict) else args),
            float(args.get("speed", SPEED_OF_SOUND) if isinstance(args, dict) else SPEED_OF_SOUND)
        ),
        description="Calculates wavelength from frequency (λ = v/f).",
        parameters={
            "type": "object",
            "properties": {
                "frequency": {"type": "number", "description": "Frequency (Hz)"},
                "speed": {"type": "number", "description": "Speed of sound (m/s), default 343"},
            },
            "required": ["frequency"]
        },
        returns="Wavelength in meters",
        examples=[
            {"input": {"frequency": 440}, "output": 0.7795},
        ],
        domain="acoustics",
        tags=["wavelength", "frequency"],
    ))

    registry.register(ToolSpec(
        name="acoustics_doppler",
        function=lambda args: doppler_frequency(*_parse_doppler(args)),
        description="Calculates observed frequency due to Doppler effect.",
        parameters={
            "type": "object",
            "properties": {
                "source_freq": {"type": "number", "description": "Source frequency (Hz)"},
                "v_source": {"type": "number", "description": "Speed of source (m/s)"},
                "v_observer": {"type": "number", "description": "Speed of observer (m/s)"},
                "approaching": {"type": "boolean", "description": "True if approaching (default)"},
            },
            "required": ["source_freq", "v_source", "v_observer"]
        },
        returns="Observed frequency in Hz",
        examples=[
            {"input": {"source_freq": 440, "v_source": 30, "v_observer": 0, "approaching": True}, "output": 482.11},
        ],
        domain="acoustics",
        tags=["doppler", "frequency", "motion"],
    ))

    registry.register(ToolSpec(
        name="acoustics_spl_distance",
        function=lambda args: spl_distance(*_parse_spl_distance(args)),
        description="Calculates SPL at a distance using inverse square law.",
        parameters={
            "type": "object",
            "properties": {
                "spl_ref": {"type": "number", "description": "Reference SPL (dB)"},
                "distance_ref": {"type": "number", "description": "Reference distance (m)"},
                "distance": {"type": "number", "description": "Target distance (m)"},
            },
            "required": ["spl_ref", "distance_ref", "distance"]
        },
        returns="SPL at target distance in dB",
        examples=[
            {"input": {"spl_ref": 90, "distance_ref": 1, "distance": 10}, "output": 70.0},
        ],
        domain="acoustics",
        tags=["spl", "distance", "inverse square"],
    ))

    registry.register(ToolSpec(
        name="acoustics_frequency_ratio_cents",
        function=lambda args: frequency_ratio_to_cents(
            float(args.get("ratio", args) if isinstance(args, dict) else args)
        ),
        description="Converts frequency ratio to cents (musical intervals).",
        parameters={
            "type": "object",
            "properties": {
                "ratio": {"type": "number", "description": "Frequency ratio (e.g., 2.0 for octave)"},
            },
            "required": ["ratio"]
        },
        returns="Interval in cents",
        examples=[
            {"input": {"ratio": 2.0}, "output": 1200.0},
            {"input": {"ratio": 1.5}, "output": 701.96},
        ],
        domain="acoustics",
        tags=["cents", "ratio", "music"],
    ))
