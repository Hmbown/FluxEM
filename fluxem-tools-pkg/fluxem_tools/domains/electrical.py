"""Electrical engineering domain - circuits, Ohm's law, power calculations.

This module provides deterministic electrical engineering computations.
"""

import math
from typing import Any, Dict, List, Optional, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def ohms_law(voltage: float = None, current: float = None, resistance: float = None) -> Dict[str, float]:
    """Apply Ohm's Law: V = I * R

    Provide any two values to calculate the third.

    Args:
        voltage: Voltage in Volts (V)
        current: Current in Amperes (A)
        resistance: Resistance in Ohms (Ω)

    Returns:
        Dict with all three values (calculated and given)
    """
    count = sum(x is not None for x in [voltage, current, resistance])
    if count != 2:
        raise ValueError("Provide exactly 2 of: voltage, current, resistance")

    if voltage is None:
        voltage = current * resistance
    elif current is None:
        current = voltage / resistance
    else:  # resistance is None
        resistance = voltage / current

    return {
        "voltage": voltage,
        "current": current,
        "resistance": resistance,
    }


def power_dc(voltage: float = None, current: float = None, resistance: float = None, power: float = None) -> Dict[str, float]:
    """Calculate DC power using P = IV = I²R = V²/R

    Provide any two values to calculate power and the others.

    Args:
        voltage: Voltage in Volts (V)
        current: Current in Amperes (A)
        resistance: Resistance in Ohms (Ω)
        power: Power in Watts (W)

    Returns:
        Dict with power and related values
    """
    count = sum(x is not None for x in [voltage, current, resistance, power])

    if count < 2:
        raise ValueError("Provide at least 2 values")

    # Calculate power from given values
    if power is None:
        if voltage is not None and current is not None:
            power = voltage * current
        elif current is not None and resistance is not None:
            power = current ** 2 * resistance
        elif voltage is not None and resistance is not None:
            power = voltage ** 2 / resistance
        else:
            raise ValueError("Cannot calculate power from given values")

    # Fill in missing values if possible
    result = {"power": power}
    if voltage is not None:
        result["voltage"] = voltage
    if current is not None:
        result["current"] = current
    if resistance is not None:
        result["resistance"] = resistance

    return result


def impedance_magnitude(resistance: float, reactance: float) -> float:
    """Calculate impedance magnitude: |Z| = sqrt(R² + X²)

    Args:
        resistance: Resistance R in Ohms
        reactance: Reactance X in Ohms (positive for inductive, negative for capacitive)

    Returns:
        Impedance magnitude in Ohms
    """
    return math.sqrt(resistance ** 2 + reactance ** 2)


def resonant_frequency(inductance: float, capacitance: float) -> float:
    """Calculate resonant frequency of an LC circuit: f = 1 / (2π√(LC))

    Args:
        inductance: Inductance in Henries (H)
        capacitance: Capacitance in Farads (F)

    Returns:
        Resonant frequency in Hertz (Hz)
    """
    return 1.0 / (2.0 * math.pi * math.sqrt(inductance * capacitance))


def voltage_divider(vin: float, r1: float, r2: float) -> float:
    """Calculate output voltage of a voltage divider: Vout = Vin * R2 / (R1 + R2)

    Args:
        vin: Input voltage in Volts
        r1: First resistor (between Vin and Vout) in Ohms
        r2: Second resistor (between Vout and ground) in Ohms

    Returns:
        Output voltage in Volts
    """
    return vin * r2 / (r1 + r2)


def series_resistance(resistances: List[float]) -> float:
    """Calculate total resistance of resistors in series: Rt = R1 + R2 + ...

    Args:
        resistances: List of resistance values in Ohms

    Returns:
        Total resistance in Ohms
    """
    return sum(resistances)


def parallel_resistance(resistances: List[float]) -> float:
    """Calculate total resistance of resistors in parallel: 1/Rt = 1/R1 + 1/R2 + ...

    Args:
        resistances: List of resistance values in Ohms

    Returns:
        Total resistance in Ohms
    """
    if any(r == 0 for r in resistances):
        return 0.0
    return 1.0 / sum(1.0 / r for r in resistances)


def capacitor_energy(capacitance: float, voltage: float) -> float:
    """Calculate energy stored in a capacitor: E = 0.5 * C * V²

    Args:
        capacitance: Capacitance in Farads (F)
        voltage: Voltage across capacitor in Volts (V)

    Returns:
        Energy in Joules (J)
    """
    return 0.5 * capacitance * voltage ** 2


def inductor_energy(inductance: float, current: float) -> float:
    """Calculate energy stored in an inductor: E = 0.5 * L * I²

    Args:
        inductance: Inductance in Henries (H)
        current: Current through inductor in Amperes (A)

    Returns:
        Energy in Joules (J)
    """
    return 0.5 * inductance * current ** 2


def rc_time_constant(resistance: float, capacitance: float) -> float:
    """Calculate RC time constant: τ = R * C

    Args:
        resistance: Resistance in Ohms (Ω)
        capacitance: Capacitance in Farads (F)

    Returns:
        Time constant in seconds
    """
    return resistance * capacitance


def rl_time_constant(resistance: float, inductance: float) -> float:
    """Calculate RL time constant: τ = L / R

    Args:
        resistance: Resistance in Ohms (Ω)
        inductance: Inductance in Henries (H)

    Returns:
        Time constant in seconds
    """
    return inductance / resistance


# =============================================================================
# Argument Parsing Helpers
# =============================================================================

def _parse_ohms_law(args):
    if isinstance(args, dict):
        return ohms_law(
            voltage=args.get("voltage", args.get("v")),
            current=args.get("current", args.get("i")),
            resistance=args.get("resistance", args.get("r")),
        )
    raise ValueError("Expected dict with voltage/current/resistance")


def _parse_power(args):
    if isinstance(args, dict):
        return power_dc(
            voltage=args.get("voltage", args.get("v")),
            current=args.get("current", args.get("i")),
            resistance=args.get("resistance", args.get("r")),
            power=args.get("power", args.get("p")),
        )
    raise ValueError("Expected dict with at least 2 of: voltage, current, resistance, power")


def _parse_impedance(args):
    if isinstance(args, dict):
        r = args.get("resistance", args.get("r", 0))
        x = args.get("reactance", args.get("x", 0))
        return impedance_magnitude(r, x)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return impedance_magnitude(args[0], args[1])
    raise ValueError("Expected {resistance, reactance} or [R, X]")


def _parse_resonance(args):
    if isinstance(args, dict):
        l = args.get("inductance", args.get("l"))
        c = args.get("capacitance", args.get("c"))
        return resonant_frequency(l, c)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return resonant_frequency(args[0], args[1])
    raise ValueError("Expected {inductance, capacitance} or [L, C]")


def _parse_voltage_divider(args):
    if isinstance(args, dict):
        vin = args.get("vin", args.get("voltage_in", args.get("v_in")))
        r1 = args.get("r1", args.get("resistance1"))
        r2 = args.get("r2", args.get("resistance2"))
        return voltage_divider(vin, r1, r2)
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return voltage_divider(args[0], args[1], args[2])
    raise ValueError("Expected {vin, r1, r2}")


def _parse_resistances(args):
    if isinstance(args, dict):
        if "resistances" in args:
            return args["resistances"]
        if "r" in args:
            return args["r"]
        return list(args.values())
    if isinstance(args, (list, tuple)):
        return list(args)
    raise ValueError("Expected list of resistances")


def _parse_capacitor_energy(args):
    if isinstance(args, dict):
        c = args.get("capacitance", args.get("c"))
        v = args.get("voltage", args.get("v"))
        return capacitor_energy(c, v)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return capacitor_energy(args[0], args[1])
    raise ValueError("Expected {capacitance, voltage}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register electrical engineering tools in the registry."""

    registry.register(ToolSpec(
        name="electrical_ohms_law",
        function=_parse_ohms_law,
        description="Apply Ohm's Law (V = IR). Provide any 2 of voltage/current/resistance to calculate the third.",
        parameters={
            "type": "object",
            "properties": {
                "voltage": {"type": "number", "description": "Voltage in Volts (V)"},
                "current": {"type": "number", "description": "Current in Amperes (A)"},
                "resistance": {"type": "number", "description": "Resistance in Ohms (Ω)"},
            },
        },
        returns="Dict with voltage, current, and resistance",
        examples=[
            {"input": {"voltage": 12, "resistance": 4}, "output": {"voltage": 12, "current": 3.0, "resistance": 4}},
            {"input": {"current": 2, "resistance": 6}, "output": {"voltage": 12.0, "current": 2, "resistance": 6}},
        ],
        domain="electrical",
        tags=["ohm", "voltage", "current", "resistance", "circuit"],
    ))

    registry.register(ToolSpec(
        name="electrical_power",
        function=_parse_power,
        description="Calculate DC power using P = IV = I²R = V²/R. Provide any 2 values.",
        parameters={
            "type": "object",
            "properties": {
                "voltage": {"type": "number", "description": "Voltage in Volts (V)"},
                "current": {"type": "number", "description": "Current in Amperes (A)"},
                "resistance": {"type": "number", "description": "Resistance in Ohms (Ω)"},
                "power": {"type": "number", "description": "Power in Watts (W)"},
            },
        },
        returns="Dict with power and related values",
        examples=[
            {"input": {"voltage": 12, "current": 2}, "output": {"power": 24.0, "voltage": 12, "current": 2}},
        ],
        domain="electrical",
        tags=["power", "watts", "energy"],
    ))

    registry.register(ToolSpec(
        name="electrical_impedance",
        function=_parse_impedance,
        description="Calculate impedance magnitude: |Z| = sqrt(R² + X²)",
        parameters={
            "type": "object",
            "properties": {
                "resistance": {"type": "number", "description": "Resistance R in Ohms"},
                "reactance": {"type": "number", "description": "Reactance X in Ohms"},
            },
            "required": ["resistance", "reactance"]
        },
        returns="Impedance magnitude in Ohms",
        examples=[
            {"input": {"resistance": 3, "reactance": 4}, "output": 5.0},
        ],
        domain="electrical",
        tags=["impedance", "ac", "reactance"],
    ))

    registry.register(ToolSpec(
        name="electrical_resonance_freq",
        function=_parse_resonance,
        description="Calculate LC circuit resonant frequency: f = 1/(2π√(LC))",
        parameters={
            "type": "object",
            "properties": {
                "inductance": {"type": "number", "description": "Inductance in Henries (H)"},
                "capacitance": {"type": "number", "description": "Capacitance in Farads (F)"},
            },
            "required": ["inductance", "capacitance"]
        },
        returns="Resonant frequency in Hertz (Hz)",
        examples=[
            {"input": {"inductance": 0.001, "capacitance": 0.000001}, "output": 5032.921},
        ],
        domain="electrical",
        tags=["resonance", "frequency", "lc", "tank"],
    ))

    registry.register(ToolSpec(
        name="electrical_voltage_divider",
        function=_parse_voltage_divider,
        description="Calculate voltage divider output: Vout = Vin * R2/(R1+R2)",
        parameters={
            "type": "object",
            "properties": {
                "vin": {"type": "number", "description": "Input voltage in Volts"},
                "r1": {"type": "number", "description": "First resistor in Ohms"},
                "r2": {"type": "number", "description": "Second resistor in Ohms"},
            },
            "required": ["vin", "r1", "r2"]
        },
        returns="Output voltage in Volts",
        examples=[
            {"input": {"vin": 12, "r1": 1000, "r2": 1000}, "output": 6.0},
        ],
        domain="electrical",
        tags=["voltage", "divider", "resistor"],
    ))

    registry.register(ToolSpec(
        name="electrical_series_resistance",
        function=lambda args: series_resistance(_parse_resistances(args)),
        description="Calculate total resistance of resistors in series: Rt = R1 + R2 + ...",
        parameters={
            "type": "object",
            "properties": {
                "resistances": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of resistance values in Ohms"
                },
            },
            "required": ["resistances"]
        },
        returns="Total resistance in Ohms",
        examples=[
            {"input": {"resistances": [100, 200, 300]}, "output": 600},
        ],
        domain="electrical",
        tags=["series", "resistance", "total"],
    ))

    registry.register(ToolSpec(
        name="electrical_parallel_resistance",
        function=lambda args: parallel_resistance(_parse_resistances(args)),
        description="Calculate total resistance of resistors in parallel: 1/Rt = 1/R1 + 1/R2 + ...",
        parameters={
            "type": "object",
            "properties": {
                "resistances": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of resistance values in Ohms"
                },
            },
            "required": ["resistances"]
        },
        returns="Total resistance in Ohms",
        examples=[
            {"input": {"resistances": [100, 100]}, "output": 50.0},
        ],
        domain="electrical",
        tags=["parallel", "resistance", "total"],
    ))

    registry.register(ToolSpec(
        name="electrical_capacitor_energy",
        function=_parse_capacitor_energy,
        description="Calculate energy stored in a capacitor: E = 0.5 * C * V²",
        parameters={
            "type": "object",
            "properties": {
                "capacitance": {"type": "number", "description": "Capacitance in Farads (F)"},
                "voltage": {"type": "number", "description": "Voltage in Volts (V)"},
            },
            "required": ["capacitance", "voltage"]
        },
        returns="Energy in Joules (J)",
        examples=[
            {"input": {"capacitance": 0.001, "voltage": 12}, "output": 0.072},
        ],
        domain="electrical",
        tags=["capacitor", "energy", "storage"],
    ))
