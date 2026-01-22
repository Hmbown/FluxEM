"""Travel domain - time zones, fuel, distance calculations.

This module provides deterministic travel computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

# Common timezone offsets from UTC
TIMEZONE_OFFSETS = {
    "UTC": 0, "GMT": 0,
    "EST": -5, "EDT": -4,
    "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7,
    "CET": 1, "CEST": 2,
    "IST": 5.5,
    "JST": 9,
    "AEST": 10, "AEDT": 11,
}


# =============================================================================
# Core Functions
# =============================================================================

def timezone_offset(from_tz: str, to_tz: str) -> float:
    """Calculate timezone offset between two timezones.

    Args:
        from_tz: Source timezone abbreviation
        to_tz: Target timezone abbreviation

    Returns:
        Offset in hours (positive = ahead)
    """
    from_upper = from_tz.upper()
    to_upper = to_tz.upper()

    if from_upper not in TIMEZONE_OFFSETS:
        raise ValueError(f"Unknown timezone: {from_tz}")
    if to_upper not in TIMEZONE_OFFSETS:
        raise ValueError(f"Unknown timezone: {to_tz}")

    return TIMEZONE_OFFSETS[to_upper] - TIMEZONE_OFFSETS[from_upper]


def convert_time(hours: float, minutes: float, offset: float) -> Tuple[int, int]:
    """Convert time by offset.

    Args:
        hours: Hour (0-23)
        minutes: Minutes (0-59)
        offset: Offset in hours

    Returns:
        (new_hours, new_minutes) adjusted
    """
    total_minutes = hours * 60 + minutes + offset * 60
    total_minutes = total_minutes % (24 * 60)  # Wrap around
    new_hours = int(total_minutes // 60)
    new_minutes = int(total_minutes % 60)
    return (new_hours, new_minutes)


def fuel_consumption(distance: float, fuel_used: float) -> float:
    """Calculate fuel consumption rate.

    Args:
        distance: Distance traveled (km or miles)
        fuel_used: Fuel used (liters or gallons)

    Returns:
        Fuel consumption (L/100km or mpg depending on units)
    """
    if distance <= 0:
        raise ValueError("Distance must be positive")
    return fuel_used / distance * 100  # Returns L/100km if metric


def fuel_needed(distance: float, consumption_rate: float) -> float:
    """Calculate fuel needed for a trip.

    Args:
        distance: Distance to travel (km or miles)
        consumption_rate: Consumption rate (L/100km or gal/100mi)

    Returns:
        Fuel needed
    """
    return distance * consumption_rate / 100


def flight_duration(distance_km: float, speed_kmh: float = 900) -> float:
    """Estimate flight duration (direct).

    Args:
        distance_km: Great circle distance in km
        speed_kmh: Aircraft speed (default 900 km/h for jets)

    Returns:
        Flight duration in hours (excluding taxi/takeoff/landing)
    """
    if speed_kmh <= 0:
        raise ValueError("Speed must be positive")
    return distance_km / speed_kmh


def average_speed(distance: float, time_hours: float) -> float:
    """Calculate average speed.

    Args:
        distance: Total distance
        time_hours: Total time in hours

    Returns:
        Average speed
    """
    if time_hours <= 0:
        raise ValueError("Time must be positive")
    return distance / time_hours


def travel_time(distance: float, speed: float) -> float:
    """Calculate travel time.

    Args:
        distance: Distance to travel
        speed: Travel speed

    Returns:
        Time in same units as distance/speed
    """
    if speed <= 0:
        raise ValueError("Speed must be positive")
    return distance / speed


def km_to_miles(km: float) -> float:
    """Convert kilometers to miles."""
    return km * 0.621371


def miles_to_km(miles: float) -> float:
    """Convert miles to kilometers."""
    return miles * 1.60934


def liter_per_100km_to_mpg(l_per_100km: float) -> float:
    """Convert L/100km to miles per gallon (US).

    Args:
        l_per_100km: Liters per 100 kilometers

    Returns:
        Miles per gallon
    """
    if l_per_100km <= 0:
        raise ValueError("Consumption must be positive")
    return 235.215 / l_per_100km


def mpg_to_liter_per_100km(mpg: float) -> float:
    """Convert miles per gallon (US) to L/100km.

    Args:
        mpg: Miles per gallon

    Returns:
        Liters per 100 kilometers
    """
    if mpg <= 0:
        raise ValueError("MPG must be positive")
    return 235.215 / mpg


def jet_lag_recovery_days(timezone_diff: float) -> float:
    """Estimate jet lag recovery time.

    Rule of thumb: ~1 day per timezone crossed

    Args:
        timezone_diff: Number of timezones crossed (absolute)

    Returns:
        Estimated recovery days
    """
    return abs(timezone_diff) * 0.5 + 0.5  # Minimum half day


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_timezone_offset(args) -> Tuple[str, str]:
    if isinstance(args, dict):
        from_tz = str(args.get("from_tz", args.get("from")))
        to_tz = str(args.get("to_tz", args.get("to")))
        return from_tz, to_tz
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return str(args[0]), str(args[1])
    raise ValueError(f"Cannot parse timezone_offset args: {args}")


def _parse_fuel_consumption(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        dist = float(args.get("distance", args.get("d")))
        fuel = float(args.get("fuel_used", args.get("fuel")))
        return dist, fuel
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse fuel_consumption args: {args}")


def _parse_travel_time(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        dist = float(args.get("distance", args.get("d")))
        speed = float(args.get("speed", args.get("v")))
        return dist, speed
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse travel_time args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register travel tools in the registry."""

    registry.register(ToolSpec(
        name="travel_timezone_offset",
        function=lambda args: timezone_offset(*_parse_timezone_offset(args)),
        description="Calculates timezone offset between two timezones in hours.",
        parameters={
            "type": "object",
            "properties": {
                "from_tz": {"type": "string", "description": "Source timezone (e.g., EST, PST, UTC)"},
                "to_tz": {"type": "string", "description": "Target timezone"},
            },
            "required": ["from_tz", "to_tz"]
        },
        returns="Offset in hours (positive = ahead)",
        examples=[
            {"input": {"from_tz": "EST", "to_tz": "PST"}, "output": -3.0},
            {"input": {"from_tz": "UTC", "to_tz": "JST"}, "output": 9.0},
        ],
        domain="travel",
        tags=["timezone", "offset", "time"],
    ))

    registry.register(ToolSpec(
        name="travel_fuel_consumption",
        function=lambda args: fuel_consumption(*_parse_fuel_consumption(args)),
        description="Calculates fuel consumption rate (L/100km or similar).",
        parameters={
            "type": "object",
            "properties": {
                "distance": {"type": "number", "description": "Distance traveled"},
                "fuel_used": {"type": "number", "description": "Fuel used"},
            },
            "required": ["distance", "fuel_used"]
        },
        returns="Fuel consumption per 100 units distance",
        examples=[
            {"input": {"distance": 500, "fuel_used": 40}, "output": 8.0},
        ],
        domain="travel",
        tags=["fuel", "consumption", "efficiency"],
    ))

    registry.register(ToolSpec(
        name="travel_flight_duration",
        function=lambda args: flight_duration(
            float(args.get("distance_km", args.get("distance")) if isinstance(args, dict) else args[0]),
            float(args.get("speed_kmh", args.get("speed", 900)) if isinstance(args, dict) else (args[1] if len(args) > 1 else 900))
        ),
        description="Estimates flight duration for a given distance.",
        parameters={
            "type": "object",
            "properties": {
                "distance_km": {"type": "number", "description": "Distance in km"},
                "speed_kmh": {"type": "number", "description": "Aircraft speed (default 900 km/h)"},
            },
            "required": ["distance_km"]
        },
        returns="Flight duration in hours",
        examples=[
            {"input": {"distance_km": 5000}, "output": 5.56},
        ],
        domain="travel",
        tags=["flight", "duration", "distance"],
    ))

    registry.register(ToolSpec(
        name="travel_speed_average",
        function=lambda args: average_speed(
            float(args.get("distance") if isinstance(args, dict) else args[0]),
            float(args.get("time_hours", args.get("time")) if isinstance(args, dict) else args[1])
        ),
        description="Calculates average speed from distance and time.",
        parameters={
            "type": "object",
            "properties": {
                "distance": {"type": "number", "description": "Total distance"},
                "time_hours": {"type": "number", "description": "Total time in hours"},
            },
            "required": ["distance", "time_hours"]
        },
        returns="Average speed",
        examples=[
            {"input": {"distance": 300, "time_hours": 4}, "output": 75.0},
        ],
        domain="travel",
        tags=["speed", "average", "distance"],
    ))

    registry.register(ToolSpec(
        name="travel_unit_localize",
        function=lambda args: km_to_miles(
            float(args.get("km", args) if isinstance(args, dict) else args)
        ),
        description="Converts kilometers to miles.",
        parameters={
            "type": "object",
            "properties": {
                "km": {"type": "number", "description": "Distance in kilometers"},
            },
            "required": ["km"]
        },
        returns="Distance in miles",
        examples=[
            {"input": {"km": 100}, "output": 62.14},
        ],
        domain="travel",
        tags=["convert", "km", "miles"],
    ))

    registry.register(ToolSpec(
        name="travel_mpg_convert",
        function=lambda args: liter_per_100km_to_mpg(
            float(args.get("l_per_100km", args.get("consumption", args)) if isinstance(args, dict) else args)
        ),
        description="Converts L/100km to miles per gallon (US).",
        parameters={
            "type": "object",
            "properties": {
                "l_per_100km": {"type": "number", "description": "Liters per 100 km"},
            },
            "required": ["l_per_100km"]
        },
        returns="Miles per gallon",
        examples=[
            {"input": {"l_per_100km": 8}, "output": 29.4},
        ],
        domain="travel",
        tags=["mpg", "fuel", "convert"],
    ))
