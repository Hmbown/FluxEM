"""Fitness domain - health calculations, exercise metrics.

This module provides deterministic fitness/health computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def bmi(weight_kg: float, height_m: float) -> float:
    """Calculate Body Mass Index.

    BMI = weight / height²

    Args:
        weight_kg: Weight in kilograms
        height_m: Height in meters

    Returns:
        BMI value
    """
    if height_m <= 0:
        raise ValueError("Height must be positive")
    return weight_kg / (height_m ** 2)


def bmi_category(bmi_value: float) -> str:
    """Categorize BMI value.

    Args:
        bmi_value: BMI value

    Returns:
        Category string
    """
    if bmi_value < 18.5:
        return "underweight"
    elif bmi_value < 25:
        return "normal"
    elif bmi_value < 30:
        return "overweight"
    else:
        return "obese"


def bmr_mifflin(weight_kg: float, height_cm: float, age: int, is_male: bool) -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation.

    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        is_male: True for male, False for female

    Returns:
        BMR in calories per day
    """
    if is_male:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def tdee(bmr: float, activity_level: str) -> float:
    """Calculate Total Daily Energy Expenditure.

    Args:
        bmr: Basal Metabolic Rate
        activity_level: One of 'sedentary', 'light', 'moderate', 'active', 'very_active'

    Returns:
        TDEE in calories per day
    """
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    level = activity_level.lower().replace(" ", "_")
    if level not in multipliers:
        raise ValueError(f"Unknown activity level: {activity_level}")
    return bmr * multipliers[level]


def max_heart_rate(age: int) -> int:
    """Calculate maximum heart rate (age-based formula).

    Max HR = 220 - age

    Args:
        age: Age in years

    Returns:
        Maximum heart rate in BPM
    """
    return 220 - age


def heart_rate_zone(max_hr: int, zone: int) -> Tuple[int, int]:
    """Calculate heart rate zone boundaries.

    Zones:
        1: 50-60% (warm-up)
        2: 60-70% (fat burn)
        3: 70-80% (aerobic)
        4: 80-90% (anaerobic)
        5: 90-100% (max effort)

    Args:
        max_hr: Maximum heart rate
        zone: Zone number (1-5)

    Returns:
        (lower_bound, upper_bound) in BPM
    """
    zone_ranges = {
        1: (0.50, 0.60),
        2: (0.60, 0.70),
        3: (0.70, 0.80),
        4: (0.80, 0.90),
        5: (0.90, 1.00),
    }
    if zone not in zone_ranges:
        raise ValueError("Zone must be 1-5")
    low, high = zone_ranges[zone]
    return (int(max_hr * low), int(max_hr * high))


def pace_to_speed(pace_min_per_km: float) -> float:
    """Convert pace (min/km) to speed (km/h).

    Args:
        pace_min_per_km: Pace in minutes per kilometer

    Returns:
        Speed in km/h
    """
    if pace_min_per_km <= 0:
        raise ValueError("Pace must be positive")
    return 60 / pace_min_per_km


def speed_to_pace(speed_kmh: float) -> float:
    """Convert speed (km/h) to pace (min/km).

    Args:
        speed_kmh: Speed in km/h

    Returns:
        Pace in minutes per kilometer
    """
    if speed_kmh <= 0:
        raise ValueError("Speed must be positive")
    return 60 / speed_kmh


def one_rep_max(weight: float, reps: int) -> float:
    """Estimate one-rep max using Epley formula.

    1RM = weight * (1 + reps/30)

    Args:
        weight: Weight lifted
        reps: Number of repetitions

    Returns:
        Estimated one-rep max
    """
    if reps <= 0:
        raise ValueError("Reps must be positive")
    if reps == 1:
        return weight
    return weight * (1 + reps / 30)


def calorie_deficit_days(pounds_to_lose: float, daily_deficit: float) -> float:
    """Calculate days to lose weight at given calorie deficit.

    ~3500 calories = 1 pound of fat

    Args:
        pounds_to_lose: Target weight loss in pounds
        daily_deficit: Daily calorie deficit

    Returns:
        Days required
    """
    if daily_deficit <= 0:
        raise ValueError("Deficit must be positive")
    calories_needed = pounds_to_lose * 3500
    return calories_needed / daily_deficit


def vo2max_estimate(hr_rest: int, hr_max: int) -> float:
    """Estimate VO2max using heart rate method.

    VO2max ≈ 15.3 * (HRmax / HRrest)

    Args:
        hr_rest: Resting heart rate (BPM)
        hr_max: Maximum heart rate (BPM)

    Returns:
        Estimated VO2max (ml/kg/min)
    """
    if hr_rest <= 0:
        raise ValueError("Resting heart rate must be positive")
    return 15.3 * (hr_max / hr_rest)


def met_calories(met: float, weight_kg: float, duration_hours: float) -> float:
    """Calculate calories burned from MET value.

    Calories = MET * weight_kg * duration_hours

    Args:
        met: Metabolic Equivalent of Task
        weight_kg: Body weight in kg
        duration_hours: Activity duration in hours

    Returns:
        Calories burned
    """
    return met * weight_kg * duration_hours


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_bmi(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        weight = float(args.get("weight_kg", args.get("weight")))
        height = float(args.get("height_m", args.get("height")))
        return weight, height
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse bmi args: {args}")


def _parse_bmr(args) -> Tuple[float, float, int, bool]:
    if isinstance(args, dict):
        weight = float(args.get("weight_kg", args.get("weight")))
        height = float(args.get("height_cm", args.get("height")))
        age = int(args.get("age"))
        is_male = args.get("is_male", args.get("male", True))
        return weight, height, age, is_male
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return float(args[0]), float(args[1]), int(args[2]), bool(args[3])
    raise ValueError(f"Cannot parse bmr args: {args}")


def _parse_tdee(args) -> Tuple[float, str]:
    if isinstance(args, dict):
        bmr = float(args.get("bmr"))
        activity = str(args.get("activity_level", args.get("activity")))
        return bmr, activity
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), str(args[1])
    raise ValueError(f"Cannot parse tdee args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register fitness tools in the registry."""

    registry.register(ToolSpec(
        name="fitness_bmi",
        function=lambda args: bmi(*_parse_bmi(args)),
        description="Calculates Body Mass Index (BMI = weight/height²).",
        parameters={
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": "Weight in kilograms"},
                "height_m": {"type": "number", "description": "Height in meters"},
            },
            "required": ["weight_kg", "height_m"]
        },
        returns="BMI value",
        examples=[
            {"input": {"weight_kg": 70, "height_m": 1.75}, "output": 22.86},
        ],
        domain="fitness",
        tags=["bmi", "weight", "health"],
    ))

    registry.register(ToolSpec(
        name="fitness_bmr",
        function=lambda args: bmr_mifflin(*_parse_bmr(args)),
        description="Calculates Basal Metabolic Rate using Mifflin-St Jeor equation.",
        parameters={
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": "Weight in kilograms"},
                "height_cm": {"type": "number", "description": "Height in centimeters"},
                "age": {"type": "integer", "description": "Age in years"},
                "is_male": {"type": "boolean", "description": "True for male, False for female"},
            },
            "required": ["weight_kg", "height_cm", "age", "is_male"]
        },
        returns="BMR in calories per day",
        examples=[
            {"input": {"weight_kg": 70, "height_cm": 175, "age": 30, "is_male": True}, "output": 1648.75},
        ],
        domain="fitness",
        tags=["bmr", "metabolism", "calories"],
    ))

    registry.register(ToolSpec(
        name="fitness_tdee",
        function=lambda args: tdee(*_parse_tdee(args)),
        description="Calculates Total Daily Energy Expenditure based on BMR and activity level.",
        parameters={
            "type": "object",
            "properties": {
                "bmr": {"type": "number", "description": "Basal Metabolic Rate"},
                "activity_level": {"type": "string", "description": "sedentary, light, moderate, active, or very_active"},
            },
            "required": ["bmr", "activity_level"]
        },
        returns="TDEE in calories per day",
        examples=[
            {"input": {"bmr": 1650, "activity_level": "moderate"}, "output": 2557.5},
        ],
        domain="fitness",
        tags=["tdee", "calories", "energy"],
    ))

    registry.register(ToolSpec(
        name="fitness_heart_rate_zones",
        function=lambda args: heart_rate_zone(
            int(args.get("max_hr", args.get("max_heart_rate")) if isinstance(args, dict) else args[0]),
            int(args.get("zone") if isinstance(args, dict) else args[1])
        ),
        description="Calculates heart rate zone boundaries (zones 1-5).",
        parameters={
            "type": "object",
            "properties": {
                "max_hr": {"type": "integer", "description": "Maximum heart rate"},
                "zone": {"type": "integer", "description": "Zone number (1-5)"},
            },
            "required": ["max_hr", "zone"]
        },
        returns="(lower_bound, upper_bound) in BPM",
        examples=[
            {"input": {"max_hr": 190, "zone": 3}, "output": [133, 152]},
        ],
        domain="fitness",
        tags=["heart rate", "zone", "cardio"],
    ))

    registry.register(ToolSpec(
        name="fitness_pace_conversion",
        function=lambda args: pace_to_speed(
            float(args.get("pace", args.get("pace_min_per_km", args)) if isinstance(args, dict) else args)
        ),
        description="Converts pace (min/km) to speed (km/h).",
        parameters={
            "type": "object",
            "properties": {
                "pace": {"type": "number", "description": "Pace in minutes per kilometer"},
            },
            "required": ["pace"]
        },
        returns="Speed in km/h",
        examples=[
            {"input": {"pace": 5}, "output": 12.0},
            {"input": {"pace": 6}, "output": 10.0},
        ],
        domain="fitness",
        tags=["pace", "speed", "running"],
    ))

    registry.register(ToolSpec(
        name="fitness_one_rep_max",
        function=lambda args: one_rep_max(
            float(args.get("weight") if isinstance(args, dict) else args[0]),
            int(args.get("reps") if isinstance(args, dict) else args[1])
        ),
        description="Estimates one-rep max using Epley formula.",
        parameters={
            "type": "object",
            "properties": {
                "weight": {"type": "number", "description": "Weight lifted"},
                "reps": {"type": "integer", "description": "Number of repetitions"},
            },
            "required": ["weight", "reps"]
        },
        returns="Estimated one-rep max",
        examples=[
            {"input": {"weight": 100, "reps": 10}, "output": 133.33},
        ],
        domain="fitness",
        tags=["1rm", "strength", "lifting"],
    ))

    registry.register(ToolSpec(
        name="fitness_calorie_deficit",
        function=lambda args: calorie_deficit_days(
            float(args.get("pounds_to_lose", args.get("weight_loss")) if isinstance(args, dict) else args[0]),
            float(args.get("daily_deficit", args.get("deficit")) if isinstance(args, dict) else args[1])
        ),
        description="Calculates days to lose weight at given calorie deficit.",
        parameters={
            "type": "object",
            "properties": {
                "pounds_to_lose": {"type": "number", "description": "Target weight loss (pounds)"},
                "daily_deficit": {"type": "number", "description": "Daily calorie deficit"},
            },
            "required": ["pounds_to_lose", "daily_deficit"]
        },
        returns="Days required",
        examples=[
            {"input": {"pounds_to_lose": 10, "daily_deficit": 500}, "output": 70.0},
        ],
        domain="fitness",
        tags=["calorie", "deficit", "weight loss"],
    ))
