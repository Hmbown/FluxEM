"""Temporal domain - dates, time operations.

This module provides deterministic date and time computations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def parse_date(date_str: str) -> datetime:
    """Parse an ISO date string (YYYY-MM-DD) to datetime."""
    date_str = date_str.strip()
    # Try various formats
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def add_days(date_str: str, days: int) -> str:
    """Add days to an ISO date string.

    Args:
        date_str: Date in ISO format (YYYY-MM-DD)
        days: Number of days to add (can be negative)

    Returns:
        New date as ISO string
    """
    dt = parse_date(date_str)
    new_dt = dt + timedelta(days=days)
    return new_dt.strftime("%Y-%m-%d")


def date_diff_days(date1: str, date2: str) -> int:
    """Compute difference in days between two dates (date2 - date1).

    Args:
        date1: Start date (YYYY-MM-DD)
        date2: End date (YYYY-MM-DD)

    Returns:
        Difference in days (positive if date2 > date1)
    """
    dt1 = parse_date(date1)
    dt2 = parse_date(date2)
    return (dt2 - dt1).days


def day_of_week(date_str: str) -> str:
    """Get day of week name for a date.

    Args:
        date_str: Date in ISO format

    Returns:
        Day name (e.g., 'Monday')
    """
    dt = parse_date(date_str)
    return DAY_NAMES[dt.weekday()]


def day_of_year(date_str: str) -> int:
    """Get day of year (1-366) for a date.

    Args:
        date_str: Date in ISO format

    Returns:
        Day of year (1-based)
    """
    dt = parse_date(date_str)
    return dt.timetuple().tm_yday


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    """Get number of days in a month."""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    else:
        raise ValueError(f"Invalid month: {month}")


def week_number(date_str: str) -> int:
    """Get ISO week number for a date.

    Args:
        date_str: Date in ISO format

    Returns:
        Week number (1-53)
    """
    dt = parse_date(date_str)
    return dt.isocalendar()[1]


def quarter(date_str: str) -> int:
    """Get calendar quarter (1-4) for a date."""
    dt = parse_date(date_str)
    return (dt.month - 1) // 3 + 1


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_add_days_args(args) -> Tuple[str, int]:
    if isinstance(args, dict):
        date_str = args.get("date", args.get("date_str"))
        days = args.get("days", args.get("n"))
        return str(date_str), int(days)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return str(args[0]), int(args[1])
    raise ValueError(f"Cannot parse add_days args: {args}")


def _parse_two_dates(args) -> Tuple[str, str]:
    if isinstance(args, dict):
        date1 = args.get("date1", args.get("start", args.get("from")))
        date2 = args.get("date2", args.get("end", args.get("to")))
        return str(date1), str(date2)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return str(args[0]), str(args[1])
    raise ValueError(f"Cannot parse two dates: {args}")


def _parse_single_date(args) -> str:
    if isinstance(args, dict):
        date_str = args.get("date", args.get("date_str", list(args.values())[0]))
        return str(date_str)
    if isinstance(args, (list, tuple)):
        return str(args[0])
    return str(args)


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register temporal tools in the registry."""

    registry.register(ToolSpec(
        name="temporal_add_days",
        function=lambda args: add_days(*_parse_add_days_args(args)),
        description="Adds days to an ISO date string.",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "days": {"type": "integer", "description": "Days to add (can be negative)"},
            },
            "required": ["date", "days"]
        },
        returns="New date as ISO string",
        examples=[
            {"input": {"date": "2024-01-01", "days": 7}, "output": "2024-01-08"},
            {"input": {"date": "2024-01-01", "days": -1}, "output": "2023-12-31"},
        ],
        domain="temporal",
        tags=["date", "add", "days"],
    ))

    registry.register(ToolSpec(
        name="temporal_diff_days",
        function=lambda args: date_diff_days(*_parse_two_dates(args)),
        description="Computes difference in days between two dates (end - start).",
        parameters={
            "type": "object",
            "properties": {
                "date1": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "date2": {"type": "string", "description": "End date (YYYY-MM-DD)"},
            },
            "required": ["date1", "date2"]
        },
        returns="Difference in days as integer",
        examples=[
            {"input": {"date1": "2024-01-01", "date2": "2024-01-10"}, "output": 9},
        ],
        domain="temporal",
        tags=["date", "difference", "days"],
    ))

    registry.register(ToolSpec(
        name="temporal_day_of_week",
        function=lambda args: day_of_week(_parse_single_date(args)),
        description="Returns day of week name for an ISO date.",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        },
        returns="Day name (e.g., 'Monday')",
        examples=[
            {"input": {"date": "2024-01-01"}, "output": "Monday"},
        ],
        domain="temporal",
        tags=["date", "weekday", "day"],
    ))
