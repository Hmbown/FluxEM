"""Temporal domain tools and encoder for dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Union

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def _parse_date(value: Union[str, date, datetime]) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.date()
    raise ValueError("Unsupported date format")


def add_days(value: Union[str, date, datetime], days: int) -> str:
    """Add days to a date and return ISO date string."""
    base = _parse_date(value)
    result = base + timedelta(days=int(days))
    return result.isoformat()


def date_diff_days(start: Union[str, date, datetime], end: Union[str, date, datetime]) -> int:
    """Return day difference (end - start) in days."""
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    return (end_date - start_date).days


def day_of_week(value: Union[str, date, datetime]) -> str:
    """Return day of week name (Monday..Sunday)."""
    base = _parse_date(value)
    names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return names[base.weekday()]


def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _days_in_month(year: int, month: int) -> int:
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    if month == 2:
        return 29 if _is_leap_year(year) else 28
    raise ValueError(f"Invalid month: {month}")


def _day_of_year(year: int, month: int, day: int) -> int:
    total = day
    for m in range(1, month):
        total += _days_in_month(year, m)
    return total


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


YEAR_OFFSET = 0
MONTH_OFFSET = 2
DAY_OFFSET = 3
DOW_OFFSET = 4
DOY_OFFSET = 5
IS_LEAP_FLAG = 6
IS_WEEKEND_FLAG = 7


@dataclass(frozen=True)
class TemporalDate:
    """Date representation for temporal encoding."""

    year: int
    month: int
    day: int
    day_of_week: int
    day_of_year: int
    is_leap: bool
    is_weekend: bool


class TemporalEncoder:
    """Encoder for dates with calendar features."""

    domain_tag = DOMAIN_TAGS["temporal_date"]
    domain_name = "temporal_date"

    def encode(self, value: Union[str, date, datetime, TemporalDate]) -> Any:
        """Encode a date into an embedding."""
        if isinstance(value, TemporalDate):
            dt = date(value.year, value.month, value.day)
        else:
            dt = _parse_date(value)

        year = dt.year
        month = dt.month
        day = dt.day
        dow = dt.weekday()
        doy = _day_of_year(year, month, day)
        is_leap = _is_leap_year(year)
        is_weekend = dow >= 5

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        year_sign, year_log = log_encode_value(float(year))
        emb = backend.at_add(emb, 16 + YEAR_OFFSET, year_sign)
        emb = backend.at_add(emb, 16 + YEAR_OFFSET + 1, year_log)

        emb = backend.at_add(emb, 16 + MONTH_OFFSET, month / 12.0)
        emb = backend.at_add(emb, 16 + DAY_OFFSET, day / 31.0)
        emb = backend.at_add(emb, 16 + DOW_OFFSET, dow / 6.0)
        emb = backend.at_add(emb, 16 + DOY_OFFSET, doy / 366.0)
        emb = backend.at_add(emb, 16 + IS_LEAP_FLAG, 1.0 if is_leap else 0.0)
        emb = backend.at_add(emb, 16 + IS_WEEKEND_FLAG, 1.0 if is_weekend else 0.0)

        return emb

    def decode(self, emb: Any) -> TemporalDate:
        """Decode an embedding back to a date summary."""
        year = int(round(log_decode_value(_scalar(emb[16 + YEAR_OFFSET]), _scalar(emb[16 + YEAR_OFFSET + 1]))))
        month = max(1, min(12, int(round(_scalar(emb[16 + MONTH_OFFSET]) * 12))))
        day = max(1, min(31, int(round(_scalar(emb[16 + DAY_OFFSET]) * 31))))
        dow = max(0, min(6, int(round(_scalar(emb[16 + DOW_OFFSET]) * 6))))
        doy = max(1, min(366, int(round(_scalar(emb[16 + DOY_OFFSET]) * 366))))
        is_leap = _scalar(emb[16 + IS_LEAP_FLAG]) > 0.5
        is_weekend = _scalar(emb[16 + IS_WEEKEND_FLAG]) > 0.5

        return TemporalDate(
            year=year,
            month=month,
            day=day,
            day_of_week=dow,
            day_of_year=doy,
            is_leap=is_leap,
            is_weekend=is_weekend,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid temporal date."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "TemporalDate",
    "TemporalEncoder",
    "add_days",
    "date_diff_days",
    "day_of_week",
]
