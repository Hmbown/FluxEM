"""
DateTime encoder for dates, times, and durations.

This encoder provides:
- Date arithmetic (date + duration = date)
- Day-of-week calculations
- Leap year handling
- Time zone aware operations

Key operations:
- add_days, add_months, add_years
- difference between dates
- day_of_week, is_weekend, is_leap_year
"""

import math
from datetime import datetime, date, time, timedelta
from typing import Any, Tuple, Union, Optional
from ...backend import get_backend

from ...core.base import (
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)

# Get backend at module level
backend = get_backend()


# DateTime domain tag
DATETIME_TAG = backend.array([0, 0, 0, 1, 1, 0, 0, 0])
DURATION_TAG = backend.array([0, 0, 0, 1, 1, 0, 0, 1])

# Embedding layout for dates (dims 8-71):
# dims 0-1:   Year (sign, log|year|)
# dims 2:     Month (1-12, normalized)
# dims 3:     Day (1-31, normalized)
# dims 4:     Day of week (0-6, normalized) - 0=Monday
# dims 5:     Day of year (1-366, normalized)
# dims 6:     Is leap year flag
# dims 7:     Is weekend flag
# dims 8-9:   Hour (0-23, normalized) and minute (0-59, normalized)
# dims 10-11: Second and microsecond
# dims 12:    Has time component flag
# dims 13-15: Reserved
# dims 16-19: Unix timestamp (sign, log|ts|, fractional part)

YEAR_OFFSET = 0
MONTH_OFFSET = 2
DAY_OFFSET = 3
DOW_OFFSET = 4
DOY_OFFSET = 5
IS_LEAP_FLAG = 6
IS_WEEKEND_FLAG = 7
HOUR_OFFSET = 8
MINUTE_OFFSET = 9
SECOND_OFFSET = 10
MICROSECOND_OFFSET = 11
HAS_TIME_FLAG = 12
UNIX_TS_OFFSET = 16

# Duration layout (dims 8-71):
# dims 0-1:   Total seconds (sign, log|seconds|)
# dims 2:     Days component (normalized)
# dims 3:     Hours component (normalized)
# dims 4:     Minutes component (normalized)
# dims 5:     Seconds component (normalized)
# dims 6:     Is negative flag

DUR_SECONDS_OFFSET = 0
DUR_DAYS_OFFSET = 2
DUR_HOURS_OFFSET = 3
DUR_MINUTES_OFFSET = 4
DUR_SECS_OFFSET = 5
DUR_NEG_FLAG = 6


def _is_leap_year(year: int) -> bool:
    """Check if year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _days_in_month(year: int, month: int) -> int:
    """Get number of days in a month."""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if _is_leap_year(year) else 28
    raise ValueError(f"Invalid month: {month}")


def _day_of_year(year: int, month: int, day: int) -> int:
    """Calculate day of year (1-366)."""
    days = day
    for m in range(1, month):
        days += _days_in_month(year, m)
    return days


class DateTimeEncoder:
    """
    Encoder for dates and times.

    Provides date arithmetic with calendar handling.
    """
    
    domain_tag = DATETIME_TAG
    domain_name = "datetime"
    
    def encode(
        self, 
        value: Union[datetime, date, str, Tuple[int, int, int]]
    ) -> Any:
        """
        Encode a date or datetime.
        
        Args:
            value: datetime, date, ISO string, or (year, month, day) tuple
            
        Returns:
            128-dim embedding
        """
        # Parse input
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        elif isinstance(value, tuple):
            value = date(value[0], value[1], value[2])
        
        if isinstance(value, datetime):
            dt = value
            has_time = True
        else:
            dt = datetime(value.year, value.month, value.day)
            has_time = False
        
        emb = create_embedding()
        
        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)
        
        # Year
        year_sign, year_log = log_encode_value(float(abs(dt.year)))
        emb = backend.at_add(emb, 8 + YEAR_OFFSET, 1.0 if dt.year >= 0 else -1.0)
        emb = backend.at_add(emb, 8 + YEAR_OFFSET + 1, year_log)
        
        # Month (1-12 -> 0-1)
        emb = backend.at_add(emb, 8 + MONTH_OFFSET, (dt.month - 1) / 11.0)
        
        # Day (1-31 -> 0-1)
        emb = backend.at_add(emb, 8 + DAY_OFFSET, (dt.day - 1) / 30.0)
        
        # Day of week (Monday=0)
        dow = dt.weekday()
        emb = backend.at_add(emb, 8 + DOW_OFFSET, dow / 6.0)
        
        # Day of year
        doy = _day_of_year(dt.year, dt.month, dt.day)
        emb = backend.at_add(emb, 8 + DOY_OFFSET, (doy - 1) / 365.0)
        
        # Flags
        is_leap = _is_leap_year(dt.year)
        is_weekend = dow >= 5
        emb = backend.at_add(emb, 8 + IS_LEAP_FLAG, 1.0 if is_leap else 0.0)
        emb = backend.at_add(emb, 8 + IS_WEEKEND_FLAG, 1.0 if is_weekend else 0.0)
        emb = backend.at_add(emb, 8 + HAS_TIME_FLAG, 1.0 if has_time else 0.0)
        
        # Time components
        if has_time:
            emb = backend.at_add(emb, 8 + HOUR_OFFSET, dt.hour / 23.0)
            emb = backend.at_add(emb, 8 + MINUTE_OFFSET, dt.minute / 59.0)
            emb = backend.at_add(emb, 8 + SECOND_OFFSET, dt.second / 59.0)
            emb = backend.at_add(emb, 8 + MICROSECOND_OFFSET, dt.microsecond / 999999.0)
        
        # Unix timestamp
        try:
            ts = dt.timestamp()
            ts_sign, ts_log = log_encode_value(abs(ts))
            emb = backend.at_add(emb, 8 + UNIX_TS_OFFSET, 1.0 if ts >= 0 else -1.0)
            emb = backend.at_add(emb, 8 + UNIX_TS_OFFSET + 1, ts_log)
        except:
            pass  # Pre-1970 dates may not have timestamp
        
        return emb
    
    def decode(self, emb: Any) -> datetime:
        """
        Decode embedding to datetime.
        
        Returns:
            datetime object
        """
        # Year
        year_sign = emb[8 + YEAR_OFFSET].item()
        year_log = emb[8 + YEAR_OFFSET + 1].item()
        year = int(round(log_decode_value(1.0, year_log)))
        if year_sign < 0:
            year = -year
        year = max(1, min(9999, year))
        
        # Month
        month = int(round(emb[8 + MONTH_OFFSET].item() * 11.0)) + 1
        month = max(1, min(12, month))
        
        # Day
        day = int(round(emb[8 + DAY_OFFSET].item() * 30.0)) + 1
        max_day = _days_in_month(year, month)
        day = max(1, min(max_day, day))
        
        # Time
        has_time = emb[8 + HAS_TIME_FLAG].item() > 0.5
        if has_time:
            hour = int(round(emb[8 + HOUR_OFFSET].item() * 23.0))
            minute = int(round(emb[8 + MINUTE_OFFSET].item() * 59.0))
            second = int(round(emb[8 + SECOND_OFFSET].item() * 59.0))
            hour = max(0, min(23, hour))
            minute = max(0, min(59, minute))
            second = max(0, min(59, second))
            return datetime(year, month, day, hour, minute, second)
        
        return datetime(year, month, day)
    
    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid datetime."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()
    
    # =========================================================================
    # Date Properties
    # =========================================================================
    
    def get_year(self, emb: Any) -> int:
        """Get year from embedding."""
        year_sign = emb[8 + YEAR_OFFSET].item()
        year_log = emb[8 + YEAR_OFFSET + 1].item()
        year = int(round(log_decode_value(1.0, year_log)))
        return year if year_sign >= 0 else -year
    
    def get_month(self, emb: Any) -> int:
        """Get month from embedding (1-12)."""
        return int(round(emb[8 + MONTH_OFFSET].item() * 11.0)) + 1
    
    def get_day(self, emb: Any) -> int:
        """Get day from embedding (1-31)."""
        return int(round(emb[8 + DAY_OFFSET].item() * 30.0)) + 1
    
    def get_day_of_week(self, emb: Any) -> int:
        """Get day of week (0=Monday, 6=Sunday)."""
        return int(round(emb[8 + DOW_OFFSET].item() * 6.0))
    
    def get_day_of_year(self, emb: Any) -> int:
        """Get day of year (1-366)."""
        return int(round(emb[8 + DOY_OFFSET].item() * 365.0)) + 1
    
    def is_leap_year(self, emb: Any) -> bool:
        """Check if year is a leap year."""
        return emb[8 + IS_LEAP_FLAG].item() > 0.5
    
    def is_weekend(self, emb: Any) -> bool:
        """Check if date is on a weekend."""
        return emb[8 + IS_WEEKEND_FLAG].item() > 0.5
    
    # =========================================================================
    # Date Arithmetic
    # =========================================================================
    
    def add_days(self, emb: Any, days: int) -> Any:
        """Add days to a date."""
        dt = self.decode(emb)
        result = dt + timedelta(days=days)
        return self.encode(result)
    
    def add_months(self, emb: Any, months: int) -> Any:
        """Add months to a date (handles month-end correctly)."""
        dt = self.decode(emb)
        
        # Calculate new year and month
        total_months = dt.year * 12 + (dt.month - 1) + months
        new_year = total_months // 12
        new_month = (total_months % 12) + 1
        
        # Handle day overflow
        max_day = _days_in_month(new_year, new_month)
        new_day = min(dt.day, max_day)
        
        result = dt.replace(year=new_year, month=new_month, day=new_day)
        return self.encode(result)
    
    def add_years(self, emb: Any, years: int) -> Any:
        """Add years to a date (handles Feb 29 correctly)."""
        dt = self.decode(emb)
        
        new_year = dt.year + years
        
        # Handle Feb 29 in non-leap year
        if dt.month == 2 and dt.day == 29 and not _is_leap_year(new_year):
            new_day = 28
        else:
            new_day = dt.day
        
        result = dt.replace(year=new_year, day=new_day)
        return self.encode(result)
    
    def difference_days(self, emb1: Any, emb2: Any) -> int:
        """Calculate difference in days between two dates."""
        dt1 = self.decode(emb1)
        dt2 = self.decode(emb2)
        
        delta = dt1 - dt2
        return delta.days


class DurationEncoder:
    """
    Encoder for time durations.
    
    Durations can be added/subtracted and applied to dates.
    """
    
    domain_tag = DURATION_TAG
    domain_name = "duration"
    
    def encode(self, value: Union[timedelta, int, float, Tuple]) -> Any:
        """
        Encode a duration.
        
        Args:
            value: timedelta, seconds (int/float), or (days, hours, minutes, seconds) tuple
            
        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if isinstance(value, timedelta):
            total_seconds = value.total_seconds()
        elif isinstance(value, tuple):
            days, hours, minutes, seconds = value
            total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        else:
            total_seconds = float(value)
        
        emb = create_embedding()
        
        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)
        
        # Total seconds (log magnitude)
        is_negative = total_seconds < 0
        abs_seconds = abs(total_seconds)
        
        sign, log_mag = log_encode_value(abs_seconds)
        emb = backend.at_add(emb, 8 + DUR_SECONDS_OFFSET, sign)
        emb = backend.at_add(emb, 8 + DUR_SECONDS_OFFSET + 1, log_mag)
        
        # Components (for human-readable representation)
        days = int(abs_seconds // 86400)
        remaining = abs_seconds % 86400
        hours = int(remaining // 3600)
        remaining = remaining % 3600
        minutes = int(remaining // 60)
        seconds = remaining % 60
        
        # Normalize to [0, 1] ranges
        emb = backend.at_add(emb, 8 + DUR_DAYS_OFFSET, min(days, 365) / 365.0)
        emb = backend.at_add(emb, 8 + DUR_HOURS_OFFSET, hours / 23.0)
        emb = backend.at_add(emb, 8 + DUR_MINUTES_OFFSET, minutes / 59.0)
        emb = backend.at_add(emb, 8 + DUR_SECS_OFFSET, seconds / 59.0)
        
        # Negative flag
        emb = backend.at_add(emb, 8 + DUR_NEG_FLAG, 1.0 if is_negative else 0.0)
        
        return emb
    
    def decode(self, emb: Any) -> timedelta:
        """Decode embedding to timedelta."""
        sign = emb[8 + DUR_SECONDS_OFFSET].item()
        log_mag = emb[8 + DUR_SECONDS_OFFSET + 1].item()
        
        total_seconds = log_decode_value(sign, log_mag)
        
        is_negative = emb[8 + DUR_NEG_FLAG].item() > 0.5
        if is_negative:
            total_seconds = -abs(total_seconds)
        
        return timedelta(seconds=total_seconds)
    
    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid duration."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()
    
    # =========================================================================
    # Duration Operations
    # =========================================================================
    
    def add(self, emb1: Any, emb2: Any) -> Any:
        """Add two durations."""
        d1 = self.decode(emb1)
        d2 = self.decode(emb2)
        return self.encode(d1 + d2)
    
    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """Subtract two durations."""
        d1 = self.decode(emb1)
        d2 = self.decode(emb2)
        return self.encode(d1 - d2)
    
    def multiply(self, emb: Any, factor: float) -> Any:
        """Multiply duration by a factor."""
        d = self.decode(emb)
        return self.encode(d * factor)
    
    def get_total_seconds(self, emb: Any) -> float:
        """Get total seconds."""
        return self.decode(emb).total_seconds()
    
    def get_days(self, emb: Any) -> int:
        """Get days component."""
        return self.decode(emb).days


# Convenience functions
def encode_date(year: int, month: int, day: int) -> Any:
    """Encode a date from components."""
    return DateTimeEncoder().encode((year, month, day))

def encode_datetime(dt: Union[datetime, str]) -> Any:
    """Encode a datetime."""
    return DateTimeEncoder().encode(dt)

def encode_duration(days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> Any:
    """Encode a duration from components."""
    return DurationEncoder().encode((days, hours, minutes, seconds))

def days_between(date1: Any, date2: Any) -> int:
    """Calculate days between two date embeddings."""
    return DateTimeEncoder().difference_days(date1, date2)
