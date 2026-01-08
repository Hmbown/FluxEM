"""Finance domain tools and encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def compound_interest(
    principal: float,
    annual_rate: float,
    years: float,
    times_per_year: int = 1,
) -> float:
    """Compound interest growth."""
    if times_per_year <= 0:
        raise ValueError("times_per_year must be > 0")
    rate = annual_rate / times_per_year
    periods = years * times_per_year
    return principal * ((1 + rate) ** periods)


def npv(rate: float, cashflows: Sequence[float]) -> float:
    """Net present value for a series of cashflows (t=0 at index 0)."""
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += cf / ((1 + rate) ** t)
    return total


def payment(
    principal: float,
    annual_rate: float,
    years: float,
    periods_per_year: int = 12,
) -> float:
    """Periodic payment for an amortized loan."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    n = int(round(years * periods_per_year))
    if n <= 0:
        raise ValueError("years must be > 0")
    rate = annual_rate / periods_per_year
    if rate == 0:
        return principal / n
    return principal * rate / (1 - (1 + rate) ** (-n))


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


MAX_CASHFLOWS = 16
RATE_OFFSET = 0
COUNT_OFFSET = 2
NPV_OFFSET = 3
TOTAL_OFFSET = 5
MEAN_OFFSET = 7
VALUES_OFFSET = 9


@dataclass(frozen=True)
class CashflowSeries:
    """Cashflow series representation for encoding."""

    rate: float
    cashflows: List[float]
    npv: float
    total: float
    mean: float
    count: int


class FinanceEncoder:
    """Encoder for cashflow series and discount rate."""

    domain_tag = DOMAIN_TAGS["finance_cashflow"]
    domain_name = "finance_cashflow"

    def encode(self, rate: float, cashflows: Sequence[float]) -> Any:
        """Encode a cashflow series into an embedding."""
        values = list(cashflows)
        if not values:
            raise ValueError("cashflows must be non-empty")

        count = len(values)
        total_val = sum(values)
        mean_val = total_val / count
        npv_val = npv(rate, values)

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        rate_sign, rate_log = log_encode_value(rate)
        emb = backend.at_add(emb, 16 + RATE_OFFSET, rate_sign)
        emb = backend.at_add(emb, 16 + RATE_OFFSET + 1, rate_log)

        count_norm = min(count, MAX_CASHFLOWS) / MAX_CASHFLOWS
        emb = backend.at_add(emb, 16 + COUNT_OFFSET, count_norm)

        npv_sign, npv_log = log_encode_value(npv_val)
        emb = backend.at_add(emb, 16 + NPV_OFFSET, npv_sign)
        emb = backend.at_add(emb, 16 + NPV_OFFSET + 1, npv_log)

        total_sign, total_log = log_encode_value(total_val)
        emb = backend.at_add(emb, 16 + TOTAL_OFFSET, total_sign)
        emb = backend.at_add(emb, 16 + TOTAL_OFFSET + 1, total_log)

        mean_sign, mean_log = log_encode_value(mean_val)
        emb = backend.at_add(emb, 16 + MEAN_OFFSET, mean_sign)
        emb = backend.at_add(emb, 16 + MEAN_OFFSET + 1, mean_log)

        for i, cf in enumerate(values[:MAX_CASHFLOWS]):
            sign, log_mag = log_encode_value(float(cf))
            offset = 16 + VALUES_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        return emb

    def decode(self, emb: Any) -> CashflowSeries:
        """Decode an embedding back to a cashflow summary."""
        rate_val = log_decode_value(_scalar(emb[16 + RATE_OFFSET]), _scalar(emb[16 + RATE_OFFSET + 1]))
        count = int(round(_scalar(emb[16 + COUNT_OFFSET]) * MAX_CASHFLOWS))
        npv_val = log_decode_value(_scalar(emb[16 + NPV_OFFSET]), _scalar(emb[16 + NPV_OFFSET + 1]))
        total_val = log_decode_value(_scalar(emb[16 + TOTAL_OFFSET]), _scalar(emb[16 + TOTAL_OFFSET + 1]))
        mean_val = log_decode_value(_scalar(emb[16 + MEAN_OFFSET]), _scalar(emb[16 + MEAN_OFFSET + 1]))

        cashflows: List[float] = []
        for i in range(max(count, 0)):
            offset = 16 + VALUES_OFFSET + 2 * i
            cf = log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1]))
            cashflows.append(cf)

        return CashflowSeries(
            rate=rate_val,
            cashflows=cashflows,
            npv=npv_val,
            total=total_val,
            mean=mean_val,
            count=count,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid finance cashflow."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "CashflowSeries",
    "FinanceEncoder",
    "compound_interest",
    "npv",
    "payment",
]
