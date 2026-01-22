"""Finance domain - compound interest, NPV, loan payments.

This module provides deterministic financial computations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def compound_interest(principal: float, rate: float, time: float, n: int = 1) -> float:
    """Calculate compound interest: A = P(1 + r/n)^(nt).

    Args:
        principal: Initial investment
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Time in years
        n: Compounding frequency per year (1=annual, 12=monthly, 365=daily)

    Returns:
        Final amount
    """
    return principal * ((1 + rate / n) ** (n * time))


def simple_interest(principal: float, rate: float, time: float) -> float:
    """Calculate simple interest: A = P(1 + rt).

    Args:
        principal: Initial investment
        rate: Annual interest rate (as decimal)
        time: Time in years

    Returns:
        Final amount
    """
    return principal * (1 + rate * time)


def npv(rate: float, cashflows: List[float]) -> float:
    """Calculate Net Present Value.

    Args:
        rate: Discount rate per period (as decimal)
        cashflows: List of cash flows (first is usually negative = investment)

    Returns:
        Net present value
    """
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += cf / ((1 + rate) ** t)
    return total


def irr(cashflows: List[float], max_iter: int = 100, tol: float = 1e-9) -> float:
    """Calculate Internal Rate of Return using Newton-Raphson.

    Args:
        cashflows: List of cash flows
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        IRR as decimal
    """
    # Initial guess
    r = 0.1

    for _ in range(max_iter):
        # Calculate NPV and its derivative
        npv_val = sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows))
        npv_deriv = sum(-t * cf / ((1 + r) ** (t + 1)) for t, cf in enumerate(cashflows))

        if abs(npv_deriv) < 1e-15:
            break

        r_new = r - npv_val / npv_deriv

        if abs(r_new - r) < tol:
            return r_new

        r = r_new

    return r


def payment(principal: float, annual_rate: float, years: float, periods_per_year: int = 12) -> float:
    """Calculate periodic loan payment.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (as decimal)
        years: Loan term in years
        periods_per_year: Payment frequency (12 for monthly)

    Returns:
        Payment amount per period
    """
    r = annual_rate / periods_per_year
    n = int(years * periods_per_year)

    if r == 0:
        return principal / n

    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def present_value(future_value: float, rate: float, periods: int) -> float:
    """Calculate present value of a future amount.

    Args:
        future_value: Future amount
        rate: Interest rate per period
        periods: Number of periods

    Returns:
        Present value
    """
    return future_value / ((1 + rate) ** periods)


def future_value_annuity(payment: float, rate: float, periods: int) -> float:
    """Calculate future value of an ordinary annuity.

    Args:
        payment: Regular payment amount
        rate: Interest rate per period
        periods: Number of periods

    Returns:
        Future value
    """
    if rate == 0:
        return payment * periods
    return payment * (((1 + rate) ** periods - 1) / rate)


def break_even(fixed_costs: float, variable_cost_per_unit: float, price_per_unit: float) -> float:
    """Calculate break-even quantity.

    Args:
        fixed_costs: Total fixed costs
        variable_cost_per_unit: Variable cost per unit
        price_per_unit: Selling price per unit

    Returns:
        Break-even quantity
    """
    margin = price_per_unit - variable_cost_per_unit
    if margin <= 0:
        return float('inf')
    return fixed_costs / margin


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_compound_interest(args) -> Tuple[float, float, float, int]:
    if isinstance(args, dict):
        principal = float(args.get("principal", args.get("p")))
        rate = float(args.get("rate", args.get("r")))
        time = float(args.get("time", args.get("years", args.get("t"))))
        n = int(args.get("n", args.get("periods_per_year", args.get("compounds_per_year", 1))))
        return principal, rate, time, n
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        n = int(args[3]) if len(args) > 3 else 1
        return float(args[0]), float(args[1]), float(args[2]), n
    raise ValueError(f"Cannot parse compound interest args: {args}")


def _parse_npv(args) -> Tuple[float, List[float]]:
    if isinstance(args, dict):
        rate = float(args.get("rate", args.get("r")))
        cashflows = [float(x) for x in args.get("cashflows", args.get("cf", []))]
        return rate, cashflows
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), [float(x) for x in args[1]]
    raise ValueError(f"Cannot parse NPV args: {args}")


def _parse_payment(args) -> Tuple[float, float, float, int]:
    if isinstance(args, dict):
        principal = float(args.get("principal", args.get("loan")))
        rate = float(args.get("rate", args.get("annual_rate")))
        years = float(args.get("years", args.get("term")))
        periods = int(args.get("periods_per_year", args.get("frequency", 12)))
        return principal, rate, years, periods
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        periods = int(args[3]) if len(args) > 3 else 12
        return float(args[0]), float(args[1]), float(args[2]), periods
    raise ValueError(f"Cannot parse payment args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register finance tools in the registry."""

    registry.register(ToolSpec(
        name="finance_compound_interest",
        function=lambda args: compound_interest(*_parse_compound_interest(args)),
        description="Calculates compound interest: A = P(1 + r/n)^(nt).",
        parameters={
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Initial investment"},
                "rate": {"type": "number", "description": "Annual rate (decimal, e.g., 0.05 for 5%)"},
                "time": {"type": "number", "description": "Time in years"},
                "n": {"type": "integer", "description": "Compounds per year (default: 1)"},
            },
            "required": ["principal", "rate", "time"]
        },
        returns="Final amount",
        examples=[
            {"input": {"principal": 1000, "rate": 0.05, "time": 2}, "output": 1102.5},
            {"input": {"principal": 1000, "rate": 0.05, "time": 2, "n": 12}, "output": 1104.94},
        ],
        domain="finance",
        tags=["interest", "compound", "investment"],
    ))

    registry.register(ToolSpec(
        name="finance_npv",
        function=lambda args: npv(*_parse_npv(args)),
        description="Calculates Net Present Value from rate and cashflows.",
        parameters={
            "type": "object",
            "properties": {
                "rate": {"type": "number", "description": "Discount rate per period"},
                "cashflows": {"type": "array", "items": {"type": "number"}, "description": "Cash flows (first usually negative)"},
            },
            "required": ["rate", "cashflows"]
        },
        returns="NPV as float",
        examples=[
            {"input": {"rate": 0.1, "cashflows": [-100, 30, 40, 50, 60]}, "output": 28.36},
        ],
        domain="finance",
        tags=["npv", "present value", "investment"],
    ))

    registry.register(ToolSpec(
        name="finance_payment",
        function=lambda args: payment(*_parse_payment(args)),
        description="Calculates periodic loan payment.",
        parameters={
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan amount"},
                "rate": {"type": "number", "description": "Annual interest rate (decimal)"},
                "years": {"type": "number", "description": "Loan term in years"},
                "periods_per_year": {"type": "integer", "description": "Payments per year (default: 12)"},
            },
            "required": ["principal", "rate", "years"]
        },
        returns="Payment amount per period",
        examples=[
            {"input": {"principal": 10000, "rate": 0.05, "years": 1, "periods_per_year": 12}, "output": 856.07},
        ],
        domain="finance",
        tags=["payment", "loan", "mortgage"],
    ))
