"""Currency/Economics domain - exchange rates, inflation, interest.

This module provides deterministic financial/economic computations.
Note: Exchange rates are examples only - real rates change constantly.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def currency_exchange(amount: float, rate: float) -> float:
    """Convert currency using exchange rate.

    Args:
        amount: Amount in source currency
        rate: Exchange rate (target/source)

    Returns:
        Amount in target currency
    """
    return amount * rate


def inflation_adjust(amount: float, inflation_rate: float, years: float) -> float:
    """Adjust amount for inflation over time.

    Future value = Present value * (1 + r)^n

    Args:
        amount: Current amount
        inflation_rate: Annual inflation rate (decimal, e.g., 0.03 for 3%)
        years: Number of years

    Returns:
        Future equivalent amount (what you'd need to have same purchasing power)
    """
    return amount * ((1 + inflation_rate) ** years)


def purchasing_power(original_amount: float, original_year: int,
                     target_year: int, annual_inflation: float) -> float:
    """Calculate equivalent purchasing power between years.

    Args:
        original_amount: Amount in original year
        original_year: Year of original amount
        target_year: Year to convert to
        annual_inflation: Average annual inflation rate

    Returns:
        Equivalent amount in target year
    """
    years = target_year - original_year
    return original_amount * ((1 + annual_inflation) ** years)


def simple_interest(principal: float, rate: float, time: float) -> float:
    """Calculate simple interest.

    I = P * r * t

    Args:
        principal: Initial amount
        rate: Interest rate per period (decimal)
        time: Number of periods

    Returns:
        Interest earned
    """
    return principal * rate * time


def compound_interest_total(principal: float, rate: float,
                            compounds_per_year: int, years: float) -> float:
    """Calculate total amount with compound interest.

    A = P * (1 + r/n)^(n*t)

    Args:
        principal: Initial amount
        rate: Annual interest rate (decimal)
        compounds_per_year: Number of times interest compounds per year
        years: Number of years

    Returns:
        Total amount (principal + interest)
    """
    if compounds_per_year <= 0:
        raise ValueError("Compounds per year must be positive")
    return principal * ((1 + rate / compounds_per_year) ** (compounds_per_year * years))


def break_even_units(fixed_costs: float, price_per_unit: float,
                     variable_cost_per_unit: float) -> float:
    """Calculate break-even point in units.

    Break-even = Fixed Costs / (Price - Variable Cost)

    Args:
        fixed_costs: Total fixed costs
        price_per_unit: Selling price per unit
        variable_cost_per_unit: Variable cost per unit

    Returns:
        Number of units to break even
    """
    margin = price_per_unit - variable_cost_per_unit
    if margin <= 0:
        raise ValueError("Price must exceed variable cost")
    return fixed_costs / margin


def roi(gain: float, cost: float) -> float:
    """Calculate return on investment.

    ROI = (Gain - Cost) / Cost * 100

    Args:
        gain: Total return/value
        cost: Initial investment

    Returns:
        ROI as percentage
    """
    if cost == 0:
        raise ValueError("Cost cannot be zero")
    return ((gain - cost) / cost) * 100


def real_interest_rate(nominal_rate: float, inflation_rate: float) -> float:
    """Calculate real interest rate using Fisher equation.

    r_real ≈ r_nominal - inflation (simplified)
    Exact: (1 + r_nominal) / (1 + inflation) - 1

    Args:
        nominal_rate: Nominal interest rate (decimal)
        inflation_rate: Inflation rate (decimal)

    Returns:
        Real interest rate (decimal)
    """
    return (1 + nominal_rate) / (1 + inflation_rate) - 1


def present_value(future_value: float, rate: float, periods: int) -> float:
    """Calculate present value of a future amount.

    PV = FV / (1 + r)^n

    Args:
        future_value: Future amount
        rate: Discount rate per period (decimal)
        periods: Number of periods

    Returns:
        Present value
    """
    return future_value / ((1 + rate) ** periods)


def rule_of_72(rate: float) -> float:
    """Estimate years to double investment using Rule of 72.

    Years ≈ 72 / (rate * 100)

    Args:
        rate: Interest rate (decimal, e.g., 0.08 for 8%)

    Returns:
        Approximate years to double
    """
    if rate <= 0:
        raise ValueError("Rate must be positive")
    return 72 / (rate * 100)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_exchange(args) -> Tuple[float, float]:
    if isinstance(args, dict):
        amount = float(args.get("amount"))
        rate = float(args.get("rate", args.get("exchange_rate")))
        return amount, rate
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), float(args[1])
    raise ValueError(f"Cannot parse exchange args: {args}")


def _parse_inflation_adjust(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        amount = float(args.get("amount"))
        rate = float(args.get("inflation_rate", args.get("rate")))
        years = float(args.get("years"))
        return amount, rate, years
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse inflation_adjust args: {args}")


def _parse_compound(args) -> Tuple[float, float, int, float]:
    if isinstance(args, dict):
        p = float(args.get("principal", args.get("P")))
        r = float(args.get("rate", args.get("r")))
        n = int(args.get("compounds_per_year", args.get("n", 12)))
        t = float(args.get("years", args.get("t")))
        return p, r, n, t
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return float(args[0]), float(args[1]), int(args[2]), float(args[3])
    raise ValueError(f"Cannot parse compound args: {args}")


def _parse_break_even(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        fixed = float(args.get("fixed_costs", args.get("fixed")))
        price = float(args.get("price_per_unit", args.get("price")))
        variable = float(args.get("variable_cost_per_unit", args.get("variable")))
        return fixed, price, variable
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse break_even args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register currency/economics tools in the registry."""

    registry.register(ToolSpec(
        name="currency_exchange",
        function=lambda args: currency_exchange(*_parse_exchange(args)),
        description="Converts currency using an exchange rate.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount in source currency"},
                "rate": {"type": "number", "description": "Exchange rate (target/source)"},
            },
            "required": ["amount", "rate"]
        },
        returns="Amount in target currency",
        examples=[
            {"input": {"amount": 100, "rate": 0.85}, "output": 85.0},
        ],
        domain="currency",
        tags=["exchange", "convert", "forex"],
    ))

    registry.register(ToolSpec(
        name="currency_inflation_adjust",
        function=lambda args: inflation_adjust(*_parse_inflation_adjust(args)),
        description="Adjusts amount for inflation over time.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Current amount"},
                "inflation_rate": {"type": "number", "description": "Annual inflation rate (decimal)"},
                "years": {"type": "number", "description": "Number of years"},
            },
            "required": ["amount", "inflation_rate", "years"]
        },
        returns="Future equivalent amount",
        examples=[
            {"input": {"amount": 100, "inflation_rate": 0.03, "years": 10}, "output": 134.39},
        ],
        domain="currency",
        tags=["inflation", "purchasing power"],
    ))

    registry.register(ToolSpec(
        name="economics_simple_interest",
        function=lambda args: simple_interest(
            float(args.get("principal", args.get("P")) if isinstance(args, dict) else args[0]),
            float(args.get("rate", args.get("r")) if isinstance(args, dict) else args[1]),
            float(args.get("time", args.get("t")) if isinstance(args, dict) else args[2])
        ),
        description="Calculates simple interest (I = P*r*t).",
        parameters={
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Initial amount"},
                "rate": {"type": "number", "description": "Interest rate per period (decimal)"},
                "time": {"type": "number", "description": "Number of periods"},
            },
            "required": ["principal", "rate", "time"]
        },
        returns="Interest earned",
        examples=[
            {"input": {"principal": 1000, "rate": 0.05, "time": 2}, "output": 100.0},
        ],
        domain="currency",
        tags=["interest", "simple", "finance"],
    ))

    registry.register(ToolSpec(
        name="economics_compound_interest",
        function=lambda args: compound_interest_total(*_parse_compound(args)),
        description="Calculates total amount with compound interest.",
        parameters={
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Initial amount"},
                "rate": {"type": "number", "description": "Annual interest rate (decimal)"},
                "compounds_per_year": {"type": "integer", "description": "Compounding frequency"},
                "years": {"type": "number", "description": "Number of years"},
            },
            "required": ["principal", "rate", "compounds_per_year", "years"]
        },
        returns="Total amount (principal + interest)",
        examples=[
            {"input": {"principal": 1000, "rate": 0.05, "compounds_per_year": 12, "years": 10}, "output": 1647.01},
        ],
        domain="currency",
        tags=["interest", "compound", "finance"],
    ))

    registry.register(ToolSpec(
        name="economics_break_even",
        function=lambda args: break_even_units(*_parse_break_even(args)),
        description="Calculates break-even point in units.",
        parameters={
            "type": "object",
            "properties": {
                "fixed_costs": {"type": "number", "description": "Total fixed costs"},
                "price_per_unit": {"type": "number", "description": "Selling price per unit"},
                "variable_cost_per_unit": {"type": "number", "description": "Variable cost per unit"},
            },
            "required": ["fixed_costs", "price_per_unit", "variable_cost_per_unit"]
        },
        returns="Number of units to break even",
        examples=[
            {"input": {"fixed_costs": 10000, "price_per_unit": 50, "variable_cost_per_unit": 30}, "output": 500.0},
        ],
        domain="currency",
        tags=["break-even", "business", "profit"],
    ))

    registry.register(ToolSpec(
        name="economics_roi",
        function=lambda args: roi(
            float(args.get("gain", args.get("return")) if isinstance(args, dict) else args[0]),
            float(args.get("cost", args.get("investment")) if isinstance(args, dict) else args[1])
        ),
        description="Calculates return on investment (ROI) as percentage.",
        parameters={
            "type": "object",
            "properties": {
                "gain": {"type": "number", "description": "Total return/value"},
                "cost": {"type": "number", "description": "Initial investment"},
            },
            "required": ["gain", "cost"]
        },
        returns="ROI as percentage",
        examples=[
            {"input": {"gain": 1500, "cost": 1000}, "output": 50.0},
        ],
        domain="currency",
        tags=["roi", "investment", "return"],
    ))
