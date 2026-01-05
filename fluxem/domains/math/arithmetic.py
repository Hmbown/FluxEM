"""
Arithmetic encoder.

Encodes scalar values into embeddings supporting closed-form arithmetic.
Addition/subtraction use a linear component; multiplication/division/powers use
a log-magnitude component. Results follow IEEE-754 floating-point semantics.
"""

import math
from typing import Any, Tuple, Union, Optional
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)

ARITH_TAG_VALUES = [1, 0, 0, 1, 0, 0, 0, 0]

# Embedding layout within domain-specific region (dims 8-71):
# dims 0-1:   Value (sign, log|value|) for magnitude
# dims 2-3:   Linear component (for addition)
# dims 4:     Is integer flag
# dims 5:     Is zero flag
# dims 6:     Is negative flag
# dims 7-9:   Reserved
# dims 10-41: Digit encoding (up to 16 digits, 2 dims each)

VALUE_OFFSET = 0
LINEAR_OFFSET = 2
IS_INT_FLAG = 4
IS_ZERO_FLAG = 5
IS_NEG_FLAG = 6
DIGITS_OFFSET = 10

# Scale for linear embedding (determines max representable number)
LINEAR_SCALE = 1e15


class ArithmeticEncoder:
    """Encode scalar values into arithmetic embeddings.

    Parameters
    ----------
    scale : float, optional
        Linear scale used for the additive component.

    Notes
    -----
    Addition/subtraction operate on the linear component. Multiplication and
    division are implemented in log-magnitude space and are subject to
    floating-point error in `log`/`exp`.
    """
    
    @property
    def domain_tag(self) -> Any:
        return get_backend().array(ARITH_TAG_VALUES)
    domain_name = "arithmetic"
    
    def __init__(self, scale: float = LINEAR_SCALE):
        """Initialize the encoder.

        Parameters
        ----------
        scale : float, optional
            Maximum representable value for linear operations.
        """
        self.scale = scale
    
    def encode(self, value: Union[int, float]) -> Any:
        """Encode a scalar value.

        Parameters
        ----------
        value : int or float
            Value to encode.

        Returns
        -------
        Any
            Embedding of shape `(EMBEDDING_DIM,)`.
        """
        backend = get_backend()
        emb = create_embedding()
        
        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)
        
        # Log-magnitude encoding (for multiplication)
        sign, log_mag = log_encode_value(float(value))
        emb = backend.at_add(emb, 8 + VALUE_OFFSET, sign)
        emb = backend.at_add(emb, 8 + VALUE_OFFSET + 1, log_mag)
        
        # Linear encoding (for addition)
        # Normalize to [-1, 1] range
        linear_val = float(value) / self.scale
        linear_val = max(-1.0, min(1.0, linear_val))  # Clamp
        emb = backend.at_add(emb, 8 + LINEAR_OFFSET, 1.0)  # Valid flag
        emb = backend.at_add(emb, 8 + LINEAR_OFFSET + 1, linear_val)
        
        # Flags
        is_int = isinstance(value, int) or (isinstance(value, float) and value == int(value))
        emb = backend.at_add(emb, 8 + IS_INT_FLAG, 1.0 if is_int else 0.0)
        emb = backend.at_add(emb, 8 + IS_ZERO_FLAG, 1.0 if abs(value) < EPSILON else 0.0)
        emb = backend.at_add(emb, 8 + IS_NEG_FLAG, 1.0 if value < 0 else 0.0)
        
        # Digit encoding (for direct integer representation)
        if is_int and abs(value) < 1e16:
            emb = self._encode_digits(emb, int(value))
        
        return emb
    
    def _encode_digits(self, emb: Any, value: int) -> Any:
        """Encode individual digits for direct integer representation."""
        backend = get_backend()
        abs_val = abs(value)
        digits = []
        
        while abs_val > 0:
            digits.append(abs_val % 10)
            abs_val //= 10
        
        # Encode up to 16 digits
        for i, digit in enumerate(digits[:16]):
            emb = backend.at_add(emb, 8 + DIGITS_OFFSET + 2*i, 1.0)  # Valid flag
            emb = backend.at_add(emb, 8 + DIGITS_OFFSET + 2*i + 1, digit / 10.0)
        
        return emb
    
    def decode(self, emb: Any) -> float:
        """
        Decode embedding to number.
        
        Returns:
            Decoded float value
        """
        # Try digit decoding first for integers
        is_int = emb[8 + IS_INT_FLAG].item() > 0.5
        
        if is_int:
            value = self._decode_digits(emb)
            if value is not None:
                is_neg = emb[8 + IS_NEG_FLAG].item() > 0.5
                return -value if is_neg else value
        
        # Fall back to log-magnitude decoding
        sign = emb[8 + VALUE_OFFSET].item()
        log_mag = emb[8 + VALUE_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)
    
    def _decode_digits(self, emb: Any) -> Optional[int]:
        """Decode digits to integer."""
        value = 0
        multiplier = 1
        
        for i in range(16):
            valid = emb[8 + DIGITS_OFFSET + 2*i].item()
            if valid < 0.5:
                break
            digit = int(round(emb[8 + DIGITS_OFFSET + 2*i + 1].item() * 10.0))
            digit = max(0, min(9, digit))
            value += digit * multiplier
            multiplier *= 10
        
        return value if multiplier > 1 else None
    
    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is valid arithmetic."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()
    
    # =========================================================================
    # Arithmetic Operations
    # =========================================================================
    
    def add(self, emb1: Any, emb2: Any) -> Any:
        """
        Add two numbers.
        
        Computed in the linear component:
        embed(a).linear + embed(b).linear = embed(a+b).linear
        """
        # Get linear values
        lin1 = emb1[8 + LINEAR_OFFSET + 1].item() * self.scale
        lin2 = emb2[8 + LINEAR_OFFSET + 1].item() * self.scale
        
        result = lin1 + lin2
        return self.encode(result)
    
    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """
        Subtract two numbers.
        
        Computed in the linear component.
        """
        lin1 = emb1[8 + LINEAR_OFFSET + 1].item() * self.scale
        lin2 = emb2[8 + LINEAR_OFFSET + 1].item() * self.scale
        
        result = lin1 - lin2
        return self.encode(result)
    
    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Multiply two numbers.
        
        Computed in log-space:
        log(a) + log(b) = log(a * b)
        """
        # Check for zeros
        is_zero1 = emb1[8 + IS_ZERO_FLAG].item() > 0.5
        is_zero2 = emb2[8 + IS_ZERO_FLAG].item() > 0.5
        
        if is_zero1 or is_zero2:
            return self.encode(0)
        
        # Get sign and log-magnitude
        sign1 = emb1[8 + VALUE_OFFSET].item()
        log1 = emb1[8 + VALUE_OFFSET + 1].item()
        sign2 = emb2[8 + VALUE_OFFSET].item()
        log2 = emb2[8 + VALUE_OFFSET + 1].item()
        
        # Multiply: add logs, multiply signs
        result_sign = sign1 * sign2
        result_log = log1 + log2
        
        magnitude = math.exp(result_log)
        result = magnitude if result_sign > 0 else -magnitude
        
        return self.encode(result)
    
    def divide(self, emb1: Any, emb2: Any) -> Any:
        """
        Divide two numbers.
        
        Computed in log-space:
        log(a) - log(b) = log(a / b)
        """
        # Check for zero divisor
        is_zero2 = emb2[8 + IS_ZERO_FLAG].item() > 0.5
        if is_zero2:
            raise ValueError("Division by zero")
        
        # Check for zero dividend
        is_zero1 = emb1[8 + IS_ZERO_FLAG].item() > 0.5
        if is_zero1:
            return self.encode(0)
        
        # Get sign and log-magnitude
        sign1 = emb1[8 + VALUE_OFFSET].item()
        log1 = emb1[8 + VALUE_OFFSET + 1].item()
        sign2 = emb2[8 + VALUE_OFFSET].item()
        log2 = emb2[8 + VALUE_OFFSET + 1].item()
        
        # Divide: subtract logs, multiply signs
        result_sign = sign1 * sign2
        result_log = log1 - log2
        
        magnitude = math.exp(result_log)
        result = magnitude if result_sign > 0 else -magnitude
        
        return self.encode(result)
    
    def negate(self, emb: Any) -> Any:
        """Negate a number."""
        value = self.decode(emb)
        return self.encode(-value)
    
    def abs(self, emb: Any) -> Any:
        """Absolute value."""
        value = self.decode(emb)
        return self.encode(abs(value))
    
    def power(self, emb: Any, n: int) -> Any:
        """
        Raise to integer power.
        
        Computed in log-space: n * log(a) = log(a^n)
        """
        if n == 0:
            return self.encode(1)
        
        is_zero = emb[8 + IS_ZERO_FLAG].item() > 0.5
        if is_zero:
            return self.encode(0)
        
        sign = emb[8 + VALUE_OFFSET].item()
        log_mag = emb[8 + VALUE_OFFSET + 1].item()
        
        # Power: multiply log by n
        result_log = log_mag * n
        
        # Sign: negative^odd = negative
        if sign < 0 and n % 2 == 1:
            result_sign = -1.0
        else:
            result_sign = 1.0
        
        if result_log > 100:
            result_log = 100  # Clamp to prevent overflow
        
        magnitude = math.exp(result_log)
        result = magnitude if result_sign > 0 else -magnitude
        
        return self.encode(result)
    
    def sqrt(self, emb: Any) -> Any:
        """
        Square root.
        
        Computed in log-space: log(a) / 2 = log(sqrt(a))
        """
        is_neg = emb[8 + IS_NEG_FLAG].item() > 0.5
        if is_neg:
            raise ValueError("Cannot take square root of negative number")
        
        is_zero = emb[8 + IS_ZERO_FLAG].item() > 0.5
        if is_zero:
            return self.encode(0)
        
        log_mag = emb[8 + VALUE_OFFSET + 1].item()
        result_log = log_mag / 2
        
        result = math.exp(result_log)
        return self.encode(result)
    
    # =========================================================================
    # Comparisons
    # =========================================================================
    
    def compare(self, emb1: Any, emb2: Any) -> int:
        """
        Compare two numbers.
        
        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        val1 = self.decode(emb1)
        val2 = self.decode(emb2)
        
        if abs(val1 - val2) < EPSILON:
            return 0
        return 1 if val1 > val2 else -1
    
    def is_zero(self, emb: Any) -> bool:
        """Check if number is zero."""
        return emb[8 + IS_ZERO_FLAG].item() > 0.5
    
    def is_negative(self, emb: Any) -> bool:
        """Check if number is negative."""
        return emb[8 + IS_NEG_FLAG].item() > 0.5
    
    def is_integer(self, emb: Any) -> bool:
        """Check if number is an integer."""
        return emb[8 + IS_INT_FLAG].item() > 0.5


# Convenience functions
def encode_number(value: Union[int, float]) -> Any:
    """Encode a number."""
    return ArithmeticEncoder().encode(value)

def decode_number(emb: Any) -> float:
    """Decode a number."""
    return ArithmeticEncoder().decode(emb)

def compute(expr: str) -> float:
    """
    Compute a simple arithmetic expression.
    
    Supports: +, -, *, /
    
    Example:
        compute("123 + 456")  # 579.0
        compute("10 * 20")    # 200.0
    """
    import re
    
    enc = ArithmeticEncoder()
    
    # Parse expression
    match = re.match(r'(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)', expr.strip())
    if not match:
        raise ValueError(f"Cannot parse expression: {expr}")
    
    a = float(match.group(1))
    op = match.group(2)
    b = float(match.group(3))
    
    emb_a = enc.encode(a)
    emb_b = enc.encode(b)
    
    if op == '+':
        result = enc.add(emb_a, emb_b)
    elif op == '-':
        result = enc.subtract(emb_a, emb_b)
    elif op == '*':
        result = enc.multiply(emb_a, emb_b)
    elif op == '/':
        result = enc.divide(emb_a, emb_b)
    else:
        raise ValueError(f"Unknown operator: {op}")
    
    return enc.decode(result)
