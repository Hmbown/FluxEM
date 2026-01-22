"""Signal processing domain - convolution, moving average, DFT.

This module provides deterministic signal processing computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def convolution(a: List[float], b: List[float]) -> List[float]:
    """Compute discrete convolution of two sequences.

    Args:
        a: First sequence (signal)
        b: Second sequence (kernel)

    Returns:
        Convolved sequence of length len(a) + len(b) - 1
    """
    n, m = len(a), len(b)
    result = [0.0] * (n + m - 1)
    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]
    return result


def moving_average(signal: List[float], window: int) -> List[float]:
    """Compute moving average with given window size.

    Args:
        signal: Input signal
        window: Window size

    Returns:
        Averaged signal (length = len(signal) - window + 1)
    """
    if window <= 0 or window > len(signal):
        raise ValueError(f"Invalid window size: {window}")

    result = []
    current_sum = sum(signal[:window])
    result.append(current_sum / window)

    for i in range(window, len(signal)):
        current_sum += signal[i] - signal[i - window]
        result.append(current_sum / window)

    return result


def dft_magnitude(signal: List[float]) -> List[float]:
    """Compute DFT magnitudes for a real-valued signal.

    Args:
        signal: Real-valued time-domain signal

    Returns:
        Magnitude spectrum
    """
    n = len(signal)
    magnitudes = []

    for k in range(n):
        real = 0.0
        imag = 0.0
        for t, x in enumerate(signal):
            angle = -2 * math.pi * k * t / n
            real += x * math.cos(angle)
            imag += x * math.sin(angle)
        magnitudes.append(math.sqrt(real ** 2 + imag ** 2))

    return magnitudes


def autocorrelation(signal: List[float], max_lag: int = None) -> List[float]:
    """Compute autocorrelation of a signal.

    Args:
        signal: Input signal
        max_lag: Maximum lag (default: len(signal) - 1)

    Returns:
        Autocorrelation values for lags 0 to max_lag
    """
    n = len(signal)
    if max_lag is None:
        max_lag = n - 1

    result = []
    for lag in range(max_lag + 1):
        corr = sum(signal[i] * signal[i + lag] for i in range(n - lag))
        result.append(corr)

    return result


def zero_crossings(signal: List[float]) -> int:
    """Count zero crossings in a signal.

    Args:
        signal: Input signal

    Returns:
        Number of zero crossings
    """
    count = 0
    for i in range(1, len(signal)):
        if (signal[i - 1] >= 0 and signal[i] < 0) or (signal[i - 1] < 0 and signal[i] >= 0):
            count += 1
    return count


def peak_to_peak(signal: List[float]) -> float:
    """Compute peak-to-peak amplitude of a signal."""
    if not signal:
        return 0.0
    return max(signal) - min(signal)


def rms(signal: List[float]) -> float:
    """Compute root mean square of a signal."""
    if not signal:
        return 0.0
    return math.sqrt(sum(x ** 2 for x in signal) / len(signal))


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_signal(args) -> List[float]:
    if isinstance(args, dict):
        sig = args.get("signal", args.get("data", args.get("values")))
        if sig is not None:
            return [float(x) for x in sig]
        return [float(x) for x in list(args.values())[0]]
    if isinstance(args, (list, tuple)):
        return [float(x) for x in args]
    raise ValueError(f"Cannot parse signal: {args}")


def _parse_two_signals(args) -> Tuple[List[float], List[float]]:
    if isinstance(args, dict):
        a = args.get("a", args.get("signal"))
        b = args.get("b", args.get("kernel"))
        if a is not None and b is not None:
            return [float(x) for x in a], [float(x) for x in b]
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(x) for x in args[0]], [float(x) for x in args[1]]
    raise ValueError(f"Cannot parse two signals: {args}")


def _parse_moving_average_args(args) -> Tuple[List[float], int]:
    if isinstance(args, dict):
        signal = args.get("signal", args.get("data"))
        window = args.get("window", args.get("window_size", args.get("size")))
        return [float(x) for x in signal], int(window)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(x) for x in args[0]], int(args[1])
    raise ValueError(f"Cannot parse moving average args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register signal processing tools in the registry."""

    registry.register(ToolSpec(
        name="signal_convolution",
        function=lambda args: convolution(*_parse_two_signals(args)),
        description="Computes discrete convolution of two sequences.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "number"}, "description": "First sequence (signal)"},
                "b": {"type": "array", "items": {"type": "number"}, "description": "Second sequence (kernel)"},
            },
            "required": ["a", "b"]
        },
        returns="Convolved sequence",
        examples=[
            {"input": {"a": [1, 2, 3], "b": [1, 1]}, "output": [1.0, 3.0, 5.0, 3.0]},
        ],
        domain="signal_processing",
        tags=["convolution", "filter", "kernel"],
    ))

    registry.register(ToolSpec(
        name="signal_moving_average",
        function=lambda args: moving_average(*_parse_moving_average_args(args)),
        description="Computes moving average with given window size.",
        parameters={
            "type": "object",
            "properties": {
                "signal": {"type": "array", "items": {"type": "number"}, "description": "Input signal"},
                "window": {"type": "integer", "description": "Window size"},
            },
            "required": ["signal", "window"]
        },
        returns="Averaged signal",
        examples=[
            {"input": {"signal": [1, 2, 3, 4, 5], "window": 2}, "output": [1.5, 2.5, 3.5, 4.5]},
        ],
        domain="signal_processing",
        tags=["moving average", "smooth", "filter"],
    ))

    registry.register(ToolSpec(
        name="signal_dft_magnitude",
        function=lambda args: dft_magnitude(_parse_signal(args)),
        description="Computes DFT magnitudes for a real-valued signal.",
        parameters={
            "type": "object",
            "properties": {
                "signal": {"type": "array", "items": {"type": "number"}, "description": "Time-domain signal"}
            },
            "required": ["signal"]
        },
        returns="Magnitude spectrum as list",
        examples=[
            {"input": {"signal": [1, 0, -1, 0]}, "output": [0.0, 2.0, 0.0, 2.0]},
        ],
        domain="signal_processing",
        tags=["dft", "fourier", "spectrum", "frequency"],
    ))
