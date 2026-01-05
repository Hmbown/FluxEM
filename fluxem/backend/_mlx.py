"""MLX backend implementation (Apple Silicon optimized)."""

from __future__ import annotations

from typing import Any, Tuple, Optional, Union, List

import numpy as np

# MLX imports (will raise ImportError if not installed)
import mlx.core as mx


class MLXBackend:
    """MLX backend for FluxEM (Apple Silicon optimized)."""

    name = "mlx"

    # Array creation
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> mx.array:
        return mx.zeros(shape, dtype=dtype or mx.float32)

    def ones(self, shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> mx.array:
        return mx.ones(shape, dtype=dtype or mx.float32)

    def array(self, data: Any, dtype: Any = None) -> mx.array:
        # Let MLX infer dtype for boolean arrays; only default to float32 for numeric
        if dtype is None:
            arr = mx.array(data)
            # Keep booleans as booleans, otherwise convert to float32
            if arr.dtype != mx.bool_:
                arr = arr.astype(mx.float32)
            return arr
        return mx.array(data, dtype=dtype)

    def arange(self, start: float, stop: Optional[float] = None, step: Optional[float] = None, dtype: Any = None) -> mx.array:
        if stop is None:
            return mx.arange(start)
        if step is None:
            return mx.arange(start, stop)
        return mx.arange(start, stop, step)

    def linspace(self, start: float, stop: float, num: int, dtype: Any = None) -> mx.array:
        result = mx.linspace(start, stop, num)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def eye(self, n: int, dtype: Any = None) -> mx.array:
        return mx.eye(n, dtype=dtype or mx.float32)

    # Math operations
    def exp(self, x: Any) -> mx.array:
        return mx.exp(x)

    def log(self, x: Any) -> mx.array:
        return mx.log(x)

    def log10(self, x: Any) -> mx.array:
        return mx.log10(x)

    def sqrt(self, x: Any) -> mx.array:
        return mx.sqrt(x)

    def abs(self, x: Any) -> mx.array:
        return mx.abs(x)

    def sin(self, x: Any) -> mx.array:
        return mx.sin(x)

    def cos(self, x: Any) -> mx.array:
        return mx.cos(x)

    def tan(self, x: Any) -> mx.array:
        return mx.tan(x)

    def arctan2(self, y: Any, x: Any) -> mx.array:
        return mx.arctan2(y, x)

    def floor(self, x: Any) -> mx.array:
        return mx.floor(x)

    def ceil(self, x: Any) -> mx.array:
        return mx.ceil(x)

    def clip(self, x: Any, min_val: Any, max_val: Any) -> mx.array:
        return mx.clip(x, min_val, max_val)

    def sign(self, x: Any) -> mx.array:
        return mx.sign(x)

    def power(self, x: Any, y: Any) -> mx.array:
        return mx.power(x, y)

    def maximum(self, x: Any, y: Any) -> mx.array:
        return mx.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> mx.array:
        return mx.minimum(x, y)

    # Linear algebra
    def dot(self, a: Any, b: Any) -> mx.array:
        # MLX uses tensordot for 1D vectors
        if a.ndim == 1 and b.ndim == 1:
            return mx.sum(a * b)
        return mx.tensordot(a, b, axes=1)

    def matmul(self, a: Any, b: Any) -> mx.array:
        return mx.matmul(a, b)

    def norm(self, x: Any, axis: Optional[int] = None) -> mx.array:
        return mx.sqrt(mx.sum(x * x, axis=axis))

    def sum(self, x: Any, axis: Optional[int] = None) -> mx.array:
        return mx.sum(x, axis=axis)

    def mean(self, x: Any, axis: Optional[int] = None) -> mx.array:
        return mx.mean(x, axis=axis)

    def prod(self, x: Any, axis: Optional[int] = None) -> mx.array:
        return mx.prod(x, axis=axis)

    def argmax(self, x: Any, axis: Optional[int] = None) -> mx.array:
        return mx.argmax(x, axis=axis)

    def argmin(self, x: Any, axis: Optional[int] = None) -> mx.array:
        return mx.argmin(x, axis=axis)

    # Comparison
    def allclose(self, a: Any, b: Any, atol: float = 1e-8) -> mx.array:
        return mx.allclose(a, b, atol=atol)

    def where(self, condition: Any, x: Any, y: Any) -> mx.array:
        return mx.where(condition, x, y)

    def all(self, x: Any, axis: Optional[int] = None) -> mx.array:
        return mx.all(x, axis=axis)

    def any(self, x: Any, axis: Optional[int] = None) -> mx.array:
        return mx.any(x, axis=axis)

    # Shape operations
    def stack(self, arrays: List[Any], axis: int = 0) -> mx.array:
        return mx.stack(arrays, axis=axis)

    def concatenate(self, arrays: List[Any], axis: int = 0) -> mx.array:
        return mx.concatenate(arrays, axis=axis)

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> mx.array:
        return mx.reshape(x, shape)

    def squeeze(self, x: Any, axis: Optional[int] = None) -> mx.array:
        if axis is None:
            return mx.squeeze(x)
        return mx.squeeze(x, axis=axis)

    def expand_dims(self, x: Any, axis: int) -> mx.array:
        return mx.expand_dims(x, axis=axis)

    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> mx.array:
        if axes is None:
            return mx.transpose(x)
        return mx.transpose(x, axes=axes)

    # Immutable update (functional style)
    def at_set(self, arr: Any, idx: Any, value: Any) -> mx.array:
        """
        MLX only has .at[].add(), so we simulate .set() via .add(value - current).

        Note: This requires reading the current value, which has a small overhead,
        but preserves immutable semantics.
        """
        current = arr[idx]
        return arr.at[idx].add(value - current)

    def at_add(self, arr: Any, idx: Any, value: Any) -> mx.array:
        """MLX uses .at[idx].add(value)"""
        return arr.at[idx].add(value)

    # Type conversion
    def to_numpy(self, x: Any) -> np.ndarray:
        return np.array(x)

    def from_numpy(self, x: Any) -> mx.array:
        return mx.array(x)

    def to_python(self, x: Any) -> Any:
        """Convert scalar array to Python scalar."""
        return x.item() if x.ndim == 0 else x.tolist()

    # Random (for initialization)
    def random_normal(self, shape: Union[int, Tuple[int, ...]], seed: Optional[int] = None) -> mx.array:
        """Generate random normal values."""
        if isinstance(shape, int):
            shape = (shape,)
        if seed is not None:
            mx.random.seed(seed)
        return mx.random.normal(shape)

    def random_uniform(self, shape: Union[int, Tuple[int, ...]], low: float = 0.0, high: float = 1.0, seed: Optional[int] = None) -> mx.array:
        """Generate random uniform values."""
        if isinstance(shape, int):
            shape = (shape,)
        if seed is not None:
            mx.random.seed(seed)
        return mx.random.uniform(low=low, high=high, shape=shape)
