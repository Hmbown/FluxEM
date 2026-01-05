"""NumPy fallback backend (no GPU acceleration)."""

from __future__ import annotations

from typing import Any, Tuple, Optional, Union, List

import numpy as np


class NumPyBackend:
    """NumPy backend for FluxEM (CPU fallback)."""

    name = "numpy"

    # Array creation
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype or np.float32)

    def array(self, data: Any, dtype: Any = None) -> np.ndarray:
        # Let NumPy infer dtype for boolean arrays; only default to float32 for numeric
        if dtype is None:
            arr = np.array(data)
            # Keep booleans as booleans, otherwise convert to float32
            if arr.dtype != np.bool_:
                arr = arr.astype(np.float32)
            return arr
        return np.array(data, dtype=dtype)

    def arange(self, start: float, stop: Optional[float] = None, step: Optional[float] = None, dtype: Any = None) -> np.ndarray:
        if stop is None:
            return np.arange(start, dtype=dtype)
        if step is None:
            return np.arange(start, stop, dtype=dtype)
        return np.arange(start, stop, step, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int, dtype: Any = None) -> np.ndarray:
        return np.linspace(start, stop, num, dtype=dtype)

    def eye(self, n: int, dtype: Any = None) -> np.ndarray:
        return np.eye(n, dtype=dtype or np.float32)

    # Math operations
    def exp(self, x: Any) -> np.ndarray:
        return np.exp(x)

    def log(self, x: Any) -> np.ndarray:
        return np.log(x)

    def log10(self, x: Any) -> np.ndarray:
        return np.log10(x)

    def sqrt(self, x: Any) -> np.ndarray:
        return np.sqrt(x)

    def abs(self, x: Any) -> np.ndarray:
        return np.abs(x)

    def sin(self, x: Any) -> np.ndarray:
        return np.sin(x)

    def cos(self, x: Any) -> np.ndarray:
        return np.cos(x)

    def tan(self, x: Any) -> np.ndarray:
        return np.tan(x)

    def arctan2(self, y: Any, x: Any) -> np.ndarray:
        return np.arctan2(y, x)

    def floor(self, x: Any) -> np.ndarray:
        return np.floor(x)

    def ceil(self, x: Any) -> np.ndarray:
        return np.ceil(x)

    def clip(self, x: Any, min_val: Any, max_val: Any) -> np.ndarray:
        return np.clip(x, min_val, max_val)

    def sign(self, x: Any) -> np.ndarray:
        return np.sign(x)

    def power(self, x: Any, y: Any) -> np.ndarray:
        return np.power(x, y)

    def maximum(self, x: Any, y: Any) -> np.ndarray:
        return np.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> np.ndarray:
        return np.minimum(x, y)

    # Linear algebra
    def dot(self, a: Any, b: Any) -> np.ndarray:
        return np.dot(a, b)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.matmul(a, b)

    def norm(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.linalg.norm(x, axis=axis)

    def sum(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.sum(x, axis=axis)

    def mean(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.mean(x, axis=axis)

    def prod(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.prod(x, axis=axis)

    def argmax(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.argmax(x, axis=axis)

    def argmin(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.argmin(x, axis=axis)

    # Comparison
    def allclose(self, a: Any, b: Any, atol: float = 1e-8) -> np.ndarray:
        # Return a NumPy scalar so callers can use .item() consistently.
        return np.allclose(a, b, atol=atol)

    def where(self, condition: Any, x: Any, y: Any) -> np.ndarray:
        return np.where(condition, x, y)

    def all(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.all(x, axis=axis)

    def any(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.any(x, axis=axis)

    # Shape operations
    def stack(self, arrays: List[Any], axis: int = 0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: List[Any], axis: int = 0) -> np.ndarray:
        return np.concatenate(arrays, axis=axis)

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> np.ndarray:
        return np.reshape(x, shape)

    def squeeze(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.squeeze(x, axis=axis)

    def expand_dims(self, x: Any, axis: int) -> np.ndarray:
        return np.expand_dims(x, axis=axis)

    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        return np.transpose(x, axes=axes)

    # Immutable update (functional style)
    def at_set(self, arr: Any, idx: Any, value: Any) -> np.ndarray:
        """NumPy: copy and modify for immutable semantics."""
        result = arr.copy()
        result[idx] = value
        return result

    def at_add(self, arr: Any, idx: Any, value: Any) -> np.ndarray:
        """NumPy: copy and add for immutable semantics."""
        result = arr.copy()
        result[idx] += value
        return result

    # Type conversion
    def to_numpy(self, x: Any) -> np.ndarray:
        return x if isinstance(x, np.ndarray) else np.array(x)

    def from_numpy(self, x: Any) -> np.ndarray:
        return x if isinstance(x, np.ndarray) else np.array(x)

    def to_python(self, x: Any) -> Any:
        """Convert scalar array to Python scalar."""
        if isinstance(x, np.ndarray):
            return x.item() if x.ndim == 0 else x.tolist()
        return x

    # Random (for initialization)
    def random_normal(self, shape: Union[int, Tuple[int, ...]], seed: Optional[int] = None) -> np.ndarray:
        """Generate random normal values."""
        if isinstance(shape, int):
            shape = (shape,)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype(np.float32)

    def random_uniform(self, shape: Union[int, Tuple[int, ...]], low: float = 0.0, high: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
        """Generate random uniform values."""
        if isinstance(shape, int):
            shape = (shape,)
        rng = np.random.default_rng(seed)
        return rng.uniform(low, high, shape).astype(np.float32)
