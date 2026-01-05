"""JAX backend implementation."""

from __future__ import annotations

from typing import Any, Tuple, Optional, Union, List

import numpy as np

# JAX imports (will raise ImportError if not installed)
import jax
import jax.numpy as jnp


class JAXBackend:
    """JAX backend for FluxEM."""

    name = "jax"

    # Array creation
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> jax.Array:
        return jnp.zeros(shape, dtype=dtype or jnp.float32)

    def ones(self, shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> jax.Array:
        return jnp.ones(shape, dtype=dtype or jnp.float32)

    def array(self, data: Any, dtype: Any = None) -> jax.Array:
        # Let JAX infer dtype for boolean arrays; only default to float32 for numeric
        if dtype is None:
            arr = jnp.array(data)
            # Keep booleans as booleans, otherwise convert to float32
            if arr.dtype != jnp.bool_:
                arr = arr.astype(jnp.float32)
            return arr
        return jnp.array(data, dtype=dtype)

    def arange(self, start: float, stop: Optional[float] = None, step: Optional[float] = None, dtype: Any = None) -> jax.Array:
        if stop is None:
            return jnp.arange(start, dtype=dtype)
        if step is None:
            return jnp.arange(start, stop, dtype=dtype)
        return jnp.arange(start, stop, step, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int, dtype: Any = None) -> jax.Array:
        result = jnp.linspace(start, stop, num)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def eye(self, n: int, dtype: Any = None) -> jax.Array:
        return jnp.eye(n, dtype=dtype or jnp.float32)

    # Math operations
    def exp(self, x: Any) -> jax.Array:
        return jnp.exp(x)

    def log(self, x: Any) -> jax.Array:
        return jnp.log(x)

    def log10(self, x: Any) -> jax.Array:
        return jnp.log10(x)

    def sqrt(self, x: Any) -> jax.Array:
        return jnp.sqrt(x)

    def abs(self, x: Any) -> jax.Array:
        return jnp.abs(x)

    def sin(self, x: Any) -> jax.Array:
        return jnp.sin(x)

    def cos(self, x: Any) -> jax.Array:
        return jnp.cos(x)

    def tan(self, x: Any) -> jax.Array:
        return jnp.tan(x)

    def arctan2(self, y: Any, x: Any) -> jax.Array:
        return jnp.arctan2(y, x)

    def floor(self, x: Any) -> jax.Array:
        return jnp.floor(x)

    def ceil(self, x: Any) -> jax.Array:
        return jnp.ceil(x)

    def clip(self, x: Any, min_val: Any, max_val: Any) -> jax.Array:
        return jnp.clip(x, min_val, max_val)

    def sign(self, x: Any) -> jax.Array:
        return jnp.sign(x)

    def power(self, x: Any, y: Any) -> jax.Array:
        return jnp.power(x, y)

    def maximum(self, x: Any, y: Any) -> jax.Array:
        return jnp.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> jax.Array:
        return jnp.minimum(x, y)

    # Linear algebra
    def dot(self, a: Any, b: Any) -> jax.Array:
        return jnp.dot(a, b)

    def matmul(self, a: Any, b: Any) -> jax.Array:
        return jnp.matmul(a, b)

    def norm(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.linalg.norm(x, axis=axis)

    def sum(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.sum(x, axis=axis)

    def mean(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.mean(x, axis=axis)

    def prod(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.prod(x, axis=axis)

    def argmax(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.argmax(x, axis=axis)

    def argmin(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.argmin(x, axis=axis)

    # Comparison
    def allclose(self, a: Any, b: Any, atol: float = 1e-8) -> jax.Array:
        return jnp.allclose(a, b, atol=atol)

    def where(self, condition: Any, x: Any, y: Any) -> jax.Array:
        return jnp.where(condition, x, y)

    def all(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.all(x, axis=axis)

    def any(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.any(x, axis=axis)

    # Shape operations
    def stack(self, arrays: List[Any], axis: int = 0) -> jax.Array:
        return jnp.stack(arrays, axis=axis)

    def concatenate(self, arrays: List[Any], axis: int = 0) -> jax.Array:
        return jnp.concatenate(arrays, axis=axis)

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> jax.Array:
        return jnp.reshape(x, shape)

    def squeeze(self, x: Any, axis: Optional[int] = None) -> jax.Array:
        return jnp.squeeze(x, axis=axis)

    def expand_dims(self, x: Any, axis: int) -> jax.Array:
        return jnp.expand_dims(x, axis=axis)

    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> jax.Array:
        return jnp.transpose(x, axes=axes)

    # Immutable update (functional style)
    def at_set(self, arr: Any, idx: Any, value: Any) -> jax.Array:
        """JAX uses .at[idx].set(value)"""
        return arr.at[idx].set(value)

    def at_add(self, arr: Any, idx: Any, value: Any) -> jax.Array:
        """JAX uses .at[idx].add(value)"""
        return arr.at[idx].add(value)

    # Type conversion
    def to_numpy(self, x: Any) -> np.ndarray:
        return np.array(x)

    def from_numpy(self, x: Any) -> jax.Array:
        return jnp.array(x)

    def to_python(self, x: Any) -> Any:
        """Convert scalar array to Python scalar."""
        return float(x) if x.ndim == 0 else x.tolist()

    # Random (for initialization)
    def random_normal(self, shape: Union[int, Tuple[int, ...]], seed: Optional[int] = None) -> jax.Array:
        """Generate random normal values."""
        if isinstance(shape, int):
            shape = (shape,)
        key = jax.random.PRNGKey(seed if seed is not None else 42)
        return jax.random.normal(key, shape)

    def random_uniform(self, shape: Union[int, Tuple[int, ...]], low: float = 0.0, high: float = 1.0, seed: Optional[int] = None) -> jax.Array:
        """Generate random uniform values."""
        if isinstance(shape, int):
            shape = (shape,)
        key = jax.random.PRNGKey(seed if seed is not None else 42)
        return jax.random.uniform(key, shape, minval=low, maxval=high)
