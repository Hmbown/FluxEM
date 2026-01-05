"""
Backend abstraction for FluxEM.

Supports JAX, MLX, and NumPy backends with identical APIs.
Auto-selects based on availability and platform.

Usage:
    from fluxem.backend import get_backend, set_backend, BackendType

    # Auto-select an available backend
    backend = get_backend()

    # Or explicitly set
    set_backend(BackendType.MLX)

    # Use backend for array operations
    emb = backend.zeros(128)
    emb = backend.at_set(emb, 0, 1.0)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, Union, Tuple, Optional, List
from enum import Enum
import os


class BackendType(Enum):
    """Supported backend types."""
    JAX = "jax"
    MLX = "mlx"
    NUMPY = "numpy"


@runtime_checkable
class Backend(Protocol):
    """Protocol that all backends must implement."""

    name: str

    # Array creation
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> Any: ...
    def ones(self, shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> Any: ...
    def array(self, data: Any, dtype: Any = None) -> Any: ...
    def arange(self, start: float, stop: Optional[float] = None, step: Optional[float] = None, dtype: Any = None) -> Any: ...
    def linspace(self, start: float, stop: float, num: int, dtype: Any = None) -> Any: ...
    def eye(self, n: int, dtype: Any = None) -> Any: ...

    # Math operations
    def exp(self, x: Any) -> Any: ...
    def log(self, x: Any) -> Any: ...
    def log10(self, x: Any) -> Any: ...
    def sqrt(self, x: Any) -> Any: ...
    def abs(self, x: Any) -> Any: ...
    def sin(self, x: Any) -> Any: ...
    def cos(self, x: Any) -> Any: ...
    def tan(self, x: Any) -> Any: ...
    def arctan2(self, y: Any, x: Any) -> Any: ...
    def floor(self, x: Any) -> Any: ...
    def ceil(self, x: Any) -> Any: ...
    def clip(self, x: Any, min_val: Any, max_val: Any) -> Any: ...
    def sign(self, x: Any) -> Any: ...
    def power(self, x: Any, y: Any) -> Any: ...
    def maximum(self, x: Any, y: Any) -> Any: ...
    def minimum(self, x: Any, y: Any) -> Any: ...

    # Linear algebra
    def dot(self, a: Any, b: Any) -> Any: ...
    def matmul(self, a: Any, b: Any) -> Any: ...
    def norm(self, x: Any, axis: Optional[int] = None) -> Any: ...
    def sum(self, x: Any, axis: Optional[int] = None) -> Any: ...
    def mean(self, x: Any, axis: Optional[int] = None) -> Any: ...
    def prod(self, x: Any, axis: Optional[int] = None) -> Any: ...
    def argmax(self, x: Any, axis: Optional[int] = None) -> Any: ...
    def argmin(self, x: Any, axis: Optional[int] = None) -> Any: ...

    # Comparison
    def allclose(self, a: Any, b: Any, atol: float = 1e-8) -> Any: ...
    def where(self, condition: Any, x: Any, y: Any) -> Any: ...
    def all(self, x: Any, axis: Optional[int] = None) -> Any: ...
    def any(self, x: Any, axis: Optional[int] = None) -> Any: ...

    # Shape operations
    def stack(self, arrays: List[Any], axis: int = 0) -> Any: ...
    def concatenate(self, arrays: List[Any], axis: int = 0) -> Any: ...
    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any: ...
    def squeeze(self, x: Any, axis: Optional[int] = None) -> Any: ...
    def expand_dims(self, x: Any, axis: int) -> Any: ...
    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> Any: ...

    # Immutable update (functional style)
    def at_set(self, arr: Any, idx: Any, value: Any) -> Any: ...
    def at_add(self, arr: Any, idx: Any, value: Any) -> Any: ...

    # Type conversion
    def to_numpy(self, x: Any) -> Any: ...
    def from_numpy(self, x: Any) -> Any: ...
    def to_python(self, x: Any) -> Any: ...

    # Random (for initialization)
    def random_normal(self, shape: Union[int, Tuple[int, ...]], seed: Optional[int] = None) -> Any: ...
    def random_uniform(self, shape: Union[int, Tuple[int, ...]], low: float = 0.0, high: float = 1.0, seed: Optional[int] = None) -> Any: ...


# Backend selection
_current_backend: Optional[Backend] = None


def get_backend() -> Backend:
    """
    Get the current backend, auto-selecting if not set.

    Selection priority:
    1. FLUXEM_BACKEND environment variable
    2. JAX (if installed)
    3. MLX (if on Apple Silicon and installed)
    4. NumPy (fallback)

    Returns:
        Backend instance
    """
    global _current_backend
    if _current_backend is None:
        _current_backend = _auto_select_backend()
    return _current_backend


def set_backend(backend: Union[BackendType, str]) -> None:
    """
    Explicitly set the backend.

    Args:
        backend: BackendType enum or string ("jax", "mlx", "numpy")
    """
    global _current_backend
    if isinstance(backend, str):
        backend = BackendType(backend.lower())
    _current_backend = _load_backend(backend)


def reset_backend() -> None:
    """Reset the backend to trigger re-selection on next get_backend() call."""
    global _current_backend
    _current_backend = None


def _auto_select_backend() -> Backend:
    """Auto-select backend based on availability and platform."""
    # Check environment variable override
    env_backend = os.environ.get("FLUXEM_BACKEND")
    if env_backend:
        try:
            return _load_backend(BackendType(env_backend.lower()))
        except (ValueError, ImportError):
            pass  # Fall through to auto-detection

    # Try JAX first (existing users expect this)
    try:
        from ._jax import JAXBackend
        return JAXBackend()
    except ImportError:
        pass

    # Try MLX on Apple Silicon
    import platform
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        try:
            from ._mlx import MLXBackend
            return MLXBackend()
        except ImportError:
            pass

    # Fallback to NumPy
    from ._numpy import NumPyBackend
    return NumPyBackend()


def _load_backend(backend_type: BackendType) -> Backend:
    """Load a specific backend."""
    if backend_type == BackendType.JAX:
        from ._jax import JAXBackend
        return JAXBackend()
    elif backend_type == BackendType.MLX:
        from ._mlx import MLXBackend
        return MLXBackend()
    else:
        from ._numpy import NumPyBackend
        return NumPyBackend()


__all__ = [
    "Backend",
    "BackendType",
    "get_backend",
    "set_backend",
    "reset_backend",
]
