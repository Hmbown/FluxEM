# FluxEM: Error Model

## Numeric Precision

FluxEM uses a **canonical basis** (default) where:
- Linear embedding uses only `x[0]`
- Logarithmic embedding uses `x[0]` (magnitude) and `x[1]` (sign)

This eliminates dot product accumulation as an error source. All errors arise from:
1. `log()` and `exp()` function rounding (primary source)
2. IEEE-754 representation limits

## Error Bounds

| Operation | Relative Error (float32) | Relative Error (float64) |
|-----------|-------------------------|-------------------------|
| Addition | < 1e-7 | < 1e-15 |
| Subtraction | < 1e-7 | < 1e-15 |
| Multiplication | < 1e-6 | < 1e-14 |
| Division | < 1e-6 | < 1e-14 |

Multiplication/division have slightly higher error because they go through `log()` → add/subtract → `exp()`.

## Inversion Sanity Check

Round-trip error for `decode(encode(x))`:

| Encoder | Test Values | Typical Relative Error (float32) |
|---------|-------------|--------------------------------|
| Linear | integers in [-100000, 100000] | 0 (exact) |
| Logarithmic | integers in [1, 10000] | < 1e-7 |

## Why the Canonical Basis?

The embedding dimension (d=256 by default) is a **compatibility wrapper**. The actual algebraic structure is:
- Linear: 1D (scalar on a line)
- Logarithmic: 2D (magnitude + sign)

By using `e0` and `e1` as the basis vectors instead of random orthonormal vectors, we:
1. Eliminate dot product accumulation (no 256 multiply-adds)
2. Decode via direct indexing (`x[0]`, `x[1]`) instead of projection
3. Match the formal definition exactly

## Legacy Behavior

If you need the old behavior (random orthonormal basis with dot product decode):

```python
from fluxem import create_unified_model
model = create_unified_model(basis="random_orthonormal")
```

This has slightly higher error (~1e-6 vs ~1e-7 for multiplication) due to dot product accumulation, but may be useful for specific integration scenarios.

## float32 vs float64

Default is float32 (JAX default). For higher precision:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

This reduces relative error by ~8 orders of magnitude.

## Known Failure Modes

| Condition | Behavior |
|-----------|----------|
| Zero input (log space) | Special-cased to zero vector |
| Division by zero | Returns signed infinity |
| Very large magnitude (> 1e38) | Overflow in `exp()` |
| Very small magnitude (< 1e-38) | Underflow, treated as zero |
| Negative base, fractional exponent | **Unsupported** — returns real-valued magnitude only |

## Reproducibility

All tests use:
- Canonical basis (default)
- Deterministic JAX operations
- Consistent embedding dimension (`dim=256` by default)

Results are reproducible across runs on the same hardware.
