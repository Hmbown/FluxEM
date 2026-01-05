# Error model

## Overview

This document summarizes numerical error sources for FluxEM encoders and operators.

## Method

### Numeric precision

FluxEM uses a canonical basis (default) where:
- Linear embedding uses only `x[0]`
- Logarithmic embedding uses `x[0]` (magnitude) and `x[1]` (sign)

This avoids dot-product accumulation as an error source. Errors arise from:
1. `log()` and `exp()` function rounding (primary source)
2. IEEE-754 representation limits

### Error bounds

| Operation | Relative Error (float32) | Relative Error (float64) |
|-----------|-------------------------|-------------------------|
| Addition | < 1e-7 | < 1e-15 |
| Subtraction | < 1e-7 | < 1e-15 |
| Multiplication | < 1e-6 | < 1e-14 |
| Division | < 1e-6 | < 1e-14 |

Multiplication/division go through `log()` and `exp()`, which increases rounding relative to linear operations.

### Inversion check

Round-trip error for `decode(encode(x))`:

| Encoder | Test Values | Relative Error (float32) |
|---------|-------------|--------------------------|
| Linear | integers in [-100000, 100000] | 0 (no rounding observed in this range) |
| Logarithmic | integers in [1, 10000] | < 1e-7 |

## Implementation

### Canonical basis

The embedding dimension (d=256 by default) is a compatibility wrapper. The effective algebraic structure is:
- Linear: 1D (scalar on a line)
- Logarithmic: 2D (magnitude + sign)

By using `e0` and `e1` as the basis vectors instead of random orthonormal vectors:
1. Avoid dot-product accumulation
2. Decode via direct indexing (`x[0]`, `x[1]`) instead of projection
3. Match the formal definition

### Legacy behavior

If you need the old behavior (random orthonormal basis with dot-product decode):

```python
from fluxem import create_unified_model
model = create_unified_model(basis="random_orthonormal")
```

This has higher rounding error than the canonical basis due to dot-product accumulation.

### float32 and float64

Default is float32 (JAX default). For higher precision:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

This reduces relative error compared to float32.

### Composition and error accumulation

#### Same-space chains

Operations that stay within one space accumulate minimal error:
- Linear chain (add/sub only): bounded by float rounding
- Log chain (mul/div/pow only): bounded by one `log`/`exp` pair

#### Cross-space chains

When operations cross between linear and log space (e.g., `(a + b) * c`):
1. The first operation completes in the source space
2. The result is decoded to a scalar
3. The scalar is re-encoded in the target space
4. The next operation executes

Each space crossing adds one `log`/`exp` round-trip.

#### Error growth with depth

| Chain Depth | Space Crossings | Relative Error (float32) |
|-------------|-----------------|--------------------------|
| 1 | 0 | < 1e-6 |
| 2 | 1 | < 2e-6 |
| 5 | 4 | < 5e-6 |
| 10 | 9 | < 1e-5 |

Error grows roughly linearly with the number of space crossings.

### Practical guidance

For deep computation chains, consider:
1. Batching same-space operations together when possible
2. Using float64 for precision-sensitive workloads (`jax.config.update("jax_enable_x64", True)`)

## Limitations

### Known failure modes

| Condition | Behavior |
|-----------|----------|
| Zero input (log space) | Special-cased to zero vector |
| Division by zero | Returns signed infinity |
| Very large magnitude (> 1e38) | Overflow in `exp()` |
| Very small magnitude (< 1e-38) | Underflow, treated as zero |
| Negative base, fractional exponent | Unsupported; returns real-valued magnitude only |

## Reproducibility

All tests use:
- Canonical basis (default)
- Deterministic JAX operations
- Consistent embedding dimension (`dim=256` by default)

Results are reproducible across runs on the same hardware.

## References

- [FORMAL_DEFINITION.md](FORMAL_DEFINITION.md)
