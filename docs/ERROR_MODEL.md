# FluxEM: Error Model

## Numeric Precision

FluxEM operates in IEEE-754 floating point. All errors arise from:
1. Rounding in exp() and log() functions
2. Accumulation in dot products
3. Representation limits (denormals, overflow)

## Tested Guarantees

| Operation | Test Range | Samples | Relative Error |
|-----------|------------|---------|----------------|
| Addition | [-100000, 100000] | 200 | < 1e-6 |
| Subtraction | [-100000, 100000] | 200 | < 1e-6 |
| Multiplication | [10, 1000] x [10, 1000] | 200 | < 1% |
| Division | [100, 10000] / [10, 100] | 200 | < 1% |

"100% OOD accuracy" means: all samples within these ranges achieved < 1% relative error (or < 0.5 absolute error when |expected| <= 1).

## Known Failure Modes

| Condition | Behavior |
|-----------|----------|
| Zero input (log space) | Special-cased to zero vector |
| Division by zero | Returns signed infinity |
| Very large magnitude (> 1e38) | Overflow in exp() |
| Very small magnitude (< 1e-38) | Underflow, treated as zero |
| Negative base, fractional exponent | Returns magnitude only (no complex) |

## float32 vs float64

Default is float32 (JAX default). For higher precision:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

This reduces relative error by ~8 orders of magnitude for most operations.

## Error Sources by Operation

### Addition/Subtraction (Linear Space)
- Primary error: floating-point rounding in dot product
- Typical relative error: < 1e-6 (float32), < 1e-14 (float64)
- Error is **independent** of operand magnitude (relative error is constant)

### Multiplication/Division (Log Space)
- Primary error: exp() and log() function rounding
- Secondary error: sign component dot product
- Typical relative error: < 1% (float32), < 0.0001% (float64)
- Error grows slightly with magnitude due to exp() amplification

### Powers/Roots (Scalar Multiplication in Log Space)
- Primary error: exp() at decode time
- Error depends on exponent magnitude
- Large exponents (> 10) may show increased error

## Reproducibility

All tests use:
- Seeded random vectors (`seed=42` by default)
- Deterministic JAX operations
- Consistent embedding dimension (`dim=256` by default)

Results are reproducible across runs on the same hardware.
