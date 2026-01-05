# Formal definition

## Overview

This document defines the embedding structure and algebraic properties used by FluxEM encoders.

## Method

### Low-rank structure

FluxEM embeddings are low-rank:

- Linear embedding: a 1D subspace in R^d (specifically, `x[0]`)
- Logarithmic embedding: a 2D subspace (specifically, `x[0]` for magnitude, `x[1]` for sign)

The high-dimensional embedding (d=256 by default) is a compatibility wrapper. Functionally, the representation is:

```
(log|n|, sign(n), is_zero(n))
```

lifted into high-dimensional space by padding with zeros. The canonical basis (`e0`, `e1`) removes dot-product accumulation and makes decoding a direct index operation.

### Algebraic structure

The representation decomposes as a product structure:

| Component | Group | Operation |
|-----------|-------|-----------|
| Magnitude | (R, +) via log | log-space addition = multiplication |
| Sign | ({+-1}, x) | sign multiplication |
| Zero | Masked element | explicit flag, not algebraic |

The homomorphism is defined in real arithmetic for the magnitude component; sign is handled discretely; zero is a special case.

### Linear embedding (addition/subtraction)

Embedding function:

```
e_lin: R -> R^d
e_lin(n) = (n / scale) * v
```

where `v in R^d` is a fixed unit vector and `scale` is a normalization constant.

Homomorphism property:

```
e_lin(a) + e_lin(b) = e_lin(a + b)
```

In finite precision, rounding is bounded by float semantics.

Decode function:

```
d_lin: R^d -> R
d_lin(x) = scale * (x dot v)
```

### Logarithmic embedding (multiplication/division)

Embedding function:

```
e_log: R \ {0} -> R^d
e_log(n) = (log|n| / log_scale) * v_mag + sign(n) * v_sign * 0.5
```

where `v_mag, v_sign in R^d` are orthonormal unit vectors (constructed via Gram-Schmidt).

Homomorphism property (magnitude component):

```
proj_mag(e_log(a)) + proj_mag(e_log(b)) = proj_mag(e_log(a * b))
```

Sign is tracked separately: `sign(a * b) = sign(a) * sign(b)`.

The sign component is not a linear homomorphism under vector addition; it is extracted and recombined in the operator definition.

Decode function:

```
d_log: R^d -> R
d_log(x) = sign(x dot v_sign) * exp(log_scale * (x dot v_mag))
```

### Zero handling

The logarithmic embedding is defined on R \ {0}. Zero is handled explicitly outside the algebraic embedding:

- Encode: map zero to the zero vector, with an explicit zero flag (norm < epsilon)
- Multiply/divide: if either operand is zero, return zero (or inf for division)
- Decode: if the zero flag is set, return 0 regardless of the embedding vector

This makes zero handling a masked branch rather than a property of vector addition.

### Precision semantics

The magnitude homomorphism is defined over the reals. Under IEEE-754 float32/float64, precision is bounded by:

- `log()`/`exp()` function rounding (primary source)
- IEEE-754 representation limits (secondary)

With the canonical basis (default), errors are at the `log`/`exp` precision floor:
- Addition/subtraction: < 1e-7 relative error (float32)
- Multiplication/division: < 1e-6 relative error (float32)

Note: "Closed under operations" means the result of any supported operation on valid inputs is a valid output, not that results are mathematically exact. All claims are subject to IEEE-754 floating-point semantics.

See [ERROR_MODEL.md](ERROR_MODEL.md) for precise bounds and composition error accumulation.

## Limitations

### Scope

FluxEM is a deterministic numeric module for hybrid systems. It provides:

- Algebraic embeddings with homomorphism properties in real arithmetic (approximate under IEEE-754)
- A drop-in numeric primitive, not a complete reasoning system

It does not:

- Learn anything (no parameters)
- Handle symbolic manipulation
- Replace general-purpose neural computation

### Unsupported operations

| Case | Reason | Behavior |
|------|--------|----------|
| Negative base + fractional exponent | Would produce complex result | Unsupported; returns real-valued magnitude surrogate |
| log(0) | Undefined | Masked to zero vector |
| Division by zero | Undefined | Returns signed infinity |

## References

- [ERROR_MODEL.md](ERROR_MODEL.md)
- [HYBRID_TRAINING.md](HYBRID_TRAINING.md)
