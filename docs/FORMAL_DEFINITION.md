# FluxEM: Formal Definition

## Low-Rank Structure

FluxEM embeddings are **low-rank**:

- **Linear embedding:** Everything lies on a **1D line** in R^d (specifically, `x[0]`)
- **Logarithmic embedding:** Everything lies in a **2D plane** (specifically, `x[0]` for magnitude, `x[1]` for sign)

The high-dimensional embedding (d=256 by default) is a **compatibility wrapper**, not additional capacity. Functionally, FluxEM represents numbers as:

```
(log|n|, sign(n), is_zero(n))
```

lifted into high-dimensional space by padding with zeros. The canonical basis (`e0`, `e1`) eliminates dot product accumulation, making decode a simple index operation.

## Algebraic Structure

The representation decomposes as a **product structure**:

| Component | Group | Operation |
|-----------|-------|-----------|
| Magnitude | (R, +) via log | log-space addition = multiplication |
| Sign | ({±1}, ×) | sign multiplication |
| Zero | Masked element | explicit flag, not algebraic |

This makes the "exactness" claim precise: the homomorphism is exact over R for magnitude; sign is handled discretely; zero is a special case.

## Linear Embedding (Addition/Subtraction)

**Embedding function:**
```
e_lin: R -> R^d
e_lin(n) = (n / scale) * v
```
where `v in R^d` is a fixed unit vector and `scale` is a normalization constant.

**Homomorphism property:**
```
e_lin(a) + e_lin(b) = e_lin(a + b)
```
This is exact as a real-valued identity. Under IEEE-754, error is bounded by floating-point precision (primarily from dot product accumulation in 256D).

**Decode function:**
```
d_lin: R^d -> R
d_lin(x) = scale * (x · v)
```

## Logarithmic Embedding (Multiplication/Division)

**Embedding function:**
```
e_log: R \ {0} -> R^d
e_log(n) = (log|n| / log_scale) * v_mag + sign(n) * v_sign * 0.5
```
where `v_mag, v_sign in R^d` are orthonormal unit vectors (constructed via Gram-Schmidt).

**Homomorphism property (magnitude component):**
```
proj_mag(e_log(a)) + proj_mag(e_log(b)) = proj_mag(e_log(a * b))
```
Sign is tracked separately: `sign(a * b) = sign(a) * sign(b)`.

The sign component is **not** a linear homomorphism under vector addition; it is extracted and recombined in the operator definition.

**Decode function:**
```
d_log: R^d -> R
d_log(x) = sign(x · v_sign) * exp(log_scale * (x · v_mag))
```

## Zero Handling

The logarithmic embedding is defined on R \ {0}. Zero is handled explicitly outside the algebraic embedding:

- **Encode:** map zero to the zero vector, with an explicit zero flag (norm < epsilon)
- **Multiply/divide:** if either operand is zero, return zero (or inf for division)
- **Decode:** if the zero flag is set, return 0 regardless of the embedding vector

This makes zero handling a **masked branch** rather than a property of vector addition.

## What "Exact" Means

The magnitude homomorphism is exact over R (isomorphic structure). Under IEEE-754 float32/float64:

- Errors arise from: `log()`/`exp()` function rounding
- NOT from: learning, approximation, model capacity, or dot product accumulation

With the canonical basis (default), errors are at the `log`/`exp` precision floor:
- Addition/subtraction: < 1e-7 relative error (float32)
- Multiplication/division: < 1e-6 relative error (float32)

See [ERROR_MODEL.md](ERROR_MODEL.md) for precise bounds.

## What This Is

FluxEM is a **deterministic numeric module** for hybrid systems. It provides:

- Algebraic embeddings with homomorphism properties in real arithmetic (approximate under IEEE-754)
- A drop-in numeric primitive, not a complete reasoning system

It does NOT:

- Learn anything (no parameters)
- Handle symbolic manipulation
- Replace general-purpose neural computation

## Unsupported Operations

| Case | Reason | Behavior |
|------|--------|----------|
| Negative base + fractional exponent | Would produce complex result | **Unsupported** — returns real-valued magnitude surrogate |
| log(0) | Undefined | Masked to zero vector |
| Division by zero | Undefined | Returns signed infinity |

## Theoretical Foundation

FluxEM implements a **Generalized Interval System** (Lewin, 1987):

- **S** = numbers (the space of objects)
- **IVLS** = R under + or R+ under * (the interval group)
- **int** = embedding distance (the interval function)

The same mathematical framework that unified 20th-century music theory provides the foundation for deterministic neural arithmetic.
