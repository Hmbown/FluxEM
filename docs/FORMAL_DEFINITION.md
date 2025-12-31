# FluxEM: Formal Definition

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
This is exact in real arithmetic. Under IEEE-754, error is bounded by floating-point precision.

**Decode function:**
```
d_lin: R^d -> R
d_lin(x) = scale * (x . v)
```

## Logarithmic Embedding (Multiplication/Division)

**Embedding function:**
```
e_log: R \ {0} -> R^d
e_log(n) = (log|n| / log_scale) * v_mag + sign(n) * v_sign
```
where `v_mag, v_sign in R^d` are orthogonal unit vectors.

**Homomorphism property (magnitude component):**
```
proj_mag(e_log(a)) + proj_mag(e_log(b)) = proj_mag(e_log(a * b))
```
Sign is tracked separately: `sign(a * b) = sign(a) * sign(b)`

**Decode function:**
```
d_log: R^d -> R
d_log(x) = sign(x . v_sign) * exp(log_scale * (x . v_mag))
```

## What "Exact" Means

The homomorphism properties are exact in R. Under IEEE-754 float32/float64:
- Errors arise from: rounding in exp/log, accumulation in dot products
- NOT from: learning, approximation, or model capacity

See [ERROR_MODEL.md](ERROR_MODEL.md) for precise bounds.

## What This Is

FluxEM is a **deterministic numeric module** for hybrid systems. It provides:
- Algebraic embeddings with guaranteed homomorphism properties
- A drop-in numeric primitive, not a complete reasoning system

It does NOT:
- Learn anything (no parameters)
- Handle symbolic manipulation
- Replace general-purpose neural computation

## Theoretical Foundation

FluxEM implements a **Generalized Interval System** (Lewin, 1987):
- **S** = numbers (the space of objects)
- **IVLS** = R under + or R+ under * (the interval group)
- **int** = embedding distance (the interval function)

The same mathematical framework that unified 20th-century music theory provides the foundation for deterministic neural arithmetic.
