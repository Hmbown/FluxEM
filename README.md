# FluxEM

**Stop teaching neural networks arithmetic. Encode the algebra.**

## The Problem

Large language models fail at arithmetic. They can explain calculus but fumble `1847 x 392`. Seven years of research (NALU 2018, xVal 2023, Abacus 2024) tried to fix this by teaching networks to learn arithmetic better.

FluxEM takes a different approach: **don't learn arithmetic. Encode it.**

## The Insight

In the right embedding space, arithmetic operations become geometry:

- **Addition:** `embed(a) + embed(b) = embed(a + b)` (linear embeddings)
- **Multiplication:** `log_embed(a) + log_embed(b) = log_embed(a * b)` (log embeddings)

This isn't learned. It's algebraic. Out-of-distribution generalization is guaranteed because the structure IS the solution.

## The Connection

This approach has a 100-year history in music theory:

| Year | Development |
|------|-------------|
| 1739 | Euler's Tonnetz - geometric pitch embedding |
| 1920s | Schoenberg's 12-tone matrix - transformations as structure |
| 1973 | Forte's set theory - classify by invariants |
| 1987 | Lewin's GIS - formal framework for transformation-based theory |
| 2025 | FluxEM - same insight applied to neural arithmetic |

FluxEM is a **Generalized Interval System** (Lewin, 1987) for numbers:
- **S** = numbers (Lewin: musical objects)
- **IVLS** = R under + or R+ under x (Lewin: interval group)
- **int** = embedding distance (Lewin: interval function)

The framework that unified 20th-century music theory also solves 21st-century neural arithmetic.

## Key Result

| Operation | OOD Accuracy (<=1% rel error) | Method |
|-----------|-------------------------------|--------|
| Addition | 100% | Linear embedding homomorphism |
| Subtraction | 100% | Linear embedding homomorphism |
| Multiplication | 100% | Log-magnitude homomorphism + sign |
| Division | 100% | Log-magnitude homomorphism + sign |
| Powers | 100% | Scalar multiplication in log-magnitude space |
| Roots | 100% | Fractional scalar multiplication |

Tested OOD ranges (relative error < 1%):
- Addition/Subtraction: [-100000, 100000]
- Multiplication: [10, 1000] x [10, 1000]
- Division: [100, 10000] / [10, 100]

## Domain and Guarantees

- **Domain:** Real numbers under IEEE-754 floating point
- **Exactness:** Identities are exact in real arithmetic; numerical error bounded by float precision
- **Zero:** Handled explicitly (log(0) is undefined)
- **Sign:** Tracked separately from magnitude in log space
- **Directions:** Fixed random unit vectors (seeded), not learned
- **Division by zero:** Returns signed infinity

## What "100%" Means

- All sampled OOD tests are within 1% relative error (or 0.5 absolute error for |expected| <= 1)
- Ranges are the sampled intervals listed above, not an unbounded guarantee

## Relation to Prior Work

| Approach | Method | Limitation |
|----------|--------|------------|
| NALU (Trask, 2018) | Log/exp with learned gates | Gates can fail to generalize |
| xVal (Golkar, 2023) | Learned scaling direction | Direction must be learned |
| Abacus (McLeish, 2024) | Positional digit encoding | Character-level, not continuous |
| **FluxEM** | Fixed algebraic structure | No learned parameters |

FluxEM's contribution: composable embeddings where arithmetic is exact by construction.

## Limitations

Be aware:
- **Zero handling:** `log(0)` is undefined; zero uses special-case logic
- **Sign:** Magnitude and sign tracked separately (like polar coordinates)
- **Precision:** "Exact" means exact in real arithmetic, limited by float32
- **Tested ranges:** OOD claims based on sampled ranges (see tests for details)
- **Negative fractional exponents:** Returns magnitude only (no complex values)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from flux_lm import create_unified_model

model = create_unified_model(dim=256)

model.compute("12345+67890")  # -> 80235.0
model.compute("456*789")      # -> 359784.0
model.compute("56088/123")    # -> 456.0
```

## Integration

FluxEM can be used as a drop-in numeric embedding primitive:

- Replace number token embeddings with FluxEM embeddings (linear or log as needed)
- Concatenate FluxEM embeddings with learned token embeddings for mixed text/number inputs
- Decode outputs with the corresponding encoder when a numeric value is required

## Research Directions

FluxEM demonstrates a general principle: **encode structure, don't learn it.**

Open questions:
- Can this extend to symbolic differentiation? (See `fluxem/differentiation.py`)
- What about compositional generalization in language?
- Is there a unified embedding space for all arithmetic?

See the [Flux Mathematics textbook](../textbook/) for the theoretical framework.

## Why "FluxEM"?

- **Flux**: From [Flux Mathematics](../textbook/), inspired by Hindemith's transformation-first thinking - intervals are primary, pitches emerge
- **EM**: Embedding Model. Like word2vec, but for numbers

The approach is technically a **Generalized Interval System** (Lewin, 1987), but the name honors the philosophical lineage: Hindemith's insight that relationships define objects, not the other way around.

## Origins

This project emerged from a music theory graduate seminar at SMU with Professor Frank, where Hindemith's intervallic analysis, set theory, and transformation-based approaches clicked into place.

The author spent 7 years as a music educator, is a trumpet player and vocalist, and saw the mismatch: ML treats numbers like arbitrary tokens while music theory has been treating pitch like geometry for a century.

FluxEM exists because a musician noticed that embedding arithmetic for neural networks is the same problem music theorists solved decades ago.

## References

### Music Theory (Theoretical Lineage)
- Lewin, D. (1987). *Generalized Musical Intervals and Transformations*
- Cohn, R. (1998). "Introduction to Neo-Riemannian Theory"
- Tymoczko, D. (2011). *A Geometry of Music*
- Forte, A. (1973). *The Structure of Atonal Music*
- Euler, L. (1739). *Tentamen novae theoriae musicae*

### ML Arithmetic (Problem Space)
- Trask, A. et al. (2018). "Neural Arithmetic Logic Units"
- Golkar, S. et al. (2023). "xVal: A Continuous Number Encoding"
- McLeish, S. et al. (2024). "Transformers Can Do Arithmetic with the Right Embeddings"

## Citation

```bibtex
@software{fluxem2025,
  title={FluxEM: Algebraic Embeddings for Deterministic Arithmetic},
  author={Hunter Bown},
  year={2025},
  note={Applies Lewin's Generalized Interval Systems to neural arithmetic}
}
```

## License

Apache 2.0
