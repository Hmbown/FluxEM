# Embedding comparison: algebraic and learned representations

## Overview

This document surveys embedding approaches for numerical and structured inputs in neural networks, with emphasis on arithmetic and scientific computation.

## Method

### Tokenization of numerals

Large language models tokenize input text into subword units. When a number like `123456` enters the model, it undergoes tokenization:

```
Input:  "123456"
Tokens: ['123', '456']     # BPE tokenization
   or:  ['1', '2', '3', '4', '5', '6']  # Character-level
   or:  ['12', '34', '56']  # Subword variation
```

Each token becomes a learned embedding vector. A model must infer arithmetic relationships from training examples, for example:
- `123 + 456 = 579`
- `1234 + 5678 = 6912`
- The same pattern generalizes to all integers

### Failure modes

**In-distribution (ID)**: Models can interpolate patterns seen during training.

**Out-of-distribution (OOD)**: When inputs exceed the training distribution, accuracy can degrade:

| Failure Mode | Example | Why It Fails |
|--------------|---------|--------------|
| Large magnitude | `999999999 + 1` | Limited coverage of long numbers |
| Long expressions | `a + b + c + d + e + f` | Shorter chains in training data |
| Novel operations | `3 ** 17` | Underrepresented operations |
| Precision | `3.14159265 * 2` | Limited decimal resolution |

Tokenization does not preserve algebraic structure for arithmetic operations.

## Implementation

### Algebraic embeddings

Algebraic embeddings define an encoder/decoder pair where selected operations correspond to closed-form operations in embedding space.

```python
# Token embeddings do not enforce arithmetic constraints.
# Algebraic embeddings enforce constraints via the encoder/operator definitions.
```

### Encoders

FluxEM includes two numeric encodings:

#### Linear encoding (addition/subtraction)

```
encode(x) = [x * scale, x * scale, ..., metadata]
decode(v) = v[0] / scale

Identity: decode(encode(a) + encode(b)) = a + b
```

Addition in the original domain becomes vector addition in embedding space.

#### Log encoding (multiplication/division/powers)

```
log_encode(x) = [log(|x|) * scale, sign_bits, ..., metadata]

Identity (magnitude): log_encode(a) + log_encode(b) = log_encode(a * b)
                     k * log_encode(a) = log_encode(a ** k)
```

Multiplication becomes addition in log-space. Powers become scalar multiplication.

### Properties (numeric)

| Property | Token embeddings | FluxEM |
|----------|------------------|--------|
| Operation fidelity | Model-dependent | Defined by encoder/operator implementation |
| Training required | Yes | No (encoder/operator) |
| OOD generalization | Distribution-dependent | Depends on parsing and floating-point range |
| Parameters | Learned | Zero (deterministic) |
| Precision | Model-dependent | Float32/float64 semantics |

### Comparison categories

#### Tokenization approaches

Methods that convert numbers to discrete tokens.

| Approach | Description | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| Character-level | Each digit is a token | Simple, consistent | No numeric structure |
| BPE/WordPiece | Subword tokenization | Works with text pipeline | Inconsistent number splits |
| Digit-position | Positional encoding per digit (Abacus) | Better OOD than BPE | Still requires learning |
| Scientific notation | Mantissa + exponent tokens | Handles magnitude | Complex tokenization |

All tokenization approaches require learning arithmetic behavior from examples.

#### Learned numeric embeddings

Methods that learn representations for numbers.

| Approach | Description | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| Word2vec-style | Treat numbers as words | Simple | No numeric structure |
| NALU (Trask, 2018) | Learned gates for +, -, *, / | End-to-end differentiable | Training instability |
| xVal (Golkar, 2023) | Learned scaling direction | Better interpolation | Still requires training |
| NumBed | Learned position-magnitude | Smooth interpolation | OOD degradation |

Learned embeddings inherit the training distribution.

#### Semantic embeddings

General-purpose text embeddings applied to numbers.

| Approach | Description | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| Sentence-Transformers | Semantic similarity embeddings | Text similarity | Numbers are arbitrary symbols |
| OpenAI Embeddings | Large-scale semantic embeddings | General purpose | No arithmetic structure |
| BERT embeddings | Contextual token embeddings | Rich representations | Arithmetic is implicit |

Semantic embeddings optimize for meaning similarity, not algebraic structure.

#### Algebraic embeddings (FluxEM)

Embeddings designed to preserve algebraic operations.

| Approach | Description | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| Linear encoding | `embed(a) + embed(b) = embed(a+b)` | Closed-form addition | Single operation |
| Log encoding | `embed(a) + embed(b) = embed(a*b)` | Closed-form multiplication | Requires log transform |
| Unified encoding | Combined linear + log | Basic operations | Slightly larger embedding |
| Domain encoding | Units, molecules, etc. | Domain-specific algebra | Domain-specific |

Correctness follows from encoder/operator definitions, subject to floating-point semantics.

## Limitations

### Scope

- Closed-form numeric operators for supported encoders
- Deterministic and reproducible embeddings

### Non-goals

- Approximate reasoning for underspecified inputs
- Operators outside the implemented algebra for each domain
- Fuzzy matching between symbolic aliases
- Automatic error correction of malformed strings

## Reproducibility

For runnable experiments and output formats, see:
- [EXPERIMENTS.md](EXPERIMENTS.md)
- [HYBRID_TRAINING.md](HYBRID_TRAINING.md)

Example summary schema (placeholders):

| Method | ID Accuracy | OOD-A | OOD-B | OOD-C | Parameters | Training |
|--------|-------------|-------|-------|-------|------------|----------|
| fluxem | ... | ... | ... | ... | 0 | none |
| model_x | ... | ... | ... | ... | ... | ... |

## References

- [NALU: Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) (Trask et al., 2018)
- [xVal: Continuous Number Encoding](https://arxiv.org/abs/2310.02989) (Golkar et al., 2023)
- [Abacus Embeddings](https://arxiv.org/abs/2405.17399) (McLeish et al., 2024)
- [FluxEM README](../README.md) - Project overview
