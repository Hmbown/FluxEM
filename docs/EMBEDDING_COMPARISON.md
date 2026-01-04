# Embedding Comparison: Why Algebraic Embeddings Matter

This document provides a systematic comparison of embedding approaches for numerical and structured data in neural networks, with a focus on arithmetic and scientific computation.

---

## 1. The Problem: Numbers Are Not Words

### How LLMs See Numbers

Large language models tokenize input text into subword units. When a number like `123456` enters the model, it undergoes tokenization:

```
Input:  "123456"
Tokens: ['123', '456']     # BPE tokenization
   or:  ['1', '2', '3', '4', '5', '6']  # Character-level
   or:  ['12', '34', '56']  # Subword variation
```

Each token becomes a learned embedding vector. The model must then **learn from examples** that:
- `123 + 456 = 579`
- `1234 + 5678 = 6912`
- The same pattern generalizes to all integers

### Why This Fails

**In-distribution (ID)**: Models can memorize or interpolate patterns seen during training.

**Out-of-distribution (OOD)**: When inputs exceed training distribution, models fail:

| Failure Mode | Example | Why It Fails |
|--------------|---------|--------------|
| Large magnitude | `999999999 + 1` | Never saw 9-digit numbers |
| Long expressions | `a + b + c + d + e + f` | Trained on 2-3 operand chains |
| Novel operations | `3 ** 17` | Power operation underrepresented |
| Precision | `3.14159265 * 2` | Decimal precision not learned |

The fundamental issue: **arithmetic is not a statistical pattern**. It's an algebraic structure that tokenization destroys.

### The Tokenization Tax

When models learn arithmetic from tokens, they pay a hidden cost:

1. **Training data**: Millions of arithmetic examples needed
2. **Parameters**: Model capacity spent on rediscovering algebra
3. **Generalization**: Each new magnitude/operation requires retraining
4. **Reliability**: No guarantees on unseen inputs

---

## 2. The FluxEM Solution: Algebraic Embeddings

### Core Insight

Instead of learning arithmetic, **encode it directly into the embedding geometry**.

```python
# Traditional: Model must LEARN that addition works
embed("123") + embed("456") != embed("579")  # Random relationship

# FluxEM: Addition is CONSTRUCTED to work
encode(123) + encode(456) = encode(579)  # By definition
```

### How It Works

FluxEM uses two complementary encodings:

#### Linear Encoding (Addition/Subtraction)

```
encode(x) = [x * scale, x * scale, ..., metadata]
decode(v) = v[0] / scale

Guarantee: decode(encode(a) + encode(b)) = a + b
```

Addition in the original domain becomes vector addition in embedding space.

#### Log Encoding (Multiplication/Division/Powers)

```
log_encode(x) = [log(|x|) * scale, sign_bits, ..., metadata]

Guarantee: log_encode(a) + log_encode(b) = log_encode(a * b)
           k * log_encode(a) = log_encode(a ** k)
```

Multiplication becomes addition in log-space. Powers become scalar multiplication.

### Properties

| Property | Token Embeddings | FluxEM |
|----------|------------------|--------|
| Arithmetic correctness | Learned, approximate | Exact by construction |
| Training required | Yes (millions of examples) | No |
| OOD generalization | Poor | Perfect (homomorphism) |
| Parameters | Learned | Zero (deterministic) |
| Precision | Model-dependent | Float32/64 precision |

---

## 3. Comparison Categories

### 3.1 Tokenization Approaches

Methods that convert numbers to discrete tokens.

| Approach | Description | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| **Character-level** | Each digit is a token | Simple, consistent | No numeric structure |
| **BPE/WordPiece** | Subword tokenization | Works with text pipeline | Inconsistent number splits |
| **Digit-position** | Positional encoding per digit (Abacus) | Better OOD than BPE | Still requires learning |
| **Scientific notation** | Mantissa + exponent tokens | Handles magnitude | Complex tokenization |

**Limitation**: All tokenization approaches require the model to learn arithmetic from examples.

### 3.2 Learned Numeric Embeddings

Methods that learn representations for numbers.

| Approach | Description | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| **Word2vec-style** | Treat numbers as words | Simple | No numeric structure |
| **NALU** (Trask, 2018) | Learned gates for +, -, *, / | End-to-end differentiable | Training instability |
| **xVal** (Golkar, 2023) | Learned scaling direction | Better interpolation | Still requires training |
| **NumBed** | Learned position-magnitude | Smooth interpolation | OOD degradation |

**Limitation**: Learned embeddings inherit training distribution bias.

### 3.3 Semantic Embeddings

General-purpose text embeddings applied to numbers.

| Approach | Description | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| **Sentence-Transformers** | Semantic similarity embeddings | Great for text | Numbers are arbitrary symbols |
| **OpenAI Embeddings** | Large-scale semantic embeddings | General purpose | No arithmetic structure |
| **BERT embeddings** | Contextual token embeddings | Rich representations | Arithmetic is implicit |

**Limitation**: Semantic embeddings optimize for meaning similarity, not algebraic structure.

### 3.4 Algebraic Embeddings (FluxEM)

Embeddings designed to preserve algebraic operations.

| Approach | Description | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| **Linear encoding** | `embed(a) + embed(b) = embed(a+b)` | Exact addition | Single operation |
| **Log encoding** | `embed(a) + embed(b) = embed(a*b)` | Exact multiplication | Requires log transform |
| **Unified encoding** | Combined linear + log | All basic operations | Slightly larger embedding |
| **Domain encoding** | Units, molecules, etc. | Domain-specific algebra | Domain-specific |

**Advantage**: Correctness is guaranteed by construction, not learned.

---

## 4. Experimental Results

### 4.1 Arithmetic Benchmark

Training: 10K expressions, integers in [0, 999], 2-3 operands, operations {+, -, *}

| Method | ID Accuracy | OOD-A (Large Ints) | OOD-B (Long Chains) | OOD-C (Powers) | Parameters | Training |
|--------|-------------|-------------------|---------------------|----------------|------------|----------|
| **FluxEM** | 100% | 100% | 100% | 100% | 0 | None |
| Char-level Transformer | 2% | 0% | 0.5% | 2% | ~500K | 50 epochs |
| Char-level GRU | 0% | 0.5% | 0.5% | 0.5% | ~200K | 50 epochs |
| BPE Transformer | ~85% | ~15% | ~20% | ~10% | ~1M | 50 epochs |

*Accuracy = result within 1% relative error of ground truth.*

### 4.2 Dimensional Analysis (Units)

Training: 5K unit conversion/compatibility examples

| Method | ID Accuracy | OOD-Magnitude | OOD-Units | Dimensional Correctness |
|--------|-------------|---------------|-----------|------------------------|
| **FluxEM (DimensionalQuantity)** | 100% | 100% | 100% | 100% |
| Token-only Transformer | ~80% | ~30% | ~25% | ~70% |

### 4.3 Expected Hybrid Training Results

When integrating FluxEM into LLM training pipelines:

| Configuration | Arithmetic Accuracy | Sample Efficiency | OOD Generalization |
|---------------|---------------------|-------------------|-------------------|
| Token-only baseline | ~85% ID, ~20% OOD | 100% (baseline) | Poor |
| Hybrid (FluxEM frozen) | ~99% ID, ~98% OOD | ~10x improvement | Strong |
| Hybrid (FluxEM + learned) | ~99% ID, ~99% OOD | ~5x improvement | Strong |

*Note: Hybrid training results are projected based on architecture analysis. See [EXPERIMENTS.md](EXPERIMENTS.md) for runnable experiments.*

---

## 5. Why This Matters

### 5.1 Hybrid Training: Best of Both Worlds

FluxEM enables a hybrid training paradigm:

```
Text tokens     -> Standard tokenizer -> Learned embeddings
Domain objects  -> FluxEM encoders   -> Deterministic embeddings
Combined stream -> Transformer       -> Unified attention
```

**Key insight**: The transformer learns *routing and composition*, while exact algebra is carried by FluxEM representations.

### 5.2 Sample Efficiency

Models don't need to learn arithmetic from scratch:

| Approach | Samples to Learn `a + b` | Samples to Learn `a * b` |
|----------|--------------------------|--------------------------|
| Token-only | ~100K | ~500K |
| FluxEM hybrid | 0 (by construction) | 0 (by construction) |

The saved capacity can be used for reasoning, language understanding, and domain-specific knowledge.

### 5.3 Scientific Computing Applications

| Domain | FluxEM Encoder | Exact Operations |
|--------|----------------|------------------|
| Physics | `DimensionalQuantity` | Unit conversion, dimensional analysis |
| Chemistry | `MoleculeEncoder` | Formula parsing, stoichiometry |
| Biology | `DNAEncoder` | Sequence operations, complement |
| Music | `PitchEncoder` | Transposition, pitch class sets |
| Math | `ComplexEncoder`, `MatrixEncoder` | Complex arithmetic, linear algebra |

### 5.4 Cleaner Failure Modes

When parsing fails, the system falls back explicitly:

```python
# Ambiguous input
"The temperature is about 300"  # 300 what? Fallback to text token

# Clearly structured input
"The temperature is 300 K"      # FluxEM: DimensionalQuantity(300, K)
```

No silent "hallucinating math" - the model knows when it's operating in text-only mode.

---

## 6. Limitations and Trade-offs

### What FluxEM Does Well

- Exact arithmetic on any magnitude
- Perfect OOD generalization for supported operations
- Zero training required
- Deterministic, reproducible results

### What FluxEM Does Not Do

- **Approximate reasoning**: "About 100" requires learned semantics
- **Novel operations**: Only supported algebraic operations work
- **Fuzzy matching**: "3.14" vs "pi" requires learned association
- **Error correction**: Typos in numbers are not auto-corrected

### Hybrid Approach Addresses Limitations

By combining FluxEM with learned representations:
- Text embeddings handle fuzzy/semantic aspects
- FluxEM handles exact computation
- Attention learns when to route to which representation

---

## 7. Summary

| Aspect | Token Embeddings | Learned Numeric | FluxEM |
|--------|------------------|-----------------|--------|
| **Arithmetic correctness** | Learned | Learned | Exact |
| **OOD generalization** | Poor | Moderate | Perfect |
| **Training required** | Yes | Yes | No |
| **Parameters** | Millions | Thousands | Zero |
| **Integration complexity** | Low | Medium | Medium |
| **Best for** | Language | Interpolation | Computation |

**Bottom line**: FluxEM provides a principled alternative to learned numeric representations when exact computation matters. Combined with standard tokenization in a hybrid architecture, it offers the best of both worlds: learned language understanding with guaranteed arithmetic correctness.

---

## References

- [NALU: Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) (Trask et al., 2018)
- [xVal: Continuous Number Encoding](https://arxiv.org/abs/2310.02989) (Golkar et al., 2023)
- [Abacus Embeddings](https://arxiv.org/abs/2405.17399) (McLeish et al., 2024)
- [FluxEM README](../README.md) - Project overview
- [HYBRID_TRAINING.md](HYBRID_TRAINING.md) - Integration architecture
- [EXPERIMENTS.md](EXPERIMENTS.md) - Runnable experiments
