# FluxEM Multi-Domain Tool-Calling Model Training Recipe

This document provides a complete recipe for training a Qwen3-4B model with FluxEM domain embeddings for multi-turn tool-calling across 22 scientific and mathematical domains.

## Overview

- **Base Model:** Qwen3-4B-Instruct
- **Total Domains:** 22 (12 original + 10 new expanded)
- **Training Method:** LoRA fine-tuning with FluxEM embedding integration
- **Target Use Case:** Multi-turn tool-calling for domain-specific computations

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU/Accelerator | 16GB VRAM | 24GB+ VRAM |
| RAM | 32GB | 64GB |
| Storage | 50GB | 100GB |
| Estimated Training Time | 36-48h (M2 Ultra) | 24-36h (A100 40GB) |

### Software Dependencies

```bash
# Core dependencies
pip install torch>=2.1.0
pip install transformers>=4.36.0
pip install peft>=0.7.0  # LoRA support
pip install accelerate>=0.25.0

# FluxEM
pip install -e .  # from FluxEM root

# Optional: MLX backend for Apple Silicon
pip install mlx>=0.5.0
```

### Model Download

```bash
# Download base model
huggingface-cli download Qwen/Qwen3-4B-Instruct --local-dir ~/.cache/huggingface/hub/Qwen3-4B-Instruct

# For MLX backend
python -c "from mlx_lm import convert; convert('Qwen/Qwen3-4B-Instruct', '~/.mlx/models/Qwen/Qwen3-4B-Instruct-MLX')"
```

## Domain Coverage

### Original Domains (12)

| Domain | Encoder | Sample Tools |
|--------|---------|--------------|
| arithmetic | ArithmeticEncoder | add, multiply, divide |
| physics | PhysicsEncoder | unit_convert, quantity_calc |
| chemistry | ChemistryEncoder | molar_mass, balance_equation |
| biology | BiologyEncoder | dna_complement, protein_translate |
| math | MathEncoder | complex_add, matrix_multiply |
| geometry | GeometryEncoder | area, perimeter |
| graphs | GraphEncoder | shortest_path, connected |
| sets | SetEncoder | union, intersection |
| logic | LogicEncoder | evaluate_prop, truth_table |
| number_theory | NumberTheoryEncoder | gcd, prime_factors |
| music | MusicEncoder | transpose, chord_analysis |
| data | DataEncoder | sort, filter |

### New Expanded Domains (10)

| Domain | Encoder | Tools | Description |
|--------|---------|-------|-------------|
| combinatorics | CombinatoricsEncoder | factorial, ncr, npr | Combinatorial counting |
| probability | ProbabilityEncoder | bernoulli_pmf, binomial_pmf, bayes | Probability distributions |
| statistics | StatisticsEncoder | mean, variance, correlation | Statistical analysis |
| information_theory | InformationTheoryEncoder | entropy, kl_divergence, mutual_info | Information measures |
| signal_processing | SignalEncoder | convolution, dft_magnitude, autocorrelation | Signal analysis |
| calculus | CalculusPolynomialEncoder | derivative, integral, evaluate | Polynomial calculus |
| temporal | TemporalEncoder | add_days, days_between, weekday | Date arithmetic |
| finance | FinanceEncoder | npv, irr, compound_interest | Financial calculations |
| optimization | OptimizationEncoder | gradient_step, least_squares, line_search | Optimization methods |
| control_systems | ControlSystemEncoder | is_stable, step_response, controllability | Control theory |

## Data Preparation

### Step 1: Generate Training Data

```bash
# Generate multi-domain tool-calling data
python experiments/scripts/generate_toolcalling_data.py \
    --config experiments/configs/toolcalling_multidomain.yaml \
    --output-dir experiments/data/toolcalling_multidomain \
    --train-size 50000 \
    --val-size 5000 \
    --test-size 5000
```

### Step 2: Validate Data

```bash
# Validate JSONL format
python -m fluxem.integration.sample_format \
    experiments/data/toolcalling_multidomain/train.jsonl \
    experiments/data/toolcalling_multidomain/val.jsonl \
    --check-encoding
```

### Data Format

Each sample follows this JSONL format:

```json
{
  "text": "What is the factorial of 5?",
  "spans": [
    {"type": "comb_term", "start": 28, "end": 29, "value": {"kind": "factorial", "n": 5}}
  ],
  "target_text": "<tool_call>combinatorics_factorial</tool_call><args>5</args><result>120</result>",
  "target_value": 120
}
```

### Domain Distribution

The training config specifies weighted sampling:

```yaml
domain_distribution:
  # Core domains (higher weight)
  arithmetic: 0.10
  physics: 0.08
  math: 0.08

  # New domains (focus)
  combinatorics: 0.06
  probability: 0.06
  statistics: 0.06
  calculus: 0.05
  finance: 0.05
  temporal: 0.04
  information_theory: 0.04
  signal_processing: 0.04
  optimization: 0.04
  control_systems: 0.03

  # ... (remaining domains)
```

## Training

### Step 3: Run Training

```bash
# Standard training with LoRA
python experiments/scripts/train_hybrid_fixed.py \
    --config experiments/configs/toolcalling_multidomain.yaml \
    --output-dir experiments/checkpoints/toolcalling_multidomain \
    --epochs 10 \
    --batch-size 8 \
    --gradient-accumulation-steps 4

# Or with MLX backend (Apple Silicon)
python experiments/scripts/train_hybrid_fixed.py \
    --config experiments/configs/toolcalling_multidomain.yaml \
    --backend mlx \
    --output-dir experiments/checkpoints/toolcalling_multidomain
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 2e-5 | Cosine decay with warmup |
| Warmup Steps | 500 | ~1% of total steps |
| Weight Decay | 0.01 | Standard regularization |
| Batch Size | 8 | Per device |
| Gradient Accumulation | 4 | Effective batch = 32 |
| Epochs | 10 | ~50,000 steps |
| Max Seq Length | 2048 | Fits most tool chains |
| LoRA Rank | 16 | Balance efficiency/quality |
| LoRA Alpha | 32 | Standard 2x rank |
| LoRA Dropout | 0.05 | Light regularization |

### LoRA Target Modules

```yaml
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### Auxiliary Losses

| Loss | Weight | Purpose |
|------|--------|---------|
| Tool Selection | 0.5 | Cross-entropy on tool choice |
| Boundary Tagging | 0.3 | Identify domain span boundaries |
| Operation Conditioned | 0.2 | Predict operation from operands |
| Invariance | 0.1 | Equivalent expressions → similar embeddings |
| Projection Regularization | 0.01 | L2 on projection weights |

## Evaluation

### Step 4: Run Benchmark

```bash
# Run evaluation on test set
python experiments/scripts/eval.py \
    --checkpoint experiments/checkpoints/toolcalling_multidomain/best \
    --test-data experiments/data/toolcalling_multidomain/test.jsonl \
    --output experiments/results/toolcalling_multidomain/eval_results.json
```

### Step 5: Run Acceptance Tests

```bash
# Unit tests for new domain encoders
python -m pytest tests/test_domain_encoders_new_domains.py -v

# Tool calling smoke tests
python -m pytest tests/test_toolcalling_new_domains.py -v

# Full acceptance test suite
python -m pytest tests/test_acceptance_new_domains.py -v
```

### Pass/Fail Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| Overall Tool Accuracy | >= 85% | >= 95% |
| Per-Domain Accuracy | >= 70% | >= 90% |
| Domain Detection Accuracy | >= 90% | >= 98% |
| Tool Execution Success | >= 95% | 100% |
| Argument Extraction F1 | >= 80% | >= 95% |
| Chain Completion Rate | >= 75% | >= 90% |

### Benchmark Prompts

The benchmark includes 30+ prompts across all 10 new domains:

```python
# Sample prompts
"What is 5 factorial?"  # combinatorics -> 120
"Calculate C(10, 3)"  # combinatorics -> 120
"Binomial PMF for n=10 k=5 p=0.5?"  # probability -> 0.246
"Mean of [2, 4, 6, 8, 10]?"  # statistics -> 6.0
"Shannon entropy of [0.5, 0.5]?"  # information_theory -> 1.0
"Derivative of [1, 2, 3]?"  # calculus -> [2, 6]
"30 days after 2024-01-01?"  # temporal -> "2024-01-31"
"NPV at 10% for [-100, 50, 50, 50]?"  # finance -> 24.34
```

## Model Export

### Step 6: Export Trained Model

```bash
# Merge LoRA weights and export
python experiments/scripts/export_model.py \
    --checkpoint experiments/checkpoints/toolcalling_multidomain/best \
    --output-dir experiments/models/fluxem-qwen3-4b-toolcalling \
    --merge-lora

# Upload to HuggingFace (optional)
huggingface-cli upload your-org/fluxem-qwen3-4b-toolcalling \
    experiments/models/fluxem-qwen3-4b-toolcalling
```

## Inference

### Using the Trained Model

```python
from experiments.qwen3_toolcalling.qwen3_wrapper import Qwen3MLXWrapper

# Load model
wrapper = Qwen3MLXWrapper(
    model_path="experiments/models/fluxem-qwen3-4b-toolcalling",
    tool_selection="pattern",
    response_style="plain"
)

# Single tool call
result = wrapper.process_query("What is the factorial of 7?")
print(result)  # 5040

# Multi-turn conversation
context = wrapper.create_context()
context = wrapper.process_turn(context, "Calculate NPV at 8% for cashflows [-1000, 400, 400, 400]")
print(context.get_last_result())  # ~30.83
```

## Troubleshooting

### Common Issues

1. **Out of Memory:** Reduce batch size or use gradient checkpointing
2. **Slow Training:** Enable mixed precision (`--mixed-precision bf16`)
3. **Poor Convergence:** Check learning rate, try warmup_steps=1000
4. **Tool Selection Errors:** Verify tool registry has all expected tools

### Validation Commands

```bash
# Verify tool registry
python -c "from experiments.qwen3_toolcalling.tool_registry import create_tool_registry; print(len(create_tool_registry()))"
# Expected: 60+ tools

# Verify encoder registry
python -c "from fluxem.integration.pipeline import DomainEncoderRegistry; r = DomainEncoderRegistry(); print(len([e for e in dir(r) if not e.startswith('_')]))"

# Verify DomainType enum
python -c "from fluxem.integration.tokenizer import DomainType; print(max(d.value for d in DomainType))"
# Expected: 35 (CONTROL)
```

## Files Reference

| File | Purpose |
|------|---------|
| `experiments/configs/toolcalling_multidomain.yaml` | Training configuration |
| `experiments/scripts/generate_toolcalling_data.py` | Data generation |
| `experiments/scripts/train_hybrid_fixed.py` | Training script |
| `experiments/scripts/eval.py` | Evaluation script |
| `experiments/qwen3_toolcalling/tool_registry.py` | Tool definitions |
| `experiments/qwen3_toolcalling/benchmark_data.py` | Benchmark prompts |
| `tests/test_acceptance_new_domains.py` | Acceptance tests |
| `fluxem/integration/tokenizer.py` | DomainType definitions |
| `fluxem/integration/pipeline.py` | Encoder registry |
| `fluxem/domains/*/` | Domain encoder implementations |

## Changelog

- **v1.0.0:** Initial 22-domain training recipe with 10 new expanded domains
