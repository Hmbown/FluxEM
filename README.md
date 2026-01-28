# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![PyPI version](https://img.shields.io/pypi/v/fluxem-tools)](https://pypi.org/project/fluxem-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![CI](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Deterministic tools for what LLMs get wrong.**

LLMs guess. Algebra computes.

The key idea: don't teach LLMs to compute. Let them decide when to delegate.

> [Project page](https://hmbown.github.io/FluxEM) · [The Vision](https://hmbown.github.io/FluxEM/vision.html)

---

## Overview

FluxEM is an umbrella project with two complementary deliverables:

- `fluxem-tools`: **510+ deterministic tools** across **42 domains** for LLM tool/function calling (recommended for production)
- `fluxem`: algebraic encoders where selected computations become vector operations (research / embedding injection)

Example (internal evals; see `experiments/qwen3_toolcalling/`):

| Query | Base Qwen3-4B | FluxEM tool route | Correct |
|-------|---------------|-------------------|---------|
| 789123 x 456789 | 360,789,123,447 | Tool call -> 360,427,326,447 | 360,427,326,447 |
| 13! | 156 | Tool call -> 6,227,020,800 | 6,227,020,800 |
| GCD(4851, 3003) | 3003 | Tool call -> 231 | 231 |

---

## Tool Domains (510+ Tools, 42 Domains)

FluxEM Tools provides comprehensive coverage across scientific, engineering, and everyday domains:

| Domain | Tools | Description |
|--------|-------|-------------|
| **text** | 23 | String manipulation, parsing, formatting |
| **electrical** | 19 | Ohm's law, circuits, power calculations |
| **gardening** | 18 | Planting schedules, soil, spacing |
| **travel** | 16 | Distance, time zones, fuel costs |
| **optics** | 16 | Lenses, diffraction, focal length |
| **fitness** | 16 | BMI, calories, heart rate zones |
| **nuclear** | 15 | Decay, half-life, radiation |
| **currency** | 15 | Exchange rates, conversions |
| **astronomy** | 15 | Orbital mechanics, magnitudes, distances |
| **thermodynamics** | 14 | Heat transfer, entropy, cycles |
| **pharmacology** | 14 | Dosing, half-life, concentrations |
| **fluid_dynamics** | 14 | Flow rates, Reynolds number, pressure |
| **diy** | 14 | Materials, measurements, costs |
| **acoustics** | 14 | Sound levels, frequency, wavelength |
| **temporal** | 13 | Date math, time zones, durations |
| **number_theory** | 13 | Primes, GCD, modular arithmetic |
| **graphs** | 13 | Paths, connectivity, cycles |
| **sets** | 12 | Union, intersection, difference |
| **security** | 12 | Hashing, permissions, access control |
| **photography** | 12 | Exposure, aperture, depth of field |
| **finance** | 12 | NPV, compound interest, loans |
| **data** | 12 | Arrays, sorting, aggregation |
| **cooking** | 12 | Unit conversion, scaling recipes |
| **color** | 12 | RGB, hex, color mixing |
| **statistics** | 11 | Mean, variance, correlation |
| **signal_processing** | 11 | FFT, convolution, filtering |
| **probability** | 11 | Distributions, Bayes, combinatorics |
| **math_advanced** | 11 | Special functions, series |
| **geometric_algebra** | 11 | Clifford algebra, rotors |
| **combinatorics** | 11 | Permutations, combinations |
| **optimization** | 10 | Gradient descent, linear programming |
| **geometry** | 10 | Areas, distances, transformations |
| **music** | 9 | Intervals, chords, transposition |
| **logic** | 9 | Boolean operations, tautologies |
| **geospatial** | 9 | Haversine, bearing, coordinates |
| **calculus** | 9 | Derivatives, integrals, polynomials |
| **physics** | 8 | Kinematics, forces, energy |
| **information_theory** | 8 | Entropy, mutual information |
| **control_systems** | 8 | Stability, transfer functions |
| **chemistry** | 8 | Molecular weight, stoichiometry |
| **biology** | 8 | DNA/RNA, GC content, translation |
| **arithmetic** | 2 | Basic operations with precision |

---

## Quick Start (Tool Calling)

```bash
pip install fluxem-tools
```

Direct (no LLM):

```python
from fluxem_tools import call_tool

call_tool("arithmetic", expr="12345 + 67890")  # 80235
call_tool("electrical_ohms_law", voltage=12, current=2)  # 6.0
call_tool("astronomy_parallax_distance", parallax_arcsec=0.1)  # 32.6 light years
call_tool("pharmacology_half_life_remaining", initial=500, half_life=4, time=12)  # 62.5
```

OpenAI function calling (minimal tool loop):

```python
import json
from openai import OpenAI

from fluxem_tools import call_tool, get_registry

client = OpenAI()
tools = get_registry().to_openai_tools()

messages = [{"role": "user", "content": "What is 789123 * 456789?"}]
resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)

msg = resp.choices[0].message
if msg.tool_calls:
    messages.append(msg)
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        result = call_tool(tc.function.name, **args)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)

print(resp.choices[0].message.content)
```

More integrations and export formats in `fluxem-tools-pkg/README.md`.

---

## RL Training with Prime Intellect Lab

FluxEM provides training environments for reinforcement learning on tool-calling tasks using [Prime Intellect Lab](https://docs.primeintellect.ai/).

**Goal**: Teach models WHEN to delegate computations to deterministic tools instead of guessing.

### Current Status (Jan 28, 2026)

- Mastery-phase RL environment is packaged with a clean `src/` layout and verified to install on both local and Prime workers.
- Retrieval and domain routing are enabled with a 210-tool subset and a 1000-problem dataset for the mastery track.
- Training is running on Prime RL (Qwen/Qwen3-4B-Instruct-2507; 300 steps; batch 64; 8 rollouts per example).
- Lab configs now use `[env.args]` tables for per-phase settings.

### Available Environments

| Environment | Type | Problems | Tools | Description |
|-------------|------|----------|-------|-------------|
| `hunterbown/fluxem-tools-env` | MultiTurnEnv | 51 | 210 | **Text-based tool calling** - mix of easy/hard problems |
| `hunterbown/fluxem-wordproblems` | ToolEnv | 400 | 510+ | Realistic word problems (requires vLLM tool-choice flags) |

> **Note**: The `fluxem-tools-env` bundles a curated subset of 210 tools from the full 510+ tool suite. This subset covers the core math/number-theory tools needed for the training problems.

### Training Configuration

```toml
# configs/lab/fluxem_tools.toml
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 500
batch_size = 64
rollouts_per_example = 8

[sampling]
max_tokens = 512  # Room for tool call + result + answer

[[env]]
id = "hunterbown/fluxem-tools-env"

[val]
num_examples = 50
rollouts_per_example = 4
interval = 25
```

### Running Training

```bash
# Install Prime CLI
uv tool install prime
prime login

# Install environment locally first
prime env install fluxem-tools-env --path ./environments/fluxem_tools_env

# Test with local eval
prime eval run fluxem-tools-env -m openai/gpt-4o-mini -n 10

# Push to Hub
prime env push --path environments/fluxem_tools_env

# Start training
prime rl run configs/lab/fluxem_tools.toml

# Monitor progress
prime rl logs <run_id> | grep "SUCCESS Step"
prime rl distributions <run_id>
prime rl rollouts <run_id> --step 50 --limit 20
```

### How It Works

The environment uses **XMLParser + MultiTurnEnv** instead of native `ToolEnv` to work without vLLM's `--enable-auto-tool-choice` flag:

```xml
<!-- Model outputs tool call as text -->
<tool_call>
<name>arithmetic</name>
<args>{"expr": "12345 * 6789"}</args>
</tool_call>

<!-- Environment executes and returns result -->
Tool result: 83810205

<!-- Model provides final answer -->
<answer>83810205</answer>
```

The model learns:
- Easy problems (5+7) → answer directly (faster)
- Hard problems (12345×6789) → use tool (guaranteed correct)

### Supported Models

- `Qwen/Qwen3-4B-Instruct-2507`
- `Qwen/Qwen3-4B-Thinking-2507`
- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `Qwen/Qwen3-30B-A3B-Thinking-2507`
- `Qwen/Qwen3-235B-A22B-Instruct-2507`

---

## MCP Server (Optional)

If you prefer a "tool server" boundary (so a client/agent can call FluxEM over MCP), FluxEM Tools ships a small MCP surface:

- `fluxem_search`: find a tool by keyword
- `fluxem_call`: execute a tool by name with JSON arguments

Install + run:

```bash
pip install "fluxem-tools[mcp]"
fluxem-tools-mcp
```

Claude Desktop config snippet:

```json
{
  "mcpServers": {
    "fluxem-tools": {
      "command": "fluxem-tools-mcp"
    }
  }
}
```

---

## When To Use FluxEM

**Use FluxEM-style deterministic tooling when you need correctness:**
- Arithmetic / number theory / statistics
- Units + dimensional analysis (physics/engineering)
- Chemistry (stoichiometry / molecular weights)
- Biology (DNA/RNA analysis, GC content)
- Finance (NPV, compound interest, loan payments)
- Geospatial (distances, bearings, coordinates)
- Music theory (transposition, chord analysis)
- Electrical (circuits, power, resistance)
- And 30+ more specialized domains...

**Don't use FluxEM for:**
- Symbolic math (e.g. `x + x -> 2x`) — use SymPy
- Semantic similarity / meaning — use text embeddings

---

## Three Modes

1. **Tool calling (primary)**: `pip install fluxem-tools` and export tool schemas to your LLM runtime
2. **Direct API (scripting)**: call tools locally via `fluxem_tools.call_tool(...)` with no LLM involved
3. **Embedding injection (research)**: use `fluxem` encoders/operators and the research code in `experiments/`

---

## Technical Details (Embeddings)

<details>
<summary><strong>FluxEM embeddings: original construction</strong></summary>

No training. No weights. Structure by construction.

```
encode(a) + encode(b) = encode(a + b)
```

<p align="center">
  <img src="docs/demo.gif" alt="FluxEM demo" width="600">
</p>

Install:

```bash
pip install fluxem
```

Minimal compute example:

```python
from fluxem import create_unified_model

model = create_unified_model()
print(model.compute("12345 + 67890"))  # 80235.0
print(model.compute("144 * 89"))       # 12816.0
```

Physics (dimensional quantities):

```python
from fluxem.domains.physics import DimensionalQuantity, Dimensions

dq = DimensionalQuantity()
velocity = dq.encode(10, Dimensions(L=1, T=-1))  # 10 m/s
duration = dq.encode(5, Dimensions(T=1))        # 5 s
distance = dq.multiply(velocity, duration)
print(dq.decode(distance))  # (50.0, [L])
```

Formal notes: [ERROR_MODEL.md](docs/ERROR_MODEL.md) and [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md).

</details>

<details>
<summary><strong>Embedding domains (fluxem package)</strong></summary>

FluxEM (the `fluxem` package) encodes typed domain values into fixed-dimensional vectors where selected operations map to linear algebra:

- Addition/subtraction: `encode(a) + encode(b) = encode(a + b)`
- Multiplication/division: `log_encode(a) + log_encode(b) = log_encode(a * b)`

| Domain | Example | Operations |
|--------|---------|------------|
| Physics | `9.8 m/s²` | Unit conversion, dimensional analysis |
| Chemistry | `C6H12O6` | Stoichiometry, mass balance |
| Biology | `ATGCCGTAG` | GC content, melting temp, translation |
| Math | `3 + 4i` | Complex numbers, matrices, vectors, polynomials |
| Logic | `p ∧ q → r` | Tautology detection, satisfiability |
| Music | `{0, 4, 7}` | Transposition, inversion, Forte numbers |
| Geometry | △ABC | Area, centroid, circumcenter |
| Graphs | G = (V, E) | Connectivity, cycles, shortest path |
| Sets | A ∪ B | Union, intersection, composition |
| Number Theory | 360 = 2³·3²·5 | Prime factorization, modular arithmetic |
| Data | [x₁, x₂, ...] | Arrays, records, tables |

</details>

---

## Research & Evaluation

- Tool-calling orchestrator + internal benchmarks: `experiments/qwen3_toolcalling/README.md`
- Broader research notes: `docs/RESEARCH_2026.md`
- RL training environments: `environments/`
- Related: [NVIDIA ToolOrchestra](https://research.nvidia.com/labs/lpr/ToolOrchestra/), [DeepSeek Engram](https://github.com/deepseek-ai/Engram)

---

## Reproducibility (From Source)

```bash
git clone https://github.com/Hmbown/FluxEM.git && cd FluxEM
pip install -e ".[jax]"
python experiments/scripts/compare_embeddings.py
```

Outputs TSV tables comparing FluxEM to baselines:

```
table=accuracy_by_encoder
approach              dataset          exact_match    numeric_accuracy
FluxEM                id               1.000000       1.000000
FluxEM                ood_magnitude    1.000000       1.000000
Character             ood_magnitude    0.000000       0.012000
```

### Benchmarks

We provide internal benchmarks to validate the workflow and surface failure modes. Key artifacts:
- Method + usage: `experiments/qwen3_toolcalling/README.md`
- Reports + plots: `experiments/qwen3_toolcalling/results/`
- Raw logs: `experiments/qwen3_toolcalling/results/benchmark_results_*.json`

```bash
python experiments/qwen3_toolcalling/run_benchmark.py \
  --quick-test \
  --save-results
```

---

## Installation

```bash
pip install fluxem-tools         # Deterministic tool suite (recommended)
pip install fluxem               # Core embeddings (NumPy)
pip install fluxem[jax]          # With JAX
pip install fluxem[mlx]          # With MLX (Apple Silicon)
pip install fluxem[full-jax]     # Full with HuggingFace
```

---

## Precision (Embeddings)

| Operation | Relative Error (float32) |
|-----------|-------------------------|
| Add/Sub   | < 1e-7 |
| Mul/Div   | < 1e-6 |

Edge cases: `log(0)` -> masked, division by zero -> signed infinity, negative base with fractional exponent -> unsupported.

---

## Related Work

| Approach | Method | Difference |
|----------|--------|------------|
| [NALU](https://arxiv.org/abs/1808.00508) | Learned log/exp gates | FluxEM: no learned parameters |
| [xVal](https://arxiv.org/abs/2310.02989) | Learned scaling | FluxEM: deterministic, multi-domain |
| [Abacus](https://arxiv.org/abs/2405.17399) | Positional digits | FluxEM: algebraic structure |

---

## Citation

```bibtex
@software{fluxem2026,
  title={FluxEM: Deterministic Tools and Algebraic Embeddings for LLM Delegation},
  author={Bown, Hunter},
  year={2026},
  url={https://github.com/Hmbown/FluxEM}
}
```

## License

MIT
