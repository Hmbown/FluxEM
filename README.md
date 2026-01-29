# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![PyPI version](https://img.shields.io/pypi/v/fluxem-tools)](https://pypi.org/project/fluxem-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![CI](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Deterministic tools for what LLMs get wrong.**

LLMs guess. Algebra computes.

The key idea: do not teach models to compute. Teach them when to delegate.

> Project page: https://hmbown.github.io/FluxEM
> Vision: https://hmbown.github.io/FluxEM/vision.html

---

## Overview

FluxEM is an umbrella project with two complementary deliverables:

- `fluxem-tools`: 510+ deterministic tools across 42 domains for tool/function calling (production)
- `fluxem`: algebraic encoders where selected computations become vector operations (research)

Example (internal evals):

| Query | Base Qwen3-4B | FluxEM tool route | Correct |
|-------|---------------|-------------------|---------|
| 789123 x 456789 | 360,789,123,447 | Tool call -> 360,427,326,447 | 360,427,326,447 |
| 13! | 156 | Tool call -> 6,227,020,800 | 6,227,020,800 |
| GCD(4851, 3003) | 3003 | Tool call -> 231 | 231 |

---

## Method: Tool-First Scaling (Production) + Algebraic Embeddings (Research)

FluxEM pursues two complementary approaches:

- **Tool-first training (production)**: keep the base model architecture unchanged and teach it to delegate. We embed
  tool descriptions and tasks, retrieve a small tool set via FAISS + domain routing, and train with RL so the model
  chooses correct tools and uses them efficiently. This is an *external* sparse capacity knob.
- **Algebraic embeddings (research)**: encode domain values so valid operations become linear algebra. This is a
  separate track focused on structure-by-construction rather than tool calling.

This is related (but not identical) to embedding-scaling papers: they expand *internal* model embeddings to add
capacity; we expand *external* tool embeddings to improve selection and delegation without changing the base model.

---

## Repository Layout

- `fluxem-tools-pkg/` - deterministic tool library (packaged as `fluxem-tools`)
- `fluxem/` - algebraic encoders and research code
- `environments/` - Prime RL training environments and configs
- `experiments/` - benchmarks, training scripts, and eval harnesses
- `docs/` - research notes and background

---

## Quick Start (Tool Calling)

```bash
pip install fluxem-tools
```

Direct calls (no LLM):

```python
from fluxem_tools import call_tool

call_tool("arithmetic", expr="12345 + 67890")
call_tool("electrical_ohms_law", voltage=12, current=2)
call_tool("astronomy_parallax_distance", parallax_arcsec=0.1)
call_tool("pharmacology_half_life_remaining", initial=500, half_life=4, time=12)
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

More integrations and export formats live in `fluxem-tools-pkg/README.md`.

---

## RL Training with Prime Intellect Lab

FluxEM provides training environments for reinforcement learning on tool-calling tasks using Prime Intellect Lab.

Goal: teach models when to delegate computations to deterministic tools instead of guessing.

### Current Status (Jan 28, 2026)

- Mastery-phase RL environment packaged with a clean `src/` layout and verified to install locally and on Prime workers.
- Retrieval + domain routing enabled over a curated 210-tool subset and a 1000-problem mastery dataset.
- Active training on Prime RL (Qwen/Qwen3-4B-Instruct-2507; 300 steps; batch 64; 8 rollouts/example).
- Lab configs standardized on `[env.args]` tables for per-phase settings.

### Available Environments

| Environment | Type | Problems | Tools | Notes |
|-------------|------|----------|-------|------|
| `hunterbown/fluxem-tools-env` | MultiTurnEnv | 51 | 210 | Text-based tool calling, mixed difficulty |
| `hunterbown/fluxem-wordproblems` | ToolEnv | 400 | 510+ | Word problems, requires vLLM tool-choice flags |

### Training Configuration

```toml
# configs/lab/fluxem_tools.toml
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 500
batch_size = 64
rollouts_per_example = 8

[sampling]
max_tokens = 512

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
prime rl logs <run_id> -f
prime rl distributions <run_id>
prime rl rollouts <run_id> --step 50 --limit 20
```

---

## Tool Domains

FluxEM Tools covers 42 domains across science, engineering, and everyday tasks. Full registry details are in `fluxem-tools-pkg/README.md`.

Examples:
- arithmetic, number_theory, statistics, probability
- physics, chemistry, biology, thermodynamics
- electrical, optics, signal_processing
- finance, currency, temporal, geospatial
- text, data, graphs, logic

---

## MCP Server (Optional)

FluxEM Tools ships a small MCP surface for tool search + execution:

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

## Research Mode (fluxem package)

FluxEM (the `fluxem` package) encodes typed domain values into fixed-dimensional vectors where selected operations map to linear algebra.

Install:

```bash
pip install fluxem
```

Minimal compute example:

```python
from fluxem import create_unified_model

model = create_unified_model()
print(model.compute("12345 + 67890"))
print(model.compute("144 * 89"))
```

Formal notes: `docs/ERROR_MODEL.md` and `docs/FORMAL_DEFINITION.md`.

---

## Related Work and Convergences

We cite related research for context and note convergent directions; this is not a claim of inspiration unless the
work predates FluxEM's first commit (Jan 28, 2026).

- "Scaling Embeddings Outperforms Scaling Experts in Language Models" (LongCat team, 2025). Predates FluxEM. We do not
  modify the base model architecture or add N-gram embeddings; instead we use tool embeddings for retrieval and routing.
  The paper's analysis of embedding scaling vs expert scaling informs how we budget retrieved tools and when retrieval
  helps most.
- Conditional memory / hashed lookup modules (Engram-style) are conceptually adjacent to FluxEM's deterministic tool
  lookup and algebraic encoders; we reference them as convergent lines of work, not as direct inspiration.
- Additional related research notes live in `docs/RESEARCH_2026.md`, with project-specific analysis in `docs/tech_report.md`.

---

## Reproducibility

```bash
git clone https://github.com/Hmbown/FluxEM.git && cd FluxEM
pip install -e ".[jax]"
python experiments/scripts/compare_embeddings.py
```

---

## License

MIT
