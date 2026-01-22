# FluxEM Tools

**210+ deterministic computation tools for LLM tool-calling.**

This is a **tool package**, not a fine-tuned model. Use with any capable LLM (GPT-4, Claude, Qwen, Llama, Gemini, etc.) that supports function/tool calling.

## Installation

```bash
pip install fluxem-tools
```

## Quick Start

```python
from fluxem_tools import get_registry, call_tool

# Get the tool registry
registry = get_registry()
print(f"Total tools: {len(registry)}")  # 210+

# Call a tool directly
result = call_tool("arithmetic", "2 + 3 * 4")
print(result)  # 14

# Physics calculation
ohms = call_tool("electrical_ohms_law", {"voltage": 12, "current": 2})
print(f"Resistance: {ohms} ohms")  # 6.0

# List available domains
from fluxem_tools import list_domains
print(list_domains())  # ['arithmetic', 'physics', 'chemistry', ...]
```

## LLM Integration

### OpenAI

```python
from openai import OpenAI
from fluxem_tools import get_registry

client = OpenAI()
tools = get_registry().to_openai_tools()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is 23 * 47?"}],
    tools=tools
)
```

### Anthropic Claude

```python
import anthropic
from fluxem_tools import get_registry

client = anthropic.Anthropic()
tools = get_registry().to_anthropic_tools()

response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Calculate BMI for 70kg, 1.75m"}],
    tools=tools
)
```

### HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fluxem_tools import get_registry

model_id = "Qwen/Qwen3-4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tools = get_registry().to_openai_tools()
# Use with model's tool calling capabilities
```

## Tool Categories (40+ domains)

### Core Mathematics (30 tools)
- **arithmetic**: Basic operations, expressions
- **number_theory**: Primes, GCD, LCM, factorization
- **combinatorics**: Factorial, permutations, combinations
- **statistics**: Mean, median, variance, correlation
- **probability**: Distributions, Bayes' rule
- **calculus**: Derivatives, integrals

### Science & Engineering (60+ tools)
- **physics**: Unit conversion, dimensional analysis
- **chemistry**: Molecular weight, balancing equations
- **biology**: DNA/RNA analysis, protein calculations
- **electrical**: Ohm's law, circuits, power
- **thermodynamics**: Heat transfer, gas laws, Carnot efficiency
- **acoustics**: Decibels, Doppler effect, wavelength
- **astronomy**: Orbital mechanics, parallax, moon phase
- **optics**: Lenses, refraction, diffraction
- **fluid_dynamics**: Reynolds number, Bernoulli, drag
- **nuclear**: Radioactive decay, binding energy

### Advanced Mathematics (25 tools)
- **math_advanced**: Vectors, matrices, complex numbers
- **geometry**: Distance, rotation, transformations
- **graphs**: Shortest path, connectivity
- **sets**: Union, intersection, complement
- **logic**: Tautology checking
- **geometric_algebra**: Clifford algebra Cl(3,0)

### Data & Information (20 tools)
- **data**: Array operations, records
- **information_theory**: Entropy, KL divergence
- **signal_processing**: Convolution, DFT, filters
- **text**: Levenshtein distance, readability metrics

### Finance & Economics (15 tools)
- **finance**: Compound interest, NPV, loan payments
- **currency**: Exchange rates, inflation adjustment

### Everyday Practical (50+ tools)
- **cooking**: Recipe scaling, unit conversion
- **fitness**: BMI, BMR, heart rate zones
- **travel**: Timezone conversion, fuel consumption
- **diy**: Paint area, tile count, lumber calculation
- **photography**: Exposure, depth of field, focal length
- **gardening**: Soil volume, water needs, spacing
- **security**: RBAC permission checking

### Music & Time (10 tools)
- **music**: Chord analysis, transposition
- **temporal**: Date arithmetic, day of week

## Tool Reference

Every tool is deterministic - same input always produces same output.

### Example Tools

| Tool | Description | Example |
|------|-------------|---------|
| `arithmetic` | Evaluate math expression | `"2 + 3 * 4"` → `14` |
| `electrical_ohms_law` | V = I × R | `{V:12, I:2}` → `6.0` |
| `chemistry_mw` | Molecular weight | `"H2O"` → `18.015` |
| `fitness_bmi` | Body Mass Index | `{weight:70, height:1.75}` → `22.86` |
| `geo_distance` | Haversine distance | NYC to LA → `3935746 m` |
| `acoustics_db_add` | Add decibels | `{60, 60}` → `63.01` |
| `photo_depth_of_field` | DoF calculation | Near/far limits |

## Export Formats

```python
from fluxem_tools import get_registry

registry = get_registry()

# OpenAI format
openai_tools = registry.to_openai_tools()

# Anthropic format
anthropic_tools = registry.to_anthropic_tools()

# Full JSON export
registry.export_json("tools.json")

# JSON Schema
schema = registry.to_json_schema()
```

## Search and Filter

```python
from fluxem_tools import search_tools, list_domains

# Search by keyword
voltage_tools = search_tools("voltage")
for tool in voltage_tools:
    print(f"{tool.name}: {tool.description}")

# Get tools by domain
domains = list_domains()
registry = get_registry()
electrical_tools = registry.get_domain_tools("electrical")
```

## Why Deterministic Tools?

LLMs are powerful but unreliable at precise computation. FluxEM Tools provides:

1. **Accuracy**: Deterministic computation, not stochastic generation
2. **Consistency**: Same input always produces same output
3. **Speed**: Direct calculation, no inference needed
4. **Coverage**: 210+ tools across 40+ domains
5. **Integration**: Works with any LLM that supports tool calling

## Benchmarks

Using base Qwen3-4B-Instruct (no fine-tuning):
- **Tool Selection Accuracy**: 91.7%
- **Argument Parsing Accuracy**: 94.2%
- **End-to-End Accuracy**: 89.3%

The tools themselves are 100% accurate - they're deterministic computations.

## Adding Custom Tools

```python
from fluxem_tools import ToolSpec, get_registry

# Create a custom tool
custom_tool = ToolSpec(
    name="my_custom_tool",
    function=lambda args: args["x"] ** 2,
    description="Square a number",
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "Number to square"}
        },
        "required": ["x"]
    },
    domain="custom",
    tags=["math", "square"]
)

registry = get_registry()
registry.register(custom_tool)
```

## License

MIT License

## Links

- [GitHub Repository](https://github.com/Hmbown/FluxEM)
- [PyPI Package](https://pypi.org/project/fluxem-tools/)
- [HuggingFace](https://huggingface.co/Hmbown/fluxem-tools)

## Citation

```bibtex
@software{fluxem_tools,
  author = {Hunter Bown},
  title = {FluxEM Tools: Deterministic Computation Tools for LLM Tool-Calling},
  year = {2026},
  url = {https://github.com/Hmbown/FluxEM}
}
```
