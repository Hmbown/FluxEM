# FluxEM: Full System Roadmap

## Vision

An LLM that reasons over **structured domain embeddings** rather than raw tokens, delegating computation to exact tools and composing results through embedding algebra—not string concatenation.

```
Input: "What's the entropy of a binomial distribution with n=5! trials and p=0.3?"

Current LLM: Generates tokens, probably makes arithmetic errors
FluxEM LLM:
  1. Detects "5!" → encodes as factorial embedding
  2. Calls factorial tool → 120 (as embedding)
  3. Detects "binomial... n=120, p=0.3" → composes into probability embedding
  4. Calls entropy tool → exact result
  5. Outputs answer with structured reasoning trace
```

---

## Current State

### What Exists ✓

| Component | Status | Location |
|-----------|--------|----------|
| Domain encoders (22 domains) | ✓ Complete | `fluxem/domains/*/` |
| Tool registry (60+ tools) | ✓ Complete | `experiments/qwen3_toolcalling/tool_registry.py` |
| Embedding layout (256-dim) | ✓ Complete | `fluxem/core/base.py` |
| Domain tags (36 types) | ✓ Complete | `fluxem/integration/tokenizer.py` |
| Training config | ✓ Draft | `experiments/configs/toolcalling_multidomain.yaml` |
| Data generation script | ✓ Draft | `experiments/scripts/generate_toolcalling_data.py` |
| Benchmark prompts | ✓ Complete | `experiments/qwen3_toolcalling/benchmark_data.py` |
| Acceptance tests | ✓ Passing | `tests/test_acceptance_new_domains.py` |
| Span detection | ✓ Complete | `fluxem/detection/` (13 domain patterns) |

### What's Missing ✗

| Component | Status | Priority |
|-----------|--------|----------|
| Span detection (auto-detect domains in text) | ✓ Complete | `fluxem/detection/` |
| Embedding projection layer (128/256 → 2048) | ✗ Not built | **Critical** |
| Embedding injection into LLM hidden states | ✗ Not built | **Critical** |
| Composition operators (embedding algebra) | ✗ Not built | **Critical** |
| Multi-tool chain planning | ✗ Not built | **High** |
| Hybrid forward pass | ✗ Not built | **Critical** |
| Training loop with embedding injection | ✗ Not built | **Critical** |
| Chain-aware data generation | ✗ Partial | **High** |
| End-to-end inference pipeline | ✗ Not built | **High** |

---

## Phase 0: Foundation Decisions

### 0.1 Embedding Dimension Standardization

**Decision needed:** 128 vs 256

Current state:
- `base.py`: EMBEDDING_DIM = 128
- Arithmetic encoders: default dim=256

**Recommendation: Standardize on 256**

Rationale:
- More domain tag space (can grow beyond 36 domains)
- More specific encoding capacity
- More composition dimensions
- Arithmetic encoders already use it

**Layout proposal (256-dim):**
```
dims 0-15:    Domain tag (16 dims, supports 64+ domains)
dims 16-127:  Domain-specific encoding (112 dims)
dims 128-191: Shared semantic features (64 dims)
dims 192-255: Cross-domain composition (64 dims)
```

**Files to update:**
- `fluxem/core/base.py` - EMBEDDING_DIM and layout constants
- All domain encoders in `fluxem/domains/*/`
- Config files

### 0.2 Composition Algebra Design

How do embeddings combine?

**Option A: Additive**
```python
composed = emb_a + emb_b
```
Simple but loses information.

**Option B: Concatenate + Project**
```python
composed = project(concat(emb_a, emb_b))  # 512 → 256
```
Preserves both, learned projection.

**Option C: Gated Fusion**
```python
gate = sigmoid(W_g @ concat(emb_a, emb_b))
composed = gate * emb_a + (1 - gate) * emb_b
```
Learned selective combination.

**Option D: Attention-based**
```python
# Treat embeddings as sequence, apply self-attention
composed = attention([emb_a, emb_b, ...])
```
Most flexible, handles variable-length chains.

**Recommendation: Start with B (concat+project), upgrade to D for chains**

---

## Phase 1: Span Detection

**Goal:** Automatically identify domain-relevant spans in input text.

### 1.1 Pattern-Based Detection (Week 1)

Create `fluxem/detection/span_detector.py`:

```python
@dataclass
class DetectedSpan:
    start: int
    end: int
    text: str
    domain: DomainType
    confidence: float
    parsed_value: Any  # Pre-parsed for encoder

class SpanDetector:
    def detect(self, text: str) -> List[DetectedSpan]:
        """Find all domain spans in text."""
        spans = []
        spans.extend(self._detect_numbers(text))
        spans.extend(self._detect_combinatorics(text))
        spans.extend(self._detect_dates(text))
        spans.extend(self._detect_formulas(text))
        # ... all domains
        return self._resolve_overlaps(spans)
```

**Patterns to implement:**

| Domain | Patterns |
|--------|----------|
| Arithmetic | `\d+\.?\d*`, scientific notation |
| Combinatorics | `n!`, `C(n,k)`, `P(n,k)`, `\d+\s*choose\s*\d+` |
| Probability | `Bernoulli`, `Binomial`, `P(X=k)` |
| Statistics | `mean of [...]`, `variance`, `std dev` |
| Temporal | ISO dates, `YYYY-MM-DD`, natural dates |
| Finance | `$`, `NPV`, `IRR`, cash flow patterns |
| Physics | numbers with units (`9.8 m/s^2`) |
| Chemistry | molecular formulas (`H2O`, `C6H12O6`) |
| Calculus | `d/dx`, `∫`, polynomial notation |
| Control | matrix notation, state space |

### 1.2 ML-Based Detection (Week 2-3)

Train a small classifier to detect spans:

```python
class LearnedSpanDetector(nn.Module):
    """Token-level domain classification."""

    def __init__(self, vocab_size, hidden_dim=256, num_domains=36):
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_domains + 1)  # +1 for "none"

    def forward(self, token_ids):
        # Returns per-token domain predictions
        # Use BIO tagging for span boundaries
        ...
```

### 1.3 Hybrid Detection

Combine pattern + ML:
- Patterns for high-precision anchors
- ML for boundary refinement and ambiguous cases

**Deliverables:**
- [ ] `fluxem/detection/__init__.py`
- [ ] `fluxem/detection/patterns.py` - regex patterns per domain
- [ ] `fluxem/detection/span_detector.py` - main detector class
- [ ] `fluxem/detection/learned_detector.py` - ML-based detector
- [ ] `tests/test_span_detection.py`

---

## Phase 2: Embedding Projection & Injection

**Goal:** Merge FluxEM embeddings into Qwen3's hidden states.

### 2.1 Projection Layer

Create `fluxem/integration/projector.py`:

```python
class FluxEMProjector(nn.Module):
    """Project FluxEM embeddings to LLM hidden dimension."""

    def __init__(
        self,
        fluxem_dim: int = 256,
        llm_dim: int = 2048,  # Qwen3-4B hidden size
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(fluxem_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, fluxem_emb: Tensor) -> Tensor:
        """Project 256-dim FluxEM to 2048-dim LLM space."""
        return self.layers(fluxem_emb)
```

### 2.2 Injection Strategies

**Strategy A: Additive Injection**
```python
def inject_additive(hidden_states, positions, projected_embs):
    """Add projected embeddings to hidden states at span positions."""
    for pos, emb in zip(positions, projected_embs):
        hidden_states[:, pos, :] += emb
    return hidden_states
```

**Strategy B: Replacement Injection**
```python
def inject_replace(hidden_states, positions, projected_embs):
    """Replace hidden states at span positions."""
    for pos, emb in zip(positions, projected_embs):
        hidden_states[:, pos, :] = emb
    return hidden_states
```

**Strategy C: Gated Injection**
```python
class GatedInjection(nn.Module):
    def __init__(self, dim):
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, hidden, projected):
        gate = self.gate(torch.cat([hidden, projected], dim=-1))
        return gate * projected + (1 - gate) * hidden
```

**Recommendation: Start with A (additive), experiment with C (gated)**

### 2.3 Injection Points

Where in the transformer to inject:

1. **Input embedding layer** - earliest, most influence
2. **After layer N** - inject mid-network
3. **Multiple layers** - inject at layers 0, 8, 16, 24

**Recommendation: Start with input layer, experiment with multi-layer**

**Deliverables:**
- [ ] `fluxem/integration/projector.py`
- [ ] `fluxem/integration/injection.py`
- [ ] `tests/test_projection.py`

---

## Phase 3: Composition Operators

**Goal:** Enable embedding algebra for multi-tool chains.

### 3.1 Binary Composition

```python
class EmbeddingComposer(nn.Module):
    """Compose two FluxEM embeddings."""

    def __init__(self, emb_dim: int = 256, method: str = "concat_project"):
        super().__init__()
        self.method = method

        if method == "concat_project":
            self.project = nn.Linear(emb_dim * 2, emb_dim)
        elif method == "gated":
            self.gate_net = nn.Linear(emb_dim * 2, emb_dim)
        elif method == "bilinear":
            self.bilinear = nn.Bilinear(emb_dim, emb_dim, emb_dim)

    def forward(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        if self.method == "concat_project":
            return self.project(torch.cat([emb_a, emb_b], dim=-1))
        elif self.method == "gated":
            gate = torch.sigmoid(self.gate_net(torch.cat([emb_a, emb_b], dim=-1)))
            return gate * emb_a + (1 - gate) * emb_b
        elif self.method == "bilinear":
            return self.bilinear(emb_a, emb_b)
```

### 3.2 Chain Composition

For multi-step tool chains:

```python
class ChainComposer(nn.Module):
    """Compose a sequence of embeddings."""

    def __init__(self, emb_dim: int = 256, max_chain: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_dim, num_heads=8)
        self.position_emb = nn.Embedding(max_chain, emb_dim)

    def forward(self, embeddings: List[Tensor]) -> Tensor:
        """Compose variable-length chain into single embedding."""
        # Stack embeddings: (chain_len, batch, emb_dim)
        x = torch.stack(embeddings, dim=0)

        # Add positional encoding
        positions = torch.arange(len(embeddings))
        x = x + self.position_emb(positions).unsqueeze(1)

        # Self-attention to compose
        composed, _ = self.attention(x, x, x)

        # Return final or pooled
        return composed.mean(dim=0)  # or composed[-1]
```

### 3.3 Operation-Aware Composition

Composition should know what operation connects embeddings:

```python
class OperationAwareComposer(nn.Module):
    """Compose embeddings with operation context."""

    OPERATIONS = ["add", "subtract", "multiply", "divide", "chain", "nest", "compare"]

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.op_embeddings = nn.Embedding(len(self.OPERATIONS), emb_dim)
        self.composer = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),  # emb_a + emb_b + op
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, emb_a: Tensor, emb_b: Tensor, operation: str) -> Tensor:
        op_idx = self.OPERATIONS.index(operation)
        op_emb = self.op_embeddings(torch.tensor(op_idx))
        return self.composer(torch.cat([emb_a, emb_b, op_emb], dim=-1))
```

**Deliverables:**
- [ ] `fluxem/composition/__init__.py`
- [ ] `fluxem/composition/binary.py`
- [ ] `fluxem/composition/chain.py`
- [ ] `fluxem/composition/operation_aware.py`
- [ ] `tests/test_composition.py`

---

## Phase 4: Hybrid Model Architecture

**Goal:** Modified Qwen3 that processes FluxEM embeddings.

### 4.1 HybridQwen3 Model

```python
class HybridQwen3(nn.Module):
    """Qwen3 with FluxEM embedding injection."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-4B-Instruct",
        fluxem_dim: int = 256,
        injection_layers: List[int] = [0],
        injection_method: str = "additive",
    ):
        super().__init__()

        # Load base model
        self.qwen = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # FluxEM components
        self.span_detector = SpanDetector()
        self.encoder_registry = DomainEncoderRegistry()
        self.projector = FluxEMProjector(fluxem_dim, self.qwen.config.hidden_size)
        self.composer = ChainComposer(fluxem_dim)

        # Injection config
        self.injection_layers = injection_layers
        self.injection_method = injection_method

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        fluxem_spans: List[DetectedSpan] = None,
        **kwargs
    ):
        # 1. Get token embeddings from Qwen
        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)

        # 2. If spans provided, encode and inject
        if fluxem_spans:
            for span in fluxem_spans:
                # Encode span to FluxEM embedding
                encoder = self.encoder_registry.get_encoder(span.domain)
                fluxem_emb = encoder.encode(span.parsed_value)

                # Project to LLM dimension
                projected = self.projector(torch.tensor(fluxem_emb))

                # Inject at span position
                if self.injection_method == "additive":
                    inputs_embeds[:, span.token_start:span.token_end, :] += projected
                elif self.injection_method == "replace":
                    inputs_embeds[:, span.token_start:span.token_end, :] = projected

        # 3. Forward through Qwen with modified embeddings
        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
```

### 4.2 Tool Calling Integration

```python
class HybridQwen3WithTools(HybridQwen3):
    """HybridQwen3 with tool calling loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_registry = create_tool_registry()
        self.max_tool_calls = 8

    def generate_with_tools(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> Dict:
        """Generate response, executing tools as needed."""

        # 1. Detect spans in input
        spans = self.span_detector.detect(prompt)

        # 2. Initial generation
        response = self._generate_step(prompt, spans)

        # 3. Tool calling loop
        tool_calls = []
        embeddings_chain = []

        for _ in range(self.max_tool_calls):
            # Check if response contains tool call
            tool_call = self._parse_tool_call(response)
            if not tool_call:
                break

            # Execute tool
            result = self._execute_tool(tool_call)
            tool_calls.append({"call": tool_call, "result": result})

            # Get result embedding
            result_emb = self._encode_result(result, tool_call.domain)
            embeddings_chain.append(result_emb)

            # Compose chain embedding
            if len(embeddings_chain) > 1:
                composed = self.composer(embeddings_chain)
            else:
                composed = embeddings_chain[0]

            # Continue generation with composed embedding
            response = self._generate_step(
                prompt + self._format_tool_results(tool_calls),
                spans + [self._span_from_composed(composed)]
            )

        return {
            "response": response,
            "tool_calls": tool_calls,
            "embedding_chain": embeddings_chain,
        }
```

**Deliverables:**
- [ ] `fluxem/models/__init__.py`
- [ ] `fluxem/models/hybrid_qwen3.py`
- [ ] `fluxem/models/tool_integration.py`
- [ ] `tests/test_hybrid_model.py`

---

## Phase 5: Training Pipeline

**Goal:** Train HybridQwen3 to use FluxEM embeddings and tools.

### 5.1 Training Data Generation

Enhance `generate_toolcalling_data.py` for chains:

```python
def generate_chain_sample(
    domains: List[str],
    chain_length: int,
    difficulty: str,
) -> Dict:
    """Generate a multi-tool chain sample."""

    # Example: factorial → binomial → entropy
    if chain_length == 3 and "combinatorics" in domains and "probability" in domains:
        n = random.randint(3, 7)
        k = random.randint(1, n-1)
        p = round(random.uniform(0.1, 0.9), 2)

        # Step 1: factorial
        fact_n = math.factorial(n)

        # Step 2: binomial pmf
        pmf = math.comb(fact_n, k) * (p ** k) * ((1-p) ** (fact_n - k))

        # Step 3: entropy (of the distribution)
        # ...

        return {
            "text": f"What's the entropy of a binomial distribution with n={n}! trials, k={k}, p={p}?",
            "spans": [
                {"type": "comb_term", "start": X, "end": Y, "value": {"kind": "factorial", "n": n}},
            ],
            "tool_chain": [
                {"tool": "combinatorics_factorial", "input": f"{n}!", "output": fact_n},
                {"tool": "probability_binomial_pmf", "input": f"n={fact_n} k={k} p={p}", "output": pmf},
                {"tool": "info_entropy", "input": "...", "output": entropy},
            ],
            "target_text": "...",
            "target_value": entropy,
        }
```

### 5.2 Training Objectives

Multiple loss components:

```python
class HybridTrainingLoss(nn.Module):
    """Combined loss for hybrid model training."""

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {
            "lm": 1.0,           # Language modeling
            "tool_selection": 0.5,  # Correct tool choice
            "span_detection": 0.3,  # Identify domain spans
            "embedding_align": 0.2,  # Embedding quality
            "chain_completion": 0.4, # Complete multi-step chains
        }

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        tool_pred: Tensor,
        tool_labels: Tensor,
        span_pred: Tensor,
        span_labels: Tensor,
        embeddings: Tensor,
        target_embeddings: Tensor,
        chain_completed: bool,
        chain_target: bool,
    ) -> Dict[str, Tensor]:

        losses = {}

        # 1. Language modeling loss
        losses["lm"] = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # 2. Tool selection loss
        losses["tool_selection"] = F.cross_entropy(tool_pred, tool_labels)

        # 3. Span detection loss (BIO tagging)
        losses["span_detection"] = F.cross_entropy(
            span_pred.view(-1, span_pred.size(-1)),
            span_labels.view(-1),
            ignore_index=-100
        )

        # 4. Embedding alignment loss
        losses["embedding_align"] = F.mse_loss(embeddings, target_embeddings)

        # 5. Chain completion loss
        losses["chain_completion"] = F.binary_cross_entropy_with_logits(
            chain_completed, chain_target.float()
        )

        # Weighted sum
        total = sum(self.weights[k] * v for k, v in losses.items())
        losses["total"] = total

        return losses
```

### 5.3 Training Loop

```python
def train_hybrid_model(
    model: HybridQwen3,
    train_data: Dataset,
    val_data: Dataset,
    config: TrainingConfig,
):
    """Train the hybrid model."""

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    loss_fn = HybridTrainingLoss(config.loss_weights)

    for epoch in range(config.epochs):
        model.train()

        for batch in train_data:
            # 1. Detect spans (or use pre-annotated)
            spans = batch["spans"]

            # 2. Forward pass with FluxEM injection
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                fluxem_spans=spans,
            )

            # 3. Compute losses
            losses = loss_fn(
                logits=outputs.logits,
                labels=batch["labels"],
                tool_pred=outputs.tool_logits,
                tool_labels=batch["tool_labels"],
                # ... etc
            )

            # 4. Backward + optimize
            losses["total"].backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Validation
        val_metrics = evaluate_hybrid_model(model, val_data)
        print(f"Epoch {epoch}: {val_metrics}")
```

### 5.4 LoRA Configuration

For efficient fine-tuning:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Also train the projector and composer
    modules_to_save=["projector", "composer"],
)
```

**Deliverables:**
- [ ] `experiments/scripts/train_hybrid.py` - main training script
- [ ] `fluxem/training/__init__.py`
- [ ] `fluxem/training/loss.py` - loss functions
- [ ] `fluxem/training/data.py` - data loading utilities
- [ ] `experiments/scripts/generate_chain_data.py` - chain data generation

---

## Phase 6: Inference Pipeline

**Goal:** End-to-end inference with tool calling.

### 6.1 Inference API

```python
class FluxEMInference:
    """High-level inference API."""

    def __init__(self, model_path: str):
        self.model = HybridQwen3WithTools.from_pretrained(model_path)
        self.model.eval()

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        return_trace: bool = False,
    ) -> Union[str, Dict]:
        """Run inference with tool calling."""

        result = self.model.generate_with_tools(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        if return_trace:
            return {
                "answer": result["response"],
                "tool_calls": result["tool_calls"],
                "embeddings": [e.tolist() for e in result["embedding_chain"]],
                "reasoning_trace": self._format_trace(result),
            }

        return result["response"]

    def _format_trace(self, result: Dict) -> str:
        """Format reasoning trace for debugging."""
        lines = []
        for i, tc in enumerate(result["tool_calls"]):
            lines.append(f"Step {i+1}: {tc['call'].tool}({tc['call'].args}) → {tc['result']}")
        return "\n".join(lines)
```

### 6.2 Streaming Inference

```python
class FluxEMStreaming:
    """Streaming inference with real-time tool calls."""

    async def stream(
        self,
        prompt: str,
        on_token: Callable[[str], None],
        on_tool_call: Callable[[Dict], None],
        on_tool_result: Callable[[Dict], None],
    ):
        """Stream generation with tool call callbacks."""

        # ... implementation for real-time streaming
        # Emits tokens as generated
        # Pauses for tool calls, emits results
        # Continues generation
```

### 6.3 Batch Inference

```python
def batch_inference(
    model: FluxEMInference,
    prompts: List[str],
    batch_size: int = 8,
) -> List[Dict]:
    """Efficient batch inference."""

    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # Batch span detection
        # Batch encoding
        # Batch forward
        # Sequential tool execution (or parallel where independent)
        ...
    return results
```

**Deliverables:**
- [ ] `fluxem/inference/__init__.py`
- [ ] `fluxem/inference/api.py` - main inference API
- [ ] `fluxem/inference/streaming.py` - streaming support
- [ ] `fluxem/inference/batch.py` - batch processing
- [ ] `examples/inference_demo.py`

---

## Phase 7: Evaluation & Benchmarks

### 7.1 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Tool Selection Accuracy | Correct tool chosen | ≥95% |
| Argument Extraction F1 | Correct args parsed | ≥90% |
| Execution Accuracy | Correct final result | ≥95% |
| Chain Completion Rate | Multi-step chains completed | ≥85% |
| Embedding Quality | Reconstruction error | ≤0.01 |
| Latency | Time per query | ≤500ms |
| Tokens Saved | vs baseline token generation | ≥40% |

### 7.2 Benchmark Suites

**Suite 1: Single Tool (existing)**
- 60+ prompts across 22 domains
- Location: `experiments/qwen3_toolcalling/benchmark_data.py`

**Suite 2: Two-Tool Chains**
```python
TWO_TOOL_CHAINS = [
    {
        "prompt": "What's 5! + 6!?",
        "chain": ["factorial(5)", "factorial(6)", "add"],
        "expected": 840,
    },
    {
        "prompt": "Mean of first 5 primes?",
        "chain": ["primes_up_to(11)", "mean"],
        "expected": 5.6,
    },
    # ... 50+ cases
]
```

**Suite 3: Three+ Tool Chains**
```python
MULTI_TOOL_CHAINS = [
    {
        "prompt": "Entropy of binomial(n=5!, k=3, p=0.4)?",
        "chain": ["factorial(5)", "binomial_pmf", "entropy"],
        "expected": ...,
    },
    # ... 30+ cases
]
```

**Suite 4: Cross-Domain Composition**
```python
CROSS_DOMAIN = [
    {
        "prompt": "NPV of cash flows where each payment is C(10,k) for k=1,2,3 at rate 8%?",
        "domains": ["combinatorics", "finance"],
        "chain": ["ncr(10,1)", "ncr(10,2)", "ncr(10,3)", "npv"],
        "expected": ...,
    },
]
```

### 7.3 Comparison Baselines

1. **Vanilla Qwen3-4B** - no FluxEM, no tools
2. **Qwen3-4B + Tools** - tools but no embeddings
3. **FluxEM Hybrid** - full system

Measure:
- Accuracy on each benchmark suite
- Token count for equivalent tasks
- Latency
- Error types (arithmetic errors, wrong tool, etc.)

**Deliverables:**
- [ ] `experiments/benchmarks/single_tool.py`
- [ ] `experiments/benchmarks/two_tool_chains.py`
- [ ] `experiments/benchmarks/multi_tool_chains.py`
- [ ] `experiments/benchmarks/cross_domain.py`
- [ ] `experiments/scripts/run_benchmarks.py`
- [ ] `experiments/scripts/compare_baselines.py`

---

## Phase 8: Optimization & Deployment

### 8.1 Performance Optimization

- **Embedding caching**: Cache frequently used embeddings
- **Batch tool execution**: Execute independent tools in parallel
- **Quantization**: INT8/INT4 for projector and composer
- **Compilation**: torch.compile for hot paths

### 8.2 Deployment Options

1. **Local CLI** - existing `experiments/qwen3_toolcalling/`
2. **REST API** - FastAPI server
3. **Python SDK** - `pip install fluxem`
4. **HuggingFace** - upload trained model

**Deliverables:**
- [ ] `fluxem/serving/api.py` - REST API
- [ ] `fluxem/serving/cli.py` - CLI interface
- [ ] `setup.py` / `pyproject.toml` updates
- [ ] Docker configuration
- [ ] HuggingFace model card

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0: Decisions | 1 day | None |
| Phase 1: Span Detection | 1-2 weeks | Phase 0 |
| Phase 2: Projection/Injection | 1 week | Phase 0 |
| Phase 3: Composition | 1-2 weeks | Phase 2 |
| Phase 4: Hybrid Model | 2 weeks | Phase 1, 2, 3 |
| Phase 5: Training | 2-3 weeks | Phase 4 |
| Phase 6: Inference | 1 week | Phase 4 |
| Phase 7: Evaluation | 1-2 weeks | Phase 5, 6 |
| Phase 8: Optimization | 1-2 weeks | Phase 7 |

**Total: 10-16 weeks for full system**

**MVP (single tool calls working): 4-6 weeks** (Phases 0-4 + basic Phase 6)

---

## Immediate Next Steps

1. **Decision**: Confirm 256-dim embeddings
2. **Start Phase 1**: Build span detector (pattern-based first)
3. **Start Phase 2**: Build projector (simple linear first)
4. **Integration test**: Inject one embedding into Qwen3, verify it doesn't break

Want me to start on any of these?
