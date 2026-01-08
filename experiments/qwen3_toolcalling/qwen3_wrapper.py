"""
Qwen3-4B MLX Wrapper for FluxEM Tool Calling.

This module provides a flexible wrapper that:
1. Loads Qwen3-4B models (preferring MLX for Apple Silicon)
2. Detects which FluxEM domain applies to a user query
3. Calls appropriate FluxEM tools
4. Generates responses incorporating tool results
"""

from typing import Dict, Optional, List, Any, Tuple, Union
from dataclasses import dataclass, field
import os
import sys
import time
import subprocess
from fluxem.backend import set_backend, BackendType

from .tool_registry import ToolDescription


# MLX is optional and must be explicitly enabled.
MLX_AVAILABLE = False
_MLX_IMPORT_ERROR: Optional[Exception] = None


def _mlx_preflight() -> bool:
    """Check MLX import in a subprocess to avoid hard crashes."""
    if os.environ.get("FLUXEM_ENABLE_MLX") != "1":
        return False
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mlx.core as mx; mx.array([1]); print('ok')",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _ensure_mlx_imported() -> bool:
    """Attempt to import MLX safely, returning availability status."""
    global MLX_AVAILABLE, _MLX_IMPORT_ERROR
    if MLX_AVAILABLE:
        return True
    if os.environ.get("FLUXEM_ENABLE_MLX") != "1":
        return False
    if not _mlx_preflight():
        return False
    try:
        import mlx.core as mx  # noqa: F401

        MLX_AVAILABLE = True
        return True
    except Exception as exc:
        _MLX_IMPORT_ERROR = exc
        MLX_AVAILABLE = False
        return False


@dataclass
class ToolContext:
    """Lightweight conversation context for multi-turn tool use."""
    last_numeric: Optional[float] = None
    last_vector: Optional[List[float]] = None
    last_vectors: List[List[float]] = field(default_factory=list)
    last_matrix: Optional[List[List[float]]] = None
    last_set: Optional[List[int]] = None
    last_sets: List[List[int]] = field(default_factory=list)
    last_sequence: Optional[str] = None
    last_array: Optional[List[Any]] = None
    last_record: Optional[Dict[str, Any]] = None
    last_table: Optional[Any] = None
    last_date: Optional[str] = None

    def reset(self) -> None:
        """Clear stored context."""
        self.last_numeric = None
        self.last_vector = None
        self.last_vectors = []
        self.last_matrix = None
        self.last_set = None
        self.last_sets = []
        self.last_sequence = None
        self.last_array = None
        self.last_record = None
        self.last_table = None
        self.last_date = None


class Qwen3MLXWrapper:
    """
    Wrapper for Qwen3-4B model with FluxEM tool-calling capabilities.

    Supports:
    - MLX backend for Apple Silicon (preferred)
    - Domain detection using LLM
    - Tool selection and execution
    - Response generation with tool results
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_thinking: bool = True,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        tool_selection: str = "pattern",
        llm_query_extraction: bool = True,
        response_style: str = "structured",
        transformers_model_path: Optional[str] = None,
        transformers_device: str = "cpu",
        transformers_trust_remote_code: bool = False,
        transformers_local_files_only: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize Qwen3 wrapper.

        Args:
            model_path: Path to MLX model (e.g., "~/.mlx/models/Qwen/Qwen3-4B-Instruct-MLX")
            use_thinking: Whether to use Qwen3's thinking mode
            temperature: Sampling temperature (0.6-1.0 typical)
            max_tokens: Maximum output tokens
            response_style: "structured" for labeled answers, "plain" for raw tool outputs
            verbose: Print debug information
        """
        self.model_path = model_path
        self.use_thinking = use_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_selection = tool_selection.lower()
        self.llm_query_extraction = llm_query_extraction
        self.response_style = response_style.lower()
        if self.response_style not in {"structured", "plain"}:
            self.response_style = "structured"
        self.transformers_model_path = transformers_model_path
        self.transformers_device = transformers_device
        self.transformers_trust_remote_code = transformers_trust_remote_code
        self.transformers_local_files_only = transformers_local_files_only
        self.verbose = verbose
        self.context = ToolContext()

        # Initialize FluxEM backend (prefer MLX, fallback to NumPy)
        try:
            if _ensure_mlx_imported():
                set_backend(BackendType.MLX)
            else:
                set_backend(BackendType.NUMPY)
        except Exception:
            set_backend(BackendType.NUMPY)

        # Model state
        self.backend = "none"
        self.model = None
        self.tokenizer = None
        self.transformers_model = None
        self.transformers_tokenizer = None
        self.is_loaded = False

        # Initialize tool registry
        from .tool_registry import create_tool_registry

        self.tool_registry = create_tool_registry()

        # Domain detection prompt - STRICT format
        self.domain_detection_prompt = """CLASSIFY the domain of this question.

DOMAINS:
arithmetic - math operations and word problems with totals/counts: 54*44, 2**16, (100/8)*3, "how many", "total"
physics - units: km→m, m/s^2, dimensions
chemistry - molecules: H2O, C6H12O6, molecular weight
biology - DNA: GATTACA, GC content, complement
math - vectors/matrices: [3,4], vector magnitude/norm, dot product, determinant
music - pitch classes: [0,4,7], chords, transpose
geometry - points/coordinates: distance([0,0],[3,4]), rotate, midpoint (not vector magnitude)
graphs - connectivity: shortest path, connected, tree
sets - operations: union, intersection, subset
logic - formulas: tautology, validity, equivalence
number_theory - integers: prime(17), gcd(12,18), mod
data - arrays/records/tables: list stats, schema, rows/columns
combinatorics - counting: combinations, permutations, factorials
probability - pmf, binomial, Bayes rule
statistics - mean, median, variance, correlation
information_theory - entropy, cross-entropy, KL divergence
signal_processing - convolution, moving average, DFT
calculus - polynomial derivative, integral, evaluate
temporal - date arithmetic, day of week
finance - compound interest, NPV, payment
optimization - least squares, gradient step, projection
control_systems - state update, stability check

Respond with EXACTLY ONE word: arithmetic, physics, chemistry, biology, math, music, geometry, graphs, sets, logic, number_theory, data, combinatorics, probability, statistics, information_theory, signal_processing, calculus, temporal, finance, optimization, control_systems, or none.

Question: {question}
Domain:"""

        # Tool call prompt - JSON format for consistency
        self.tool_call_prompt = """You are a precise tool caller. Extract the computation/query needed.

TOOL: {tool_name}
TOOL DESCRIPTION: {tool_description}
INPUT FORMAT: {input_format}

Extract the exact input from the question:
Question: {question}

Respond with EXACT JSON:
{{"query": "the exact computation or data"}}"""

        # Tool router prompt - select tool + query in one step
        self.tool_router_prompt = """You are a tool router. Select the most appropriate tool for the question.

REQUIREMENTS:
1. Always select a tool when the question involves a computation that matches one of the tools.
2. For word problems, extract the arithmetic or calculation expression directly - do NOT leave query empty.
3. If the question asks for a numeric answer, counts, totals, sums, or differences, choose arithmetic.
4. Convert "how many", "what is the total", "how much" etc. into the appropriate expression.
5. Only select "none" if the question is completely outside all tool domains.

EXAMPLES:
- "A box has 12 apples. You eat 5. How many are left?" -> {{"tool_name": "arithmetic", "query": "12 - 5"}}
- "What is 15 * 17?" -> {{"tool_name": "arithmetic", "query": "15 * 17"}}
- "Convert 88 ft/s to m/s" -> {{"tool_name": "physics_convert", "query": "88 ft/s to m/s"}}
- "Given array [1, 2, 3], what's the mean?" -> {{"tool_name": "data_array_summary", "query": "[1, 2, 3]"}}

TOOLS:
{tools}

Return EXACT JSON:
{{"tool_name": "tool", "query": "tool input"}}

Question: {question}
Answer:"""

        # Response generation prompt
        self.response_prompt = """You have FluxEM - perfect computational tools.

QUESTION: {question}
TOOL USED: {tool_name}
TOOL RESULT: {tool_result}

Answer the question using only the tool result. Be concise.

Answer:"""

    def load_model(self) -> bool:
        """
        Load Qwen3-4B MLX model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.model_path:
            if self.verbose:
                print("No model path provided. Will use domain detection without LLM.")
            return False

        if not _ensure_mlx_imported():
            if self.verbose:
                print("MLX not available. Trying transformers backend.")
            return self._load_transformers_model()

        try:
            from mlx_lm import load, generate
            from transformers import AutoTokenizer

            self.model_path = os.path.expanduser(self.model_path)
            self.model, self.tokenizer = load(self.model_path)
            self.is_loaded = True
            self.backend = "mlx"

            if self.verbose:
                print(f"Model loaded from: {self.model_path}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Error loading model: {e}")
            self.is_loaded = False
            return False

    def _load_transformers_model(self) -> bool:
        model_id = self.transformers_model_path or self.model_path
        if not model_id:
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            if self.verbose:
                print(f"Transformers backend unavailable: {exc}")
            return False

        try:
            self.transformers_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=self.transformers_local_files_only,
                trust_remote_code=self.transformers_trust_remote_code,
            )
            if self.transformers_tokenizer.pad_token is None:
                self.transformers_tokenizer.pad_token = self.transformers_tokenizer.eos_token

            self.transformers_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=self.transformers_local_files_only,
                torch_dtype=torch.float32,
                trust_remote_code=self.transformers_trust_remote_code,
            )
            device = torch.device(self.transformers_device)
            self.transformers_model.to(device)
            self.transformers_model.eval()

            self.is_loaded = True
            self.backend = "transformers"
            if self.verbose:
                print(f"Transformers model loaded from: {model_id}")
            return True
        except Exception as exc:
            if self.verbose:
                print(f"Error loading transformers model: {exc}")
            self.is_loaded = False
            self.backend = "none"
            return False

    def detect_domain(self, prompt: str) -> str:
        """
        Detect which FluxEM domain applies to a user prompt.

        Args:
            prompt: User's question or query

        Returns:
            Domain name (lowercase) or "none"
        """
        if self.is_loaded:
            detection_prompt = self.domain_detection_prompt.format(question=prompt)

            # Generate domain detection
            response = self._generate(detection_prompt, max_tokens=50)

            # Extract domain name from response
            # Look for common patterns like "**domain**", "domain:" or just the word at the end
            import re

            domain_match = re.search(
                r"\*\*(\w+)\*\*|domain[:\s]+(\w+)|arithmetic|physics|chemistry|biology|math|music|geometry|graphs|sets|logic|number_theory|data|combinatorics|probability|statistics|information_theory|signal_processing|calculus|temporal|finance|optimization|control_systems",
                response,
                re.IGNORECASE,
            )
            if domain_match:
                domain = (
                    domain_match.group(1)
                    or domain_match.group(2)
                    or domain_match.group(0)
                )
                domain = domain.strip().lower()
            else:
                domain = response.strip().lower()

            if self.verbose:
                print(f"LLM detected domain: {domain}")

            # Validate domain
            valid_domains = {tool.domain for tool in self.tool_registry.values()}
            if domain not in valid_domains and domain != "none":
                if self.verbose:
                    print(f"Unknown domain '{domain}', defaulting to 'none'")
                domain = "none"

            if domain == "none":
                pattern_domain = self._detect_domain_pattern(prompt)
                if pattern_domain != "none":
                    domain = pattern_domain

            domain = self._override_domain(prompt, domain)
            return domain
        else:
            # Fallback: simple pattern-based detection
            if self.verbose:
                print("Using pattern-based domain detection (fallback)")
            return self._detect_domain_pattern(prompt)

    def _detect_domain_pattern(self, prompt: str) -> str:
        """
        Fallback pattern-based domain detection without LLM.

        Simple keyword and pattern matching.
        """
        import re

        prompt_lower = prompt.lower()

        # Graphs (Graph(...) or nodes/edges)
        if "graph(" in prompt_lower or (
            "nodes=" in prompt_lower and "edges=" in prompt_lower
        ):
            return "graphs"

        # Sets (brace notation)
        if re.search(r"\{[^}]+\}", prompt_lower):
            return "sets"

        # Biology patterns (DNA/RNA sequences)
        if re.search(r"\b[atgcu]{4,}\b", prompt_lower):
            return "biology"

        # Arithmetic patterns
        arithmetic_keywords = [
            "calculate",
            "compute",
            "add",
            "subtract",
            "multiply",
            "divide",
            "square",
            "cube",
            "power",
            "root",
            "factorial",
        ]
        arithmetic_op = re.search(r"[\d\)\]]\s*(\+|-|\*|/|\*\*|\^|%)\s*[\d\(\[]", prompt_lower)

        # Physics patterns
        physics_keywords = [
            "unit",
            "dimension",
            "dimensions",
            "convert",
            "meter",
            "meters",
            "second",
            "newton",
            "joule",
            "watt",
            "force",
            "velocity",
            "acceleration",
        ]

        # Chemistry patterns
        chemistry_keywords = [
            "molecule",
            "atom",
            "bond",
            "formula",
            "molecular",
            "molecular weight",
            "reaction",
            "stoichiometry",
            "molar",
            "mass",
            "chemical",
            "balance",
        ]

        # Biology patterns
        biology_keywords = [
            "dna",
            "rna",
            "protein",
            "gene",
            "base",
            "nucleotide",
            "complement",
            "reverse complement",
            "gc content",
            "transcribe",
            "translat",
            "sequence",
            "codon",
        ]

        # Math patterns
        math_patterns = [
            r"\bvector\b",
            r"\bmatrix\b",
            r"\bdeterminant\b",
            r"\bnorm\b",
            r"\bmagnitude\b",
            r"\bnormalize\b",
            r"\bdot\b",
            r"\bcross\b",
            r"\blinear\b",
            r"\btranspose\b",
            r"\binverse\b",
        ]

        # Music patterns
        music_keywords = [
            "pitch class",
            "chord",
            "scale",
            "transpose",
            "semitone",
            "prime form",
            "normal form",
            "atonal",
        ]

        # Geometry patterns
        geometry_keywords = [
            "point",
            "distance",
            "midpoint",
            "angle",
            "transform",
            "rotate",
            "collinear",
            "coplanar",
        ]

        # Graph patterns
        graph_keywords = [
            "graph",
            "node",
            "edge",
            "path",
            "connect",
            "shortest",
            "bipartite",
            "cycle",
            "tree",
        ]

        # Sets patterns
        sets_keywords = [
            "union",
            "intersection",
            "subset",
            "complement",
            "set",
            "superset",
            "disjoint",
            "cardinality",
        ]

        # Logic patterns
        logic_keywords = [
            "tautology",
            "valid",
            "proposition",
            "implication",
            "implies",
            "contradiction",
            "satisfiable",
            "boolean",
            "equivalent",
        ]

        # Number theory patterns
        nt_keywords = [
            "prime",
            "divisor",
            "gcd",
            "mod",
            "modulo",
            "factor",
            "modular",
            "congruent",
            "euler",
            "phi",
        ]

        # Data patterns (strong cues vs. context-dependent stats)
        data_keywords_strong = [
            "array",
            "list",
            "table",
            "row",
            "rows",
            "column",
            "columns",
            "schema",
            "record",
            "dataset",
            "dataframe",
            "csv",
        ]
        data_keywords_weak = [
            "length",
            "size",
            "count",
            "mean",
            "average",
            "sum",
            "min",
            "max",
            "std",
            "stdev",
            "variance",
            "field",
            "fields",
        ]

        # Combinatorics patterns
        combinatorics_keywords = [
            "combination",
            "permutation",
            "factorial",
            "ncr",
            "npr",
            "choose",
            "arrangements",
        ]

        # Probability patterns
        probability_keywords = [
            "probability",
            "bernoulli",
            "binomial",
            "bayes",
            "pmf",
        ]

        # Statistics patterns
        statistics_keywords = [
            "mean",
            "median",
            "variance",
            "std",
            "stdev",
            "correlation",
            "corr",
        ]

        # Information theory patterns
        info_keywords = [
            "entropy",
            "cross entropy",
            "kl divergence",
            "kl",
            "information",
        ]

        # Signal processing patterns
        signal_keywords = [
            "signal",
            "convolution",
            "moving average",
            "dft",
            "fft",
            "filter",
        ]

        # Calculus patterns
        calculus_keywords = [
            "derivative",
            "differentiate",
            "integral",
            "integrate",
            "polynomial",
        ]

        # Temporal patterns
        temporal_keywords = [
            "date",
            "day of week",
            "weekday",
            "calendar",
            "days between",
            "add days",
        ]

        # Finance patterns
        finance_keywords = [
            "interest",
            "npv",
            "present value",
            "payment",
            "amortization",
            "loan",
        ]

        # Optimization patterns
        optimization_keywords = [
            "least squares",
            "gradient",
            "project",
            "constraint",
            "optimize",
        ]

        # Control systems patterns
        control_keywords = [
            "state update",
            "stability",
            "control system",
            "system matrix",
            "a matrix",
            "b matrix",
        ]
        if any(kw in prompt_lower for kw in music_keywords):
            return "music"

        if any(kw in prompt_lower for kw in nt_keywords):
            return "number_theory"

        if any(re.search(pattern, prompt_lower) for pattern in math_patterns):
            return "math"

        if any(kw in prompt_lower for kw in geometry_keywords):
            return "geometry"

        if any(kw in prompt_lower for kw in physics_keywords):
            return "physics"

        if any(kw in prompt_lower for kw in chemistry_keywords):
            return "chemistry"

        if any(kw in prompt_lower for kw in biology_keywords):
            return "biology"

        if any(kw in prompt_lower for kw in graph_keywords):
            return "graphs"

        if any(kw in prompt_lower for kw in sets_keywords):
            return "sets"

        if any(kw in prompt_lower for kw in logic_keywords):
            return "logic"

        if any(kw in prompt_lower for kw in data_keywords_strong):
            return "data"
        if any(kw in prompt_lower for kw in data_keywords_weak):
            has_context = any(
                [
                    self.context.last_array,
                    self.context.last_record,
                    self.context.last_table,
                ]
            )
            if has_context:
                return "data"
            if re.search(r"\[[^\]]+\]|\{[^}]+\}", prompt_lower):
                return "data"

        if any(kw in prompt_lower for kw in combinatorics_keywords):
            return "combinatorics"

        if any(kw in prompt_lower for kw in probability_keywords):
            return "probability"

        if any(kw in prompt_lower for kw in statistics_keywords):
            if "mean" in prompt_lower or "median" in prompt_lower:
                return "statistics"
            if "variance" in prompt_lower or "std" in prompt_lower or "stdev" in prompt_lower:
                return "statistics"
            if "correlation" in prompt_lower or "corr" in prompt_lower:
                return "statistics"

        if any(kw in prompt_lower for kw in info_keywords):
            return "information_theory"

        if any(kw in prompt_lower for kw in signal_keywords):
            return "signal_processing"

        if any(kw in prompt_lower for kw in calculus_keywords):
            return "calculus"

        if any(kw in prompt_lower for kw in temporal_keywords):
            return "temporal"

        if any(kw in prompt_lower for kw in finance_keywords):
            return "finance"

        if any(kw in prompt_lower for kw in optimization_keywords):
            return "optimization"

        if any(kw in prompt_lower for kw in control_keywords):
            return "control_systems"

        if arithmetic_op or any(kw in prompt_lower for kw in arithmetic_keywords):
            return "arithmetic"

        return "none"

    def _override_domain(self, prompt: str, domain: str) -> str:
        """Apply lightweight disambiguation rules for overlapping domains."""
        if domain != "geometry":
            return domain

        import re

        prompt_lower = prompt.lower()
        pattern_domain = self._detect_domain_pattern(prompt)
        if pattern_domain == "math":
            return "math"

        if re.search(r"\b(vector|magnitude|norm|dot|cross)\b", prompt_lower):
            return "math"

        return domain

    def _select_tools_for_prompt(self, domain: str, prompt: str) -> List[str]:
        """Select tool candidates for a domain based on prompt cues."""
        import re

        prompt_lower = prompt.lower()

        if domain == "arithmetic":
            return ["arithmetic"]

        if domain == "physics":
            if "dimension" in prompt_lower:
                return ["physics_dimensions"]
            if (
                "convert" in prompt_lower
                or "meters" in prompt_lower
                or "km" in prompt_lower
                or "kg" in prompt_lower
            ):
                return ["physics_convert"]
            return ["physics_convert", "physics_dimensions"]

        if domain == "chemistry":
            if "formula" in prompt_lower or "glucose" in prompt_lower:
                return ["chemistry_formula", "chemistry_molecule"]
            if "balance" in prompt_lower or "->" in prompt_lower:
                return ["chemistry_balance_simple"]
            if "molecular weight" in prompt_lower:
                return ["chemistry_molecule"]
            return ["chemistry_molecule", "chemistry_formula"]

        if domain == "biology":
            if "reverse complement" in prompt_lower:
                return ["biology_reverse_complement_gc"]
            if "gc content" in prompt_lower:
                return ["biology_gc_content"]
            if "molecular weight" in prompt_lower:
                return ["biology_mw"]
            if "complement" in prompt_lower:
                return ["biology_complement"]
            return ["biology_gc_content", "biology_mw", "biology_complement"]

        if domain == "math":
            if "dot product" in prompt_lower or "dot" in prompt_lower:
                return ["math_dot"]
            if "determinant" in prompt_lower:
                return ["math_determinant"]
            if "normalize" in prompt_lower:
                return ["math_normalize"]
            if "magnitude" in prompt_lower or "norm" in prompt_lower:
                return ["math_vector"]
            return ["math_vector", "math_dot"]

        if domain == "music":
            if "prime form" in prompt_lower:
                return ["music_prime_form"]
            if "normal form" in prompt_lower:
                return ["music_normal_form"]
            if "transpose" in prompt_lower:
                return ["music_transpose"]
            if "chord" in prompt_lower:
                return ["music_chord_type"]
            return ["music_prime_form", "music_normal_form"]

        if domain == "geometry":
            if "midpoint" in prompt_lower:
                return ["geometry_midpoint"]
            if "rotate" in prompt_lower:
                return ["geometry_rotate"]
            if "distance" in prompt_lower or "origin" in prompt_lower:
                return ["geometry_distance"]
            return ["geometry_distance", "geometry_midpoint"]

        if domain == "graphs":
            if "shortest path" in prompt_lower:
                return ["graphs_shortest_path"]
            if "connected" in prompt_lower:
                return ["graphs_is_connected"]
            if "tree" in prompt_lower:
                return ["graphs_is_tree"]
            if (
                "how many nodes" in prompt_lower
                or "nodes does this graph have" in prompt_lower
            ):
                return ["graphs_node_count"]
            return ["graphs_properties"]

        if domain == "sets":
            if "union" in prompt_lower:
                return ["sets_union"]
            if "intersection" in prompt_lower:
                return ["sets_intersection"]
            if "subset" in prompt_lower:
                return ["sets_subset"]
            if "complement" in prompt_lower:
                return ["sets_complement"]
            return ["sets_union", "sets_intersection", "sets_subset"]

        if domain == "logic":
            return ["logic_tautology"]

        if domain == "number_theory":
            if "gcd" in prompt_lower:
                return ["number_theory_gcd"]
            if "modular inverse" in prompt_lower or (
                "inverse" in prompt_lower and "mod" in prompt_lower
            ):
                return ["number_theory_mod_inverse"]
            if "mod" in prompt_lower and re.search(
                r"\^\s*\(?-?1\)?(?!\d)", prompt_lower
            ):
                return ["number_theory_mod_inverse"]
            if "mod" in prompt_lower or "modular" in prompt_lower:
                if "pow" in prompt_lower or "^" in prompt_lower or "**" in prompt_lower:
                    return ["number_theory_mod_pow"]
            if (
                re.search(r"\d+(?:st|nd|rd|th)\s+prime", prompt_lower)
                or "nth prime" in prompt_lower
            ):
                return ["number_theory_nth_prime"]
            if "primes up to" in prompt_lower or "all primes" in prompt_lower:
                return ["number_theory_primes_up_to"]
            if (
                "prime" in prompt_lower
                and "nth" not in prompt_lower
                and "100" not in prompt_lower
            ):
                return ["number_theory_is_prime"]
            return [
                "number_theory_mod_pow",
                "number_theory_gcd",
                "number_theory_nth_prime",
            ]

        if domain == "data":
            if any(kw in prompt_lower for kw in ["schema", "record", "fields"]):
                return ["data_record_schema"]
            if any(kw in prompt_lower for kw in ["table", "row", "rows", "column", "columns"]):
                return ["data_table_summary"]
            if any(kw in prompt_lower for kw in ["length", "size", "count"]):
                return ["data_array_length"]
            return ["data_array_summary"]

        if domain == "combinatorics":
            if "factorial" in prompt_lower:
                return ["combinatorics_factorial"]
            if "permutation" in prompt_lower or "npr" in prompt_lower:
                return ["combinatorics_npr"]
            if "combination" in prompt_lower or "ncr" in prompt_lower or "choose" in prompt_lower:
                return ["combinatorics_ncr"]
            return ["combinatorics_multiset"]

        if domain == "probability":
            if "bernoulli" in prompt_lower:
                return ["probability_bernoulli_pmf"]
            if "binomial" in prompt_lower:
                return ["probability_binomial_pmf"]
            if "bayes" in prompt_lower:
                return ["probability_bayes_rule"]
            return ["probability_binomial_pmf", "probability_bernoulli_pmf"]

        if domain == "statistics":
            if "median" in prompt_lower:
                return ["statistics_median"]
            if "variance" in prompt_lower or "std" in prompt_lower or "stdev" in prompt_lower:
                return ["statistics_variance"]
            if "correlation" in prompt_lower or "corr" in prompt_lower:
                return ["statistics_corr"]
            return ["statistics_mean"]

        if domain == "information_theory":
            if "cross" in prompt_lower:
                return ["info_cross_entropy"]
            if "kl" in prompt_lower:
                return ["info_kl_divergence"]
            return ["info_entropy"]

        if domain == "signal_processing":
            if "convolution" in prompt_lower:
                return ["signal_convolution"]
            if "moving average" in prompt_lower:
                return ["signal_moving_average"]
            if "dft" in prompt_lower or "fft" in prompt_lower:
                return ["signal_dft_magnitude"]
            return ["signal_convolution"]

        if domain == "calculus":
            if "integral" in prompt_lower or "integrate" in prompt_lower:
                return ["calculus_integral"]
            if "derivative" in prompt_lower or "differentiate" in prompt_lower:
                return ["calculus_derivative"]
            if "evaluate" in prompt_lower or "value" in prompt_lower:
                return ["calculus_evaluate"]
            return ["calculus_derivative"]

        if domain == "temporal":
            if "day of week" in prompt_lower or "weekday" in prompt_lower:
                return ["temporal_day_of_week"]
            if "difference" in prompt_lower or "days between" in prompt_lower:
                return ["temporal_diff_days"]
            return ["temporal_add_days"]

        if domain == "finance":
            if "npv" in prompt_lower or "present value" in prompt_lower:
                return ["finance_npv"]
            if "payment" in prompt_lower or "amortization" in prompt_lower or "loan" in prompt_lower:
                return ["finance_payment"]
            return ["finance_compound_interest"]

        if domain == "optimization":
            if "least squares" in prompt_lower:
                return ["optimization_least_squares_2x2"]
            if "gradient" in prompt_lower:
                return ["optimization_gradient_step"]
            if "project" in prompt_lower or "constraint" in prompt_lower:
                return ["optimization_project_box"]
            return ["optimization_least_squares_2x2"]

        if domain == "control_systems":
            if "stability" in prompt_lower:
                return ["control_is_stable_2x2"]
            return ["control_state_update"]

        return []

    def call_tool(self, domain: str, query: str) -> Dict[str, Any]:
        """
        Call appropriate FluxEM tool for the detected domain.

        Returns:
            Dictionary with tool call results
        """
        start_time = time.time()
        tool_names = self._select_tools_for_prompt(domain, query)

        if not tool_names:
            if self.verbose:
                print(f"No tools available for domain: {domain}")
            return {
                "tool_name": None,
                "success": False,
                "result": None,
                "error": "No tools available",
                "execution_time_ms": 0.0,
            }

        last_error = None
        for tool_name in tool_names:
            tool_result = self.call_tool_by_name(tool_name, query, start_time=start_time)
            if tool_result.get("success"):
                return tool_result
            last_error = tool_result.get("error")

        execution_time = (time.time() - start_time) * 1000
        return {
            "tool_name": tool_names[0] if tool_names else None,
            "success": False,
            "result": None,
            "error": last_error or "Tool failed",
            "execution_time_ms": execution_time,
        }

    def call_tool_by_name(
        self,
        tool_name: str,
        query: str,
        start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        tool_desc = self.tool_registry.get(tool_name)
        if not tool_desc:
            return {
                "tool_name": tool_name,
                "success": False,
                "result": None,
                "error": "Unknown tool",
                "execution_time_ms": 0.0,
            }

        if start_time is None:
            start_time = time.time()

        if self.llm_query_extraction and self.is_loaded:
            import re

            needs_llm = False
            if tool_name == "arithmetic":
                needs_llm = not re.search(r"[\+\-\*/%\^]", query)
            if needs_llm:
                llm_query = self._extract_query_with_llm(query, tool_name, tool_desc)
                if llm_query:
                    query = llm_query

        parsed_query = self._parse_for_tool(query, tool_name)
        try:
            if self.verbose:
                print(f"Calling tool: {tool_name} with query: {parsed_query}")
            result = tool_desc.function(parsed_query)
            execution_time = (time.time() - start_time) * 1000
            self._update_context(tool_name, parsed_query, result)
            if self.verbose:
                print(f"Tool result: {result} (took {execution_time:.2f}ms)")
            return {
                "tool_name": tool_name,
                "success": True,
                "result": result,
                "error": None,
                "execution_time_ms": execution_time,
            }
        except Exception as exc:
            execution_time = (time.time() - start_time) * 1000
            if self.verbose:
                print(f"Tool {tool_name} failed: {exc}")
            return {
                "tool_name": tool_name,
                "success": False,
                "result": None,
                "error": str(exc),
                "execution_time_ms": execution_time,
            }

    def _format_tool_router_descriptions(self) -> str:
        descriptions = []
        for tool_name, tool_desc in self.tool_registry.items():
            descriptions.append(
                f"- {tool_name} ({tool_desc.domain}): {tool_desc.description} "
                f"[input: {tool_desc.input_format}]"
            )
        return "\n".join(descriptions)

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        import json
        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        blob = match.group(0)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            cleaned = blob.replace("'", '"')
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    def _select_tool_with_llm(self, question: str) -> Optional[Dict[str, str]]:
        if not self.is_loaded:
            return None

        tools_block = self._format_tool_router_descriptions()
        prompt = self.tool_router_prompt.format(tools=tools_block, question=question)
        response = self._generate_with_messages(
            [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        if self.verbose:
            print(f"Tool router raw response: {response}")

        payload = self._extract_json_payload(response)
        tool_name = None
        query = None
        if isinstance(payload, dict):
            tool_name = payload.get("tool_name") or payload.get("tool") or payload.get("name")
            query = payload.get("query") or payload.get("input")

        if not tool_name:
            import re

            tool_match = re.search(r'tool_name"\s*:\s*"([^"]+)"', response)
            query_match = re.search(r'query"\s*:\s*"([^"]*)"', response)
            tool_name = tool_match.group(1) if tool_match else None
            query = query_match.group(1) if query_match else None

        if not tool_name:
            return None

        tool_name = tool_name.strip().lower()
        if tool_name == "none":
            return {"tool_name": "none", "query": ""}

        if tool_name not in self.tool_registry:
            if self.verbose:
                print(f"LLM selected unknown tool: {tool_name}")
            return None

        if self.verbose:
            print(f"LLM selected tool: {tool_name} | query: {query}")

        query = (query or "").strip()
        return {"tool_name": tool_name, "query": query}

    def _call_tool_with_llm(self, prompt: str) -> Dict[str, Any]:
        selection = self._select_tool_with_llm(prompt)
        if not selection or selection.get("tool_name") == "none":
            return {
                "tool_name": None,
                "success": False,
                "result": None,
                "error": "No tool selected",
                "execution_time_ms": 0.0,
                "domain": "none",
            }

        tool_name = selection["tool_name"]
        tool_desc = self.tool_registry.get(tool_name)
        domain = tool_desc.domain if tool_desc else "none"

        query = selection.get("query") or ""
        if self.llm_query_extraction and (not query or len(query) < 3):
            llm_query = self._extract_query_with_llm(prompt, tool_name, tool_desc)
            if llm_query:
                query = llm_query
        if not query:
            query = prompt

        tool_result = self.call_tool_by_name(tool_name, query)
        tool_result["domain"] = domain
        return tool_result

    def _call_tool_auto(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        if self.tool_selection in ("llm", "hybrid") and self.is_loaded:
            tool_result = self._call_tool_with_llm(prompt)
            if tool_result.get("tool_name"):
                return tool_result.get("domain", "none"), tool_result

        domain = self.detect_domain(prompt)
        if domain == "none":
            return domain, {
                "tool_name": None,
                "success": False,
                "result": None,
                "error": "No tool called",
                "execution_time_ms": 0.0,
                "domain": domain,
            }

        tool_info = self.call_tool(domain, prompt)
        tool_info["domain"] = domain
        return domain, tool_info

    def _extract_query_with_llm(
        self, question: str, tool_name: str, tool_desc: ToolDescription
    ) -> Optional[str]:
        """
        Use LLM to extract the exact query for a tool.

        This provides more consistent query extraction than regex-based parsing.
        Includes retry logic with stronger constraints.
        """
        if not self.is_loaded:
            return None

        import re

        # Build base prompt with stronger constraints
        prompt = self.tool_call_prompt.format(
            tool_name=tool_desc.name,
            tool_description=tool_desc.description,
            input_format=tool_desc.input_format,
            question=question,
        )
        
        # Add enhanced instructions for arithmetic word problems
        if tool_name == "arithmetic":
            prompt += (
                "\nFor word problems, output a single arithmetic expression using only "
                "numbers, + - * / ^ and parentheses. Do not include words.\n"
                "If the question includes numbers, the query must be non-empty.\n"
                "Examples:\n"
                "Question: A box has 12 apples. You eat 5. How many are left?\n"
                "Response: {\"query\": \"12 - 5\"}\n"
                "Question: What is 15 * 17?\n"
                "Response: {\"query\": \"15 * 17\"}\n"
                "Question: Calculate (2^30) / 13\n"
                "Response: {\"query\": \"(2**30) / 13\"}"
            )

        # Retry logic: try up to 3 times for non-empty query
        max_retries = 3
        for attempt in range(max_retries):
            response = self._generate_with_messages(
                [
                    {"role": "system", "content": "Return JSON only. Query must be non-empty if question contains numbers."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
            )
            
            if self.verbose:
                print(f"Tool query attempt {attempt + 1}/{max_retries} raw response: {response}")

            # Extract JSON with improved regex
            json_match = re.search(r'\{\s*"query"\s*:\s*"([^"]*)"\s*\}', response)
            if json_match:
                query = json_match.group(1).strip()
                
                # For arithmetic, enforce non-empty query when question has numbers
                if tool_name == "arithmetic":
                    has_numbers = bool(re.search(r'\d+', question))
                    if has_numbers and not query:
                        if attempt < max_retries - 1:
                            if self.verbose:
                                print(f"Retry {attempt + 1}: Arithmetic query is empty but question has numbers")
                            # Strengthen prompt for retry
                            prompt += "\n\nERROR: Query was empty. You MUST provide a non-empty arithmetic expression."
                            continue
                        else:
                            if self.verbose:
                                print(f"Failed after {max_retries} attempts: No valid arithmetic query extracted")
                            return None
                
                if self.verbose:
                    print(f"LLM extracted query: {query}")
                return query

            # If extraction failed and we have retries left
            if attempt < max_retries - 1:
                if self.verbose:
                    print(f"Retry {attempt + 1}: JSON extraction failed, trying again")
                continue

        if self.verbose:
            print(f"LLM extraction failed after {max_retries} attempts, last response: {response}")
        return None

    def _parse_for_tool(self, query: str, tool_name: str) -> Any:
        """
        Parse user query into appropriate format for the tool.

        Args:
            query: User's raw query
            tool_name: Name of the tool being called

        Returns:
            Parsed query in the format expected by the tool
        """
        import re
        import math
        import ast
        import json

        def _parse_vector(text):
            matches = re.findall(r"\[([^\]]+)\]", text)
            if not matches:
                return None
            values = re.findall(r"-?\d+\.?\d*", matches[0])
            return [float(v) for v in values]

        def _parse_vectors(text):
            matches = re.findall(r"\[([^\]]+)\]", text)
            vectors = []
            for match in matches[:2]:
                values = re.findall(r"-?\d+\.?\d*", match)
                vectors.append([float(v) for v in values])
            return vectors if len(vectors) == 2 else None

        def _parse_matrix(text):
            matrix_matches = re.findall(
                r"\[\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\]", text
            )
            if matrix_matches:
                rows = []
                for row_text in matrix_matches[0]:
                    values = re.findall(r"-?\d+\.?\d*", row_text)
                    rows.append([float(v) for v in values])
                return rows
            # Fallback: extract all bracketed rows
            row_matches = re.findall(r"\[([^\]]+)\]", text)
            rows = []
            for row in row_matches:
                values = re.findall(r"-?\d+\.?\d*", row)
                if values:
                    rows.append([float(v) for v in values])
            return rows if rows else None

        def _parse_sets(text):
            set_matches = re.findall(r"\{([^}]+)\}", text)
            sets = []
            for set_text in set_matches[:2]:
                values = re.findall(r"-?\d+", set_text)
                sets.append([int(v) for v in values])
            return sets if len(sets) >= 2 else None

        def _extract_balanced(text: str, open_char: str, close_char: str) -> Optional[str]:
            start = text.find(open_char)
            if start == -1:
                return None
            depth = 0
            for idx in range(start, len(text)):
                char = text[idx]
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            return None

        def _parse_literal(text: str) -> Any:
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                pass
            try:
                return ast.literal_eval(text)
            except Exception:
                return None

        def _parse_data_array(text: str) -> Optional[list]:
            blob = _extract_balanced(text, "[", "]")
            if blob:
                parsed = _parse_literal(blob)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            numbers = re.findall(r"-?\d+\.?\d*", text)
            if numbers:
                return [float(n) for n in numbers]
            return None

        def _parse_data_record(text: str) -> Optional[dict]:
            blob = _extract_balanced(text, "{", "}")
            if blob:
                parsed = _parse_literal(blob)
                if isinstance(parsed, dict):
                    return parsed
            return None

        def _parse_data_table(text: str) -> Optional[Any]:
            blob = _extract_balanced(text, "[", "]") or _extract_balanced(text, "{", "}")
            if blob:
                parsed = _parse_literal(blob)
                if isinstance(parsed, (list, dict, tuple)):
                    return parsed
            return None

        def _parse_list_literals(text: str, max_items: int = 2) -> List[Any]:
            items = []
            remaining = text
            for _ in range(max_items):
                blob = _extract_balanced(remaining, "[", "]")
                if not blob:
                    break
                parsed = _parse_literal(blob)
                if isinstance(parsed, list):
                    items.append(parsed)
                idx = remaining.find(blob)
                if idx == -1:
                    break
                remaining = remaining[idx + len(blob) :]
            return items

        def _parse_dates(text: str) -> List[str]:
            return re.findall(r"\d{4}-\d{2}-\d{2}", text)

        def _parse_graph(text):
            from fluxem.domains.graphs.graphs import Graph

            nodes_match = re.search(r"nodes=\{([^}]+)\}", text)
            edges_match = re.search(r"edges=\[([^\]]+)\]", text)
            if not nodes_match or not edges_match:
                return None
            nodes = {int(n) for n in re.findall(r"\d+", nodes_match.group(1))}
            edges = []
            for pair in re.findall(r"\((\d+)\s*,\s*(\d+)\)", edges_match.group(1)):
                edges.append((int(pair[0]), int(pair[1])))
            return Graph(nodes=nodes, edges=set(edges), directed=False)

        def _parse_angle(text):
            text = text.replace("\u03c0", "pi")
            if "degree" in text:
                deg_match = re.search(r"(-?\d+\.?\d*)\s*degrees?", text)
                if deg_match:
                    return math.radians(float(deg_match.group(1)))
            if "pi" in text:
                pi_match = re.search(r"(\d+)?\s*\*?\s*pi(?:\s*/\s*(\d+))?", text)
                if pi_match:
                    numerator = float(pi_match.group(1)) if pi_match.group(1) else 1.0
                    denominator = float(pi_match.group(2)) if pi_match.group(2) else 1.0
                    return math.pi * numerator / denominator
            num_match = re.search(r"(-?\d+\.?\d*)", text)
            if num_match:
                return float(num_match.group(1))
            return None

        if tool_name == "arithmetic":
            expr_candidates = re.findall(
                r"(?:\bpi\b|\be\b|[0-9\.\+\-\*/%\^\(\)]+|\s)+",
                query,
                re.IGNORECASE,
            )
            expr_candidates = [
                c.strip()
                for c in expr_candidates
                if re.search(r"(\d|pi|e)", c, re.IGNORECASE)
                and re.search(r"[\+\-\*/%\^]", c)
            ]
            if expr_candidates:
                return max(expr_candidates, key=len).strip(" .?")
            if self.context.last_numeric is not None:
                query_lower = query.lower()
                num_match = re.search(r"-?\d+\.?\d*", query_lower)
                if num_match:
                    value = float(num_match.group(0))
                    base = self.context.last_numeric
                    if "add" in query_lower or "plus" in query_lower:
                        return f"{base} + {value}"
                    if "subtract" in query_lower or "minus" in query_lower:
                        return f"{base} - {value}"
                    if "multiply" in query_lower or "times" in query_lower:
                        return f"{base} * {value}"
                    if "divide" in query_lower or "over" in query_lower:
                        return f"{base} / {value}"
            return query.strip()

        if tool_name == "physics_dimensions":
            unit_match = re.search(
                r"(?:dimension(?:s)?)\s+(?:of|for)\s+([A-Za-z][A-Za-z0-9/\^\-\*]+)",
                query,
                re.IGNORECASE,
            )
            if unit_match:
                return unit_match.group(1).strip()
            unit_match = re.search(r"\d+\.?\d*\s*([a-zA-Z/^\-\d]+)", query)
            if unit_match:
                return unit_match.group(1).strip()
            return query.strip()

        if tool_name == "physics_convert":
            return query.strip()

        if tool_name in [
            "biology_gc_content",
            "biology_mw",
            "biology_complement",
            "biology_reverse_complement_gc",
        ]:
            # Look for DNA sequence after "of" or "for"
            dna_match = re.search(
                r"(?:of|for)\s+([ATGCNatgcu]{3,})", query, re.IGNORECASE
            )
            if dna_match:
                return dna_match.group(1).upper()
            # Fallback: find longest DNA sequence
            matches = re.findall(r"[ATGCNatgcu]{3,}", query)
            if matches:
                return max(matches, key=len).upper()
            if self.context.last_sequence is not None:
                return self.context.last_sequence
            return query.strip()

        if tool_name in ["math_vector", "math_normalize"]:
            vector = _parse_vector(query)
            if vector is not None:
                return vector
            if self.context.last_vector is not None:
                return self.context.last_vector
            return query.strip()

        if tool_name == "math_dot":
            vectors = _parse_vectors(query)
            if vectors is not None:
                return vectors
            if len(self.context.last_vectors) >= 2:
                return self.context.last_vectors[:2]
            return query.strip()

        if tool_name == "math_determinant":
            matrix = _parse_matrix(query)
            if matrix is not None:
                return matrix
            if self.context.last_matrix is not None:
                return self.context.last_matrix
            return query.strip()

        if tool_name in ["music_prime_form", "music_normal_form", "music_chord_type"]:
            vector = _parse_vector(query)
            return [int(v) for v in vector] if vector is not None else query.strip()

        if tool_name == "music_transpose":
            vector = _parse_vector(query)
            semitone_match = re.search(r"by\s+(-?\d+)\s+semitones?", query.lower())
            semitones = int(semitone_match.group(1)) if semitone_match else 0
            return (vector if vector is not None else [], semitones)

        if tool_name in ["geometry_distance", "geometry_midpoint"]:
            vectors = _parse_vectors(query)
            if vectors:
                return vectors
            if "origin" in query.lower():
                single = _parse_vector(query)
                if single is not None:
                    return [[0.0, 0.0], single]
                if self.context.last_vector is not None:
                    return [[0.0, 0.0], self.context.last_vector]
            if len(self.context.last_vectors) >= 2:
                return self.context.last_vectors[:2]
            return query.strip()

        if tool_name == "geometry_rotate":
            vector = _parse_vector(query)
            angle = _parse_angle(query.lower())
            if vector is not None and angle is not None:
                return (vector, angle)
            if vector is None and angle is not None and self.context.last_vector is not None:
                return (self.context.last_vector, angle)
            return query.strip()

        if tool_name in [
            "sets_union",
            "sets_intersection",
            "sets_subset",
            "sets_complement",
        ]:
            sets = _parse_sets(query)
            if sets is not None:
                return sets
            if len(self.context.last_sets) >= 2:
                return self.context.last_sets[:2]
            return query.strip()

        if tool_name in ["data_array_summary", "data_array_length"]:
            data = _parse_data_array(query)
            if data is not None:
                return data
            if self.context.last_array is not None:
                return self.context.last_array
            return query.strip()

        if tool_name == "data_record_schema":
            record = _parse_data_record(query)
            if record is not None:
                return record
            if self.context.last_record is not None:
                return self.context.last_record
            return query.strip()

        if tool_name == "data_table_summary":
            table = _parse_data_table(query)
            if table is not None:
                return table
            if self.context.last_table is not None:
                return self.context.last_table
            return query.strip()

        if tool_name in [
            "combinatorics_factorial",
            "combinatorics_ncr",
            "combinatorics_npr",
            "combinatorics_multiset",
        ]:
            nums = [int(n) for n in re.findall(r"-?\d+", query)]
            return nums if nums else query.strip()

        if tool_name in [
            "probability_bernoulli_pmf",
            "probability_binomial_pmf",
            "probability_bayes_rule",
        ]:
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            return nums if nums else query.strip()

        if tool_name in [
            "statistics_mean",
            "statistics_median",
            "statistics_variance",
            "info_entropy",
            "signal_dft_magnitude",
            "calculus_derivative",
            "calculus_integral",
        ]:
            values = _parse_data_array(query)
            if values is not None:
                return values
            if self.context.last_array is not None:
                return self.context.last_array
            return query.strip()

        if tool_name in ["statistics_corr", "info_cross_entropy", "info_kl_divergence"]:
            lists = _parse_list_literals(query, max_items=2)
            if len(lists) >= 2:
                return lists[:2]
            if len(self.context.last_vectors) >= 2:
                return self.context.last_vectors[:2]
            return query.strip()

        if tool_name == "signal_convolution":
            lists = _parse_list_literals(query, max_items=2)
            if len(lists) >= 2:
                return lists[:2]
            return query.strip()

        if tool_name == "signal_moving_average":
            lists = _parse_list_literals(query, max_items=1)
            window_match = re.search(r"window\s*(\d+)", query.lower())
            window = int(window_match.group(1)) if window_match else None
            if lists and window is not None:
                return [lists[0], window]
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if len(nums) >= 2 and lists:
                return [lists[0], int(nums[-1])]
            if self.context.last_array is not None and nums:
                return [self.context.last_array, int(nums[-1])]
            return query.strip()

        if tool_name == "calculus_evaluate":
            lists = _parse_list_literals(query, max_items=1)
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if lists and nums:
                x_val = nums[-1]
                return [lists[0], x_val]
            return query.strip()

        if tool_name in ["temporal_add_days", "temporal_diff_days", "temporal_day_of_week"]:
            dates = _parse_dates(query)
            query_for_nums = query
            for date_str in dates:
                query_for_nums = query_for_nums.replace(date_str, "")
            nums = [int(n) for n in re.findall(r"-?\d+", query_for_nums)]
            if tool_name == "temporal_add_days":
                if dates and nums:
                    return [dates[0], nums[-1]]
                if self.context.last_date and nums:
                    return [self.context.last_date, nums[-1]]
            if tool_name == "temporal_diff_days":
                if len(dates) >= 2:
                    return [dates[0], dates[1]]
                if len(dates) == 1 and self.context.last_date:
                    return [self.context.last_date, dates[0]]
            if tool_name == "temporal_day_of_week":
                if dates:
                    return [dates[0]]
                if self.context.last_date:
                    return [self.context.last_date]
            return query.strip()

        if tool_name == "finance_compound_interest":
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if len(nums) >= 3:
                return nums[:4]
            return query.strip()

        if tool_name == "finance_npv":
            lists = _parse_list_literals(query, max_items=1)
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if lists and nums:
                return [nums[0], lists[0]]
            if len(nums) >= 2:
                return [nums[0], nums[1:]]
            return query.strip()

        if tool_name == "finance_payment":
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if len(nums) >= 3:
                return nums[:4]
            return query.strip()

        if tool_name == "optimization_least_squares_2x2":
            lists = _parse_list_literals(query, max_items=3)
            if len(lists) >= 2:
                return [lists[0], lists[1]]
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if len(nums) >= 6:
                return [[nums[0], nums[1], nums[2], nums[3]], [nums[4], nums[5]]]
            return query.strip()

        if tool_name == "optimization_gradient_step":
            lists = _parse_list_literals(query, max_items=2)
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if len(lists) >= 2 and nums:
                return [lists[0], lists[1], nums[-1]]
            return query.strip()

        if tool_name == "optimization_project_box":
            lists = _parse_list_literals(query, max_items=1)
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if lists and len(nums) >= 2:
                return [lists[0], nums[-2], nums[-1]]
            return query.strip()

        if tool_name == "control_state_update":
            dict_blob = _extract_balanced(query, "{", "}")
            if dict_blob:
                parsed = _parse_literal(dict_blob)
                if isinstance(parsed, dict):
                    return parsed
            lists = _parse_list_literals(query, max_items=4)
            if len(lists) >= 4:
                return lists[:4]
            return query.strip()

        if tool_name == "control_is_stable_2x2":
            lists = _parse_list_literals(query, max_items=2)
            if len(lists) >= 2 and all(len(row) == 2 for row in lists[:2]):
                return lists[:2]
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
            if len(nums) >= 4:
                return [[nums[0], nums[1]], [nums[2], nums[3]]]
            return query.strip()

        if tool_name in [
            "graphs_shortest_path",
            "graphs_properties",
            "graphs_node_count",
            "graphs_is_connected",
            "graphs_is_tree",
        ]:
            graph = _parse_graph(query)
            if graph is None:
                return query.strip()
            if tool_name == "graphs_shortest_path":
                match = re.search(
                    r"from\s+node\s+(\d+)\s+to\s+node\s+(\d+)", query.lower()
                )
                if not match:
                    match = re.search(r"from\s+(\d+)\s+to\s+(\d+)", query.lower())
                if match:
                    return (graph, int(match.group(1)), int(match.group(2)))
            return graph

        if tool_name == "logic_tautology":
            if "equivalent" in query.lower() or "implies" in query.lower():
                return query.strip()
            formula_match = re.search(r'["\']([^"\']+)["\']', query)
            return formula_match.group(1) if formula_match else query.strip()

        if tool_name == "number_theory_is_prime":
            num_match = re.search(r"(-?\d+)", query)
            if num_match:
                return int(num_match.group(1))
            if self.context.last_numeric is not None:
                return int(round(self.context.last_numeric))
            return query.strip()

        if tool_name == "number_theory_gcd":
            nums = [int(n) for n in re.findall(r"-?\d+", query)]
            return nums if nums else query.strip()

        if tool_name == "number_theory_mod_pow":
            nums = [int(n) for n in re.findall(r"-?\d+", query)]
            if len(nums) >= 3:
                return [nums[0], nums[1], nums[2]]
            return query.strip()

        if tool_name == "number_theory_mod_inverse":
            nums = [int(n) for n in re.findall(r"-?\d+", query)]
            if len(nums) >= 2:
                return [nums[0], nums[1]]
            return query.strip()

        if tool_name == "number_theory_primes_up_to":
            num_match = re.search(
                r"up to (\d+)|all primes.*?(\d+)", query, re.IGNORECASE
            )
            if num_match:
                n = int(num_match.group(1) or num_match.group(2))
                return n
            num_match = re.search(r"(-?\d+)", query)
            return int(num_match.group(1)) if num_match else query.strip()

        if tool_name == "number_theory_nth_prime":
            num_match = re.search(
                r"(\d+)(?:st|nd|rd|th)?\s*prime", query, re.IGNORECASE
            )
            if num_match:
                return int(num_match.group(1))
            num_match = re.search(r"(-?\d+)", query)
            return int(num_match.group(1)) if num_match else query.strip()

        if tool_name == "chemistry_formula":
            name_match = re.search(
                r"formula\s+of\s+([a-zA-Z\s]+)\??", query, re.IGNORECASE
            )
            if name_match:
                return name_match.group(1).strip().lower()
            return query.strip().lower()

        if tool_name == "chemistry_molecule":
            candidates = re.findall(
                r"[A-Z][A-Za-z0-9]*(?:\([A-Za-z0-9]+\)\d*)*",
                query,
            )
            filtered = []
            for candidate in candidates:
                if not candidate:
                    continue
                upper_count = sum(1 for ch in candidate if ch.isupper())
                if re.search(r"\d", candidate) or "(" in candidate or upper_count >= 2:
                    filtered.append(candidate)
            if filtered:
                return max(filtered, key=len)
            return query.strip()

        if tool_name == "chemistry_balance_simple":
            if ":" in query:
                return query.split(":", 1)[1].strip()
            return query.strip()

        return query.strip()

    def _split_prompt(self, prompt: str) -> List[str]:
        import re

        normalized = " ".join(prompt.strip().split())
        if not normalized:
            return []
        return [
            part.strip()
            for part in re.split(r"(?<=[\.\?\!])\s+", normalized)
            if part.strip()
        ]

    def _is_actionable_segment(self, segment: str) -> bool:
        import re

        segment_lower = segment.lower()
        if "?" in segment:
            return True
        if re.search(r"\d+\s*(?:\+|-|\*|/|\*\*|%)\s*\d+", segment_lower):
            return True
        keywords = [
            "calculate",
            "compute",
            "find",
            "determine",
            "solve",
            "evaluate",
            "what",
            "convert",
            "gc content",
            "prime",
            "gcd",
            "dot",
            "magnitude",
            "norm",
            "distance",
            "midpoint",
            "rotate",
            "determinant",
            "normalize",
            "array",
            "list",
            "table",
            "record",
            "schema",
            "column",
            "row",
            "mean",
            "average",
            "sum",
            "min",
            "max",
            "std",
            "factorial",
            "combination",
            "permutation",
            "probability",
            "entropy",
            "convolution",
            "moving average",
            "dft",
            "derivative",
            "integral",
            "day of week",
            "date",
            "npv",
            "payment",
            "interest",
            "gradient",
            "least squares",
            "stability",
        ]
        return any(kw in segment_lower for kw in keywords)

    def _extract_vector_literal(self, text: str) -> Optional[str]:
        import re

        matches = re.findall(r"\[([^\]]+)\]", text)
        for match in reversed(matches):
            values = re.findall(r"-?\d+\.?\d*", match)
            if len(values) >= 2:
                return "[" + ", ".join(values) + "]"
        return None

    def _extract_balanced(self, text: str, open_char: str, close_char: str) -> Optional[str]:
        start = text.find(open_char)
        if start == -1:
            return None
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _parse_literal(self, text: str) -> Optional[Any]:
        if not text:
            return None
        try:
            import json

            return json.loads(text)
        except Exception:
            pass
        try:
            import ast

            return ast.literal_eval(text)
        except Exception:
            return None

    def _parse_inline_array(self, text: str) -> Optional[List[Any]]:
        import re

        blob = self._extract_balanced(text, "[", "]")
        if blob:
            parsed = self._parse_literal(blob)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return [float(n) for n in numbers]
        return None

    def _parse_inline_record(self, text: str) -> Optional[Dict[str, Any]]:
        blob = self._extract_balanced(text, "{", "}")
        if blob:
            parsed = self._parse_literal(blob)
            if isinstance(parsed, dict):
                return parsed
        return None

    def _parse_inline_table(self, text: str) -> Optional[Any]:
        blob = self._extract_balanced(text, "[", "]") or self._extract_balanced(text, "{", "}")
        if blob:
            parsed = self._parse_literal(blob)
            if isinstance(parsed, (list, dict, tuple)):
                return parsed
        return None

    def _update_context_from_text(self, text: str) -> None:
        import re

        text_lower = text.lower()

        vector_literal = self._extract_vector_literal(text)
        if vector_literal:
            values = re.findall(r"-?\d+\.?\d*", vector_literal)
            if values:
                self.context.last_vector = [float(v) for v in values]

        array_literal = self._parse_inline_array(text)
        if array_literal is not None:
            if any(kw in text_lower for kw in ["array", "list"]) or len(array_literal) > 2:
                self.context.last_array = array_literal

        set_matches = re.findall(r"\{([^}]+)\}", text)
        if set_matches:
            sets = []
            for set_text in set_matches[:2]:
                values = re.findall(r"-?\d+", set_text)
                if values:
                    sets.append([int(v) for v in values])
            if sets:
                self.context.last_sets = sets
                self.context.last_set = sets[-1]

        dna_match = re.findall(r"\b[ATGCU]{4,}\b", text, re.IGNORECASE)
        if dna_match:
            self.context.last_sequence = max(dna_match, key=len).upper()

        date_match = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        if date_match:
            self.context.last_date = date_match[-1]

        if any(kw in text_lower for kw in ["array", "list"]):
            arr = self._parse_inline_array(text)
            if arr is not None:
                self.context.last_array = arr

        if "record" in text_lower or "schema" in text_lower:
            record = self._parse_inline_record(text)
            if record is not None:
                self.context.last_record = record

        if "table" in text_lower:
            table = self._parse_inline_table(text)
            if table is not None:
                self.context.last_table = table

    def _update_context(
        self, tool_name: str, parsed_query: Any, result: Any
    ) -> None:
        import re

        if isinstance(result, (int, float)) and not isinstance(result, bool):
            self.context.last_numeric = float(result)

        if tool_name in ["math_vector", "math_normalize"]:
            if isinstance(parsed_query, list):
                self.context.last_vector = [float(v) for v in parsed_query]

        if tool_name == "math_dot":
            if isinstance(parsed_query, list) and len(parsed_query) == 2:
                if all(isinstance(vec, list) for vec in parsed_query):
                    self.context.last_vectors = parsed_query
                    self.context.last_vector = parsed_query[-1]

        if tool_name == "math_determinant":
            if isinstance(parsed_query, list):
                self.context.last_matrix = parsed_query

        if tool_name in ["geometry_distance", "geometry_midpoint"]:
            if isinstance(parsed_query, list) and len(parsed_query) == 2:
                if all(isinstance(vec, list) for vec in parsed_query):
                    self.context.last_vectors = parsed_query
                    self.context.last_vector = parsed_query[-1]

        if tool_name == "geometry_rotate":
            if isinstance(parsed_query, tuple) and parsed_query and isinstance(parsed_query[0], list):
                self.context.last_vector = parsed_query[0]

        if tool_name.startswith("sets_"):
            if isinstance(parsed_query, list) and parsed_query:
                if all(isinstance(vec, list) for vec in parsed_query):
                    self.context.last_sets = parsed_query
                    self.context.last_set = parsed_query[-1]
            if isinstance(result, list):
                self.context.last_set = result

        if tool_name.startswith("biology_"):
            if isinstance(parsed_query, str):
                self.context.last_sequence = parsed_query.upper()
            if isinstance(result, str) and re.search(r"^[ATGCU]+$", result):
                self.context.last_sequence = result.upper()

        if tool_name.startswith("data_array"):
            if isinstance(parsed_query, list):
                self.context.last_array = parsed_query

        if tool_name == "data_record_schema":
            if isinstance(parsed_query, dict):
                self.context.last_record = parsed_query

        if tool_name == "data_table_summary":
            if isinstance(parsed_query, (list, dict, tuple)):
                self.context.last_table = parsed_query

        if tool_name.startswith("statistics_") or tool_name.startswith("info_"):
            if isinstance(parsed_query, list):
                self.context.last_array = parsed_query

        if tool_name.startswith("signal_"):
            if isinstance(parsed_query, list):
                self.context.last_array = parsed_query[0] if parsed_query else parsed_query
            if isinstance(result, list):
                self.context.last_array = result

        if tool_name.startswith("calculus_"):
            if isinstance(parsed_query, list):
                if parsed_query and isinstance(parsed_query[0], list):
                    self.context.last_array = parsed_query[0]
                else:
                    self.context.last_array = parsed_query

        if tool_name.startswith("temporal_"):
            if isinstance(parsed_query, list) and parsed_query:
                if isinstance(parsed_query[0], str):
                    self.context.last_date = parsed_query[0]
            if isinstance(result, str) and re.match(r"\\d{4}-\\d{2}-\\d{2}", result):
                self.context.last_date = result

        if tool_name.startswith("number_theory"):
            if isinstance(parsed_query, int):
                self.context.last_numeric = float(parsed_query)

    def _augment_prompt_with_context(
        self, segment: str, context: Dict[str, Optional[str]]
    ) -> str:
        import re

        vector_literal = context.get("vector")
        if not vector_literal and self.context.last_vector:
            vector_literal = "[" + ", ".join(str(v) for v in self.context.last_vector) + "]"
        if vector_literal and not self._extract_vector_literal(segment):
            if re.search(r"\b(vector|magnitude|norm|dot|cross|normalize)\b", segment.lower()):
                return f"{segment} Vector {vector_literal}."
        return segment

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, (dict, list, tuple)):
            import json

            try:
                return json.dumps(value, ensure_ascii=True)
            except Exception:
                return str(value)
        return str(value)

    def _format_response_text(
        self,
        result: Any,
        error: Optional[str] = None,
        index: Optional[int] = None,
    ) -> str:
        prefix = f"{index}) " if index is not None else ""
        if self.response_style == "plain":
            if error:
                return f"{prefix}{error}"
            return f"{prefix}{self._format_value(result)}"

        if error:
            return f"{prefix}Error: {error}"
        formatted = self._format_value(result)
        if index is None:
            return f"Answer: {formatted}"
        return f"{prefix}{formatted}"

    def _generate_with_tools_single(
        self, prompt: str, response_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response using FluxEM tools.

        Args:
            prompt: User's question or query

        Returns:
            Generated response with tool usage information
        """
        total_start = time.time()

        domain, tool_info = self._call_tool_auto(prompt)

        tool_execution_time = tool_info.get("execution_time_ms", 0.0)
        tool_result = tool_info.get("result")
        error = tool_info.get("error") if not tool_info.get("success") else None
        if tool_info.get("success") is False and not error:
            error = "Tool call failed"

        # Return tool result directly as response (no LLM regeneration)
        response_text = self._format_response_text(
            tool_result, error=error, index=response_index
        )

        total_time = (time.time() - total_start) * 1000

        return {
            "domain": domain,
            "tool_name": tool_info.get("tool_name"),
            "tool_success": tool_info.get("success", False),
            "result": tool_result,
            "error": error,
            "execution_time_ms": tool_execution_time,
            "total_time_ms": total_time,
            "response": response_text,
        }

    def generate_with_tools(self, prompt: str) -> Dict[str, Any]:
        segments = self._split_prompt(prompt)
        if len(segments) <= 1:
            self._update_context_from_text(prompt)
            return self._generate_with_tools_single(prompt)

        total_start = time.time()
        context: Dict[str, Optional[str]] = {"vector": None}
        if self.context.last_vector:
            context["vector"] = "[" + ", ".join(str(v) for v in self.context.last_vector) + "]"
        sub_results = []
        responses = []
        tool_time_total = 0.0

        for segment in segments:
            self._update_context_from_text(segment)
            vector_literal = self._extract_vector_literal(segment)
            if vector_literal:
                context["vector"] = vector_literal

            augmented = self._augment_prompt_with_context(segment, context)

            if not self._is_actionable_segment(segment):
                continue

            single_result = self._generate_with_tools_single(
                augmented, response_index=len(sub_results) + 1
            )
            sub_results.append(
                {
                    "prompt": segment,
                    "augmented_prompt": augmented,
                    "result": single_result,
                }
            )
            responses.append(single_result.get("response", ""))
            tool_time_total += single_result.get("execution_time_ms", 0.0)

        if not sub_results:
            return self._generate_with_tools_single(prompt)

        if len(sub_results) == 1:
            result = sub_results[0]["result"]
            result["response"] = self._format_response_text(
                result.get("result"),
                error=result.get("error"),
                index=None,
            )
            return result

        total_time = (time.time() - total_start) * 1000
        domains = [item["result"].get("domain") for item in sub_results]
        tool_names = [item["result"].get("tool_name") for item in sub_results]
        tool_successes = [bool(item["result"].get("tool_success")) for item in sub_results]
        results = [item["result"].get("result") for item in sub_results]
        errors = [item["result"].get("error") for item in sub_results if item["result"].get("error")]
        combined_response = "\n".join(r for r in responses if r)

        return {
            "domain": domains[0] if domains else "none",
            "domains": domains,
            "tool_name": "multiple",
            "tool_names": tool_names,
            "tool_success": all(tool_successes) if tool_successes else False,
            "tool_successes": tool_successes,
            "result": results,
            "error": None
            if all(tool_successes)
            else (errors[0] if errors else "One or more tool calls failed"),
            "errors": errors,
            "execution_time_ms": tool_time_total,
            "total_time_ms": total_time,
            "response": combined_response,
            "sub_results": sub_results,
            "multi_turn": True,
        }

    def generate_baseline(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response WITHOUT using FluxEM tools (baseline).

        Args:
            prompt: User's question or query

        Returns:
            Generated response from LLM without tool assistance
        """
        start_time = time.time()
        if self.model and self.is_loaded:
            # Simple prompt without tool context
            baseline_prompt = f"""You are a helpful AI assistant.
Question: {prompt}
Provide your answer:"""

            response = self._generate(baseline_prompt, max_tokens=self.max_tokens)
        else:
            # Fallback when model not loaded
            response = "Model not loaded. Baseline response unavailable."

        total_time = (time.time() - start_time) * 1000
        return {"response": response, "time_ms": total_time}

    def _format_tool_descriptions(self) -> str:
        """Format all tool descriptions for the LLM prompt."""
        descriptions = []
        for tool_name, tool_desc in self.tool_registry.items():
            descriptions.append(
                f"{tool_name}: {tool_desc.description}\n"
                f"  Input format: {tool_desc.input_format}\n"
                f"  Output format: {tool_desc.output_format}\n"
                f"  Example: {tool_desc.example}\n"
            )
        return "\n".join(descriptions)

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text using the loaded MLX model.
        """
        if self.is_loaded and self.backend == "mlx" and self.model and self.tokenizer:
            try:
                from mlx_lm import generate

                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                return response.strip()
            except Exception as e:
                if self.verbose:
                    print(f"Generation error (mlx): {e}")

        if (
            self.is_loaded
            and self.backend == "transformers"
            and self.transformers_model
            and self.transformers_tokenizer
        ):
            try:
                if hasattr(self.transformers_tokenizer, "apply_chat_template"):
                    prompt = self.transformers_tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                return self._generate_transformers(prompt, max_tokens)
            except Exception as e:
                if self.verbose:
                    print(f"Generation error (transformers): {e}")

        return f"I need more context to answer: {prompt[:100]}..."

    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        import torch

        inputs = self.transformers_tokenizer(prompt, return_tensors="pt")
        device = self.transformers_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.transformers_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.transformers_tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        response = self.transformers_tokenizer.decode(
            generated, skip_special_tokens=True
        )
        return response.strip()

    def _generate_with_messages(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        if (
            self.is_loaded
            and self.backend == "transformers"
            and self.transformers_model
            and self.transformers_tokenizer
            and hasattr(self.transformers_tokenizer, "apply_chat_template")
        ):
            prompt = self.transformers_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self._generate_transformers(prompt, max_tokens)

        fallback = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages
        )
        return self._generate(fallback, max_tokens)

    def _simulated_math_response(self, prompt: str) -> str:
        """Simulate math operations for testing."""
        import re

        if "dot product" in prompt.lower():
            vectors = re.findall(r"\[([^\]]+)\]", prompt)
            if len(vectors) >= 2:
                v1 = [float(v) for v in re.findall(r"-?\d+\.?\d*", vectors[0])]
                v2 = [float(v) for v in re.findall(r"-?\d+\.?\d*", vectors[1])]
                if len(v1) == len(v2):
                    result = sum(a * b for a, b in zip(v1, v2))
                    return f"The dot product is {result}"
        if "determinant" in prompt.lower():
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", prompt)]
            if len(nums) >= 4:
                a, b, c, d = nums[:4]
                det = a * d - b * c
                return f"The determinant is {det}"
        return "I need the math tool to answer this."

    def _simulated_arithmetic_response(self, prompt: str) -> str:
        """Simulate arithmetic computation for testing."""
        import re

        expr_match = re.search(r"(\d+\.?\d*\s*[\+\-\*/\/\^%]\s*\d+\.?\d*)", prompt)
        if expr_match:
            expr = expr_match.group(1)
            try:
                result = eval(expr)
                return f"The result of {expr} is {result}"
            except:
                return f"I couldn't compute {expr}"
        return "I need to use the arithmetic tool to calculate this accurately."

    def _simulated_conversion_response(self, prompt: str) -> str:
        """Simulate unit conversion for testing."""
        import re

        match = re.search(r"(\d+\.?\d*)\s*km\s+to\s+meters", prompt)
        if match:
            value = float(match.group(1))
            return f"{value} km is equal to {value * 1000:.0f} meters"
        return "I need the physics conversion tool to handle unit conversions."

    def _simulated_gc_response(self, prompt: str) -> str:
        """Simulate GC content calculation for testing."""
        import re

        dna_match = re.search(r"[ATGCNatgcun]+", prompt.upper())
        if dna_match:
            seq = dna_match.group(0)
            gc_count = seq.count("G") + seq.count("C")
            gc_content = gc_count / len(seq)
            return f"The GC content of {seq} is {gc_content:.1%}"
        return "I need the biology GC content tool to calculate this."

    def _simulated_music_response(self, prompt: str) -> str:
        """Simulate pitch class operations for testing."""
        import re

        pcs_match = re.search(
            r"pitch\s+class(?:es)?\s*\[?\s*([\d\s*,\s*]*)\s*\]?\]", prompt.lower()
        )
        if pcs_match:
            pcs_str = pcs_match.group(1)
            numbers = [int(n.strip()) for n in pcs_str.split(",") if n.strip()]
            if "prime" in prompt.lower() or "normal" in prompt.lower():
                return f"Prime/normal form: {numbers}"
            return f"Pitch class: {numbers}"
        return "I need the music theory tool to analyze this properly."

    def _simulated_geometry_response(self, prompt: str) -> str:
        """Simulate geometric calculations for testing."""
        import re
        import math

        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", prompt)]
        if "rotate" in prompt.lower() and len(numbers) >= 3:
            x, y, angle = numbers[:3]
            cos_a = math.cos(math.radians(angle))
            sin_a = math.sin(math.radians(angle))
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            return f"Rotated point: [{rx:.2f}, {ry:.2f}]"
        if len(numbers) >= 4:
            p1 = numbers[:2]
            p2 = numbers[2:4]
            distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            return f"The distance between these points is approximately {distance:.2f}"
        if "origin" in prompt.lower() and len(numbers) >= 2:
            distance = (numbers[0] ** 2 + numbers[1] ** 2) ** 0.5
            return f"The distance from the origin is approximately {distance:.2f}"
        return "I need the geometry tool to calculate this accurately."

    def _simulated_sets_response(self, prompt: str) -> str:
        """Simulate set operations for testing."""
        import re

        sets_match = re.findall(r"\{([^}]+)\}", prompt)
        if len(sets_match) >= 2:
            set1 = {int(v) for v in re.findall(r"-?\d+", sets_match[0])}
            set2 = {int(v) for v in re.findall(r"-?\d+", sets_match[1])}
            if "union" in prompt.lower():
                return f"The union of these sets is {sorted(set1 | set2)}"
            if "intersection" in prompt.lower():
                return f"The intersection of these sets is {sorted(set1 & set2)}"
            if "subset" in prompt.lower():
                return f"{set1.issubset(set2)}"
        return "I need the sets tool to handle this properly."

    def _simulated_logic_response(self, prompt: str) -> str:
        """Simulate logic evaluation for testing."""
        import re

        formula_match = re.search(
            r'is\s+["]?([^"]+)["]?\s+a\s+tautology', prompt.lower()
        )
        if formula_match:
            formula = formula_match.group(1)
            if "not" in formula and "or" in formula:
                return f"Yes, '{formula}' is a tautology because (p or not p) is always true."
        return "I need the logic tool to evaluate this properly."

    def _simulated_number_theory_response(self, prompt: str) -> str:
        """Simulate number theory computations for testing."""
        import re

        if "prime" in prompt.lower():
            num_match = re.search(r"is\s+(\d+)\s+a\s+prime", prompt)
            if num_match:
                n = int(num_match.group(1))
                is_prime = all(n % i != 0 for i in range(2, int(n**0.5) + 1))
                return f"{n} is {'prime' if is_prime else 'not prime'}"
        elif "gcd" in prompt.lower():
            nums_match = re.search(r"gcd\s+of\s*(\d+)\s+and\s+(\d+)", prompt)
            if nums_match:
                from math import gcd

                a, b = int(nums_match.group(1)), int(nums_match.group(2))
                result = gcd(a, b)
                return f"The GCD of {a} and {b} is {result}"
        return "I need the number theory tool to compute this accurately."

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "backend": self.backend,
            "mlx_available": _ensure_mlx_imported(),
            "use_thinking": self.use_thinking,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tool_selection": self.tool_selection,
            "llm_query_extraction": self.llm_query_extraction,
            "response_style": self.response_style,
            "transformers_model_path": self.transformers_model_path,
            "transformers_device": self.transformers_device,
            "num_tools": len(self.tool_registry),
            "domains": list(self.tool_registry.keys()),
        }

    def reset_context(self) -> None:
        """Reset stored tool context for multi-turn sessions."""
        self.context.reset()


def create_wrapper(
    model_path: Optional[str] = None,
    use_thinking: bool = True,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    tool_selection: str = "pattern",
    llm_query_extraction: bool = True,
    response_style: str = "structured",
    transformers_model_path: Optional[str] = None,
    transformers_device: str = "cpu",
    transformers_trust_remote_code: bool = False,
    transformers_local_files_only: bool = True,
    verbose: bool = False,
) -> Qwen3MLXWrapper:
    """
    Convenience function to create a Qwen3 wrapper.

    Args:
        model_path: Path to MLX model
        use_thinking: Whether to enable thinking mode
        verbose: Print debug information

    Returns:
        Initialized Qwen3MLXWrapper instance
    """
    return Qwen3MLXWrapper(
        model_path=model_path,
        use_thinking=use_thinking,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_selection=tool_selection,
        llm_query_extraction=llm_query_extraction,
        response_style=response_style,
        transformers_model_path=transformers_model_path,
        transformers_device=transformers_device,
        transformers_trust_remote_code=transformers_trust_remote_code,
        transformers_local_files_only=transformers_local_files_only,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Demo the wrapper
    print("Qwen3-4B MLX Wrapper Demo")
    print("=" * 50)

    wrapper = create_wrapper(verbose=True)

    # Try to load model
    loaded = wrapper.load_model()
    if not loaded:
        print("\nNote: Model loading requires MLX Qwen3-4B model files.")
        print("This is a demonstration wrapper with simulated responses.")
        print("\nTo use actual Qwen3-4B MLX models:")
        print("1. Download Qwen3-4B MLX model from Hugging Face")
        print("2. Install mlx-lm package: pip install mlx-lm")
        print("3. Update model_path to the downloaded model path")

    print("\nModel Info:")
    print(wrapper.get_model_info())

    # Demo tool calling with simulated responses
    print("\n" + "=" * 50)
    print("Demo: Tool Calling")
    print("=" * 50)

    test_queries = [
        "What is 54 * 44?",
        "Calculate 2**16",
        "Convert 5 km to meters",
        "What's the GC content of GATTACA?",
        "What is the prime form of [0, 4, 7]?",
        "What's the distance between [0, 0] and [3, 4]?",
        "What is the union of {1, 2, 3} and {2, 3, 4}?",
        "Is 'p or not p' a tautology?",
        "What is the GCD of 12 and 18?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = wrapper.generate_with_tools(query)
        print(f"Response: {response}")
