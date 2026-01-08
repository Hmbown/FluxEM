"""
Run official LLM benchmarks with FluxEM + Qwen3 tool-calling.

Supports GSM8K and Hendrycks MATH datasets via Hugging Face datasets.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(1, str(Path(__file__).parent.parent.parent))

from .qwen3_wrapper import create_wrapper, Qwen3MLXWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official benchmarks with FluxEM tool-calling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math"],
        default="gsm8k",
        help="Official dataset to run (default: gsm8k).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of samples (0 = full split).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/.mlx/models/Qwen/Qwen3-4B-Instruct-MLX",
        help="Path to Qwen3-4B MLX model",
    )
    parser.add_argument(
        "--transformers-model-path",
        type=str,
        default=None,
        help="Transformers model ID or local path (fallback when MLX unavailable).",
    )
    parser.add_argument(
        "--transformers-device",
        type=str,
        default="cpu",
        help="Device for transformers backend (cpu/cuda/mps).",
    )
    parser.add_argument(
        "--transformers-trust-remote-code",
        action="store_true",
        help="Allow trust_remote_code for transformers loading.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow transformers to download model files if missing locally.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for baseline generation (default: 2048).",
    )
    parser.add_argument(
        "--tool-selection",
        choices=["pattern", "llm", "hybrid"],
        default="llm",
        help="Tool selection strategy (default: llm).",
    )
    parser.add_argument(
        "--llm-query-extraction",
        action="store_true",
        help="Use LLM to refine tool queries when selecting tools via LLM.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline evaluation (LLM without tools).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/qwen3_toolcalling/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    return parser.parse_args()


def _extract_last_number(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if not matches:
        return None
    token = matches[-1]
    try:
        if "." in token:
            return float(token)
        return float(int(token))
    except ValueError:
        return None


def _parse_fraction(token: str) -> Optional[float]:
    if re.match(r"^-?\d+/\d+$", token):
        num, den = token.split("/", 1)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    frac_match = re.match(r"^\\frac\{(-?\d+)\}\{(\d+)\}$", token)
    if frac_match:
        num, den = frac_match.group(1), frac_match.group(2)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def _extract_gsm8k_answer(answer: str) -> Optional[float]:
    if not answer:
        return None
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
    frac_val = _parse_fraction(answer)
    if frac_val is not None:
        return frac_val
    return _extract_last_number(answer)


def _extract_math_answer(sample: Dict[str, Any]) -> str:
    if "answer" in sample and sample["answer"]:
        return str(sample["answer"]).strip()
    solution = sample.get("solution", "")
    match = re.search(r"\\boxed\{([^}]*)\}", solution)
    if match:
        return match.group(1).strip()
    return solution.strip()


def _extract_prediction(response: Dict[str, Any]) -> Tuple[Optional[float], str]:
    if response.get("tool_success"):
        result = response.get("result")
        if isinstance(result, (int, float)):
            return float(result), str(result)
        if isinstance(result, (list, tuple)):
            return None, str(result)
        if isinstance(result, str):
            num = _extract_last_number(result)
            return num, result
        return None, str(result)
    text = response.get("response", "")
    return _extract_last_number(text), text


def _compare_numeric(expected: Optional[float], predicted: Optional[float]) -> bool:
    if expected is None or predicted is None:
        return False
    return abs(expected - predicted) < 1e-6


def load_official_dataset(dataset_name: str, split: str):
    if dataset_name == "gsm8k":
        return load_dataset("gsm8k", "main", split=split)
    if dataset_name == "math":
        last_error = None
        try:
            configs = get_dataset_config_names("EleutherAI/hendrycks_math")
            datasets = [
                load_dataset("EleutherAI/hendrycks_math", config, split=split)
                for config in configs
            ]
            return concatenate_datasets(datasets)
        except Exception as exc:
            last_error = exc
        candidates = [
            ("hendrycks/math", "all"),
            ("hendrycks/math", None),
        ]
        for dataset_id, config in candidates:
            try:
                if config:
                    return load_dataset(dataset_id, config, split=split)
                return load_dataset(dataset_id, split=split)
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load MATH dataset: {last_error}") from last_error
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def run_benchmark(
    wrapper: Qwen3MLXWrapper,
    dataset_name: str,
    split: str,
    limit: int,
    run_baseline: bool,
    verbose: bool,
) -> Dict[str, Any]:
    dataset = load_official_dataset(dataset_name, split)
    if limit and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    results = []
    correct_tool = 0
    correct_baseline = 0
    tool_success_count = 0

    for idx, sample in enumerate(dataset):
        if dataset_name == "gsm8k":
            prompt = sample["question"]
            expected_value = _extract_gsm8k_answer(sample["answer"])
            expected_text = str(expected_value) if expected_value is not None else ""
        else:
            prompt = sample["problem"]
            expected_text = _extract_math_answer(sample)
            expected_value = _extract_last_number(expected_text)

        if verbose:
            print(f"\n[{idx + 1}/{len(dataset)}] {prompt[:80]}...")

        tool_response = wrapper.generate_with_tools(prompt)
        tool_num, tool_text = _extract_prediction(tool_response)
        tool_success = bool(tool_response.get("tool_success"))
        tool_success_count += 1 if tool_success else 0

        tool_correct = False
        if dataset_name == "gsm8k":
            tool_correct = _compare_numeric(expected_value, tool_num)
        else:
            tool_correct = _normalize_text(expected_text) == _normalize_text(tool_text)
            if not tool_correct and expected_value is not None:
                tool_correct = _compare_numeric(expected_value, tool_num)

        if tool_correct:
            correct_tool += 1

        baseline_correct = None
        baseline_response = None
        if run_baseline:
            baseline_response = wrapper.generate_baseline(prompt)
            baseline_num = _extract_last_number(baseline_response.get("response", ""))
            if dataset_name == "gsm8k":
                baseline_correct = _compare_numeric(expected_value, baseline_num)
            else:
                baseline_correct = _normalize_text(expected_text) == _normalize_text(
                    baseline_response.get("response", "")
                )
                if not baseline_correct and expected_value is not None:
                    baseline_correct = _compare_numeric(expected_value, baseline_num)
            if baseline_correct:
                correct_baseline += 1

        results.append(
            {
                "prompt": prompt,
                "expected": expected_text,
                "tool_response": tool_response,
                "tool_correct": tool_correct,
                "baseline_response": baseline_response,
                "baseline_correct": baseline_correct,
            }
        )

    total = len(results)
    tool_accuracy = (correct_tool / total * 100) if total else 0.0
    baseline_accuracy = (correct_baseline / total * 100) if total else None
    tool_success_rate = (tool_success_count / total * 100) if total else 0.0

    return {
        "dataset": dataset_name,
        "split": split,
        "total": total,
        "tool_accuracy": tool_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "tool_success_rate": tool_success_rate,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = create_wrapper(
        model_path=args.model_path,
        use_thinking=True,
        temperature=0.6,
        max_tokens=args.max_tokens,
        tool_selection=args.tool_selection,
        llm_query_extraction=args.llm_query_extraction,
        transformers_model_path=args.transformers_model_path,
        transformers_device=args.transformers_device,
        transformers_trust_remote_code=args.transformers_trust_remote_code,
        transformers_local_files_only=not args.allow_model_download,
        verbose=args.verbose,
    )

    wrapper.load_model()

    start_time = time.time()
    report = run_benchmark(
        wrapper=wrapper,
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        run_baseline=not args.no_baseline,
        verbose=args.verbose,
    )
    report["elapsed_seconds"] = time.time() - start_time
    report["model_info"] = wrapper.get_model_info()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"official_benchmark_{args.dataset}_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2))

    print("\nOfficial Benchmark Results")
    print("=" * 60)
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Samples: {report['total']}")
    print(f"Tool Accuracy: {report['tool_accuracy']:.1f}%")
    if report["baseline_accuracy"] is not None:
        print(f"Baseline Accuracy: {report['baseline_accuracy']:.1f}%")
    print(f"Tool Success Rate: {report['tool_success_rate']:.1f}%")
    print(f"Elapsed: {report['elapsed_seconds']:.1f}s")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
