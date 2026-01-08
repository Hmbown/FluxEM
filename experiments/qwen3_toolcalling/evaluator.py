"""
Evaluator for FluxEM + Qwen3-4B Tool Calling Benchmark.

Calculates comprehensive metrics comparing tool-calling performance
against baseline (LLM without tools).
"""

from typing import Dict, List, Any, Tuple, Optional
import time
import json
from dataclasses import dataclass, field


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class ToolCallResult:
    """Result from a single tool call."""
    domain: str
    tool_name: str
    success: bool
    result: Any
    error: Optional[str]
    execution_time_ms: float


@dataclass
class EvaluationResult:
    """Result from evaluating a single query."""
    prompt: str
    domain: str
    expected_domain: str
    domain_correct: bool
    
    # Tool-calling metrics
    tool_detected: bool
    tool_name: Optional[str]
    tool_success: bool
    tool_result: Any
    tool_execution_time_ms: float
    
    # Baseline metrics
    baseline_response: Optional[str]
    baseline_time_ms: Optional[float]
    baseline_correct: Optional[bool]
    
    # Answer accuracy
    answer_correct: bool
    answer_expected: Any
    
    # Timing
    total_time_ms: float


@dataclass
class DomainMetrics:
    """Aggregated metrics for a single domain."""
    domain: str
    total_queries: int
    domain_detection_accuracy: float  # % correct domain
    tool_success_rate: float  # % successful tool calls
    answer_accuracy: float  # % correct answers
    baseline_accuracy: Optional[float]  # % correct baseline answers
    avg_tool_time_ms: float
    avg_baseline_time_ms: Optional[float]
    improvement_ratio: Optional[float]  # tool-calling / baseline accuracy


# =============================================================================
# Evaluator
# =============================================================================

class Evaluator:
    """
    Evaluates FluxEM tool-calling performance vs baseline.
    
    Calculates:
    - Domain detection accuracy
    - Tool execution success rate
    - Answer accuracy
    - Response time comparison
    - Performance improvement ratios
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize evaluator.
        
        Args:
            verbose: Print detailed evaluation information
        """
        self.verbose = verbose
        self.results: List[EvaluationResult] = []
    
    def evaluate_response(
        self,
        prompt: str,
        tool_result: Optional[ToolCallResult],
        baseline_response: Optional[str],
        expected_domain: str,
        expected_answer: Any,
        tool_time_ms: float = 0.0,
        baseline_time_ms: Optional[float] = None
    ) -> EvaluationResult:
        """
        Evaluate a single query response.
        
        Args:
            prompt: User's original question
            tool_result: Result from FluxEM tool call (or None)
            baseline_response: Response from baseline LLM
            expected_domain: The correct domain for this question
            expected_answer: The correct answer
            
        Returns:
            EvaluationResult with all metrics
        """
        # Determine domain detection correctness
        tool_detected = tool_result is not None
        actual_domain = tool_result.domain if tool_detected else "none"
        domain_correct = actual_domain == expected_domain if tool_detected else False
        
        # Determine answer correctness
        answer_correct = False
        answer_expected = expected_answer
        baseline_correct: Optional[bool] = None

        tool_success = tool_result.success if tool_detected else False

        if tool_detected and tool_success:
            # Tool was called and succeeded
            if tool_result.success:
                # Compare tool result with expected
                answer_correct = self._compare_answers(tool_result.result, expected_answer)
        if baseline_response is not None:
            baseline_correct = self._compare_baseline_answer(
                baseline_response, expected_answer
            )

        return EvaluationResult(
            prompt=prompt,
            domain=expected_domain,
            expected_domain=expected_domain,
            domain_correct=domain_correct,
            tool_detected=tool_detected,
            tool_name=tool_result.tool_name if tool_detected else None,
            tool_success=tool_result.success if tool_detected else False,
            tool_result=tool_result.result if tool_detected else None,
            tool_execution_time_ms=tool_time_ms,
            baseline_response=baseline_response,
            baseline_time_ms=baseline_time_ms,
            baseline_correct=baseline_correct,
            answer_correct=answer_correct,
            answer_expected=expected_answer,
            total_time_ms=max(
                tool_time_ms, baseline_time_ms or 0.0
            )
            if tool_detected
            else (baseline_time_ms or 0.0),
        )
    
    def _compare_answers(self, actual: Any, expected: Any) -> bool:
        """
        Compare actual answer with expected answer.
        
        Supports:
        - Numeric comparisons with tolerance
        - List/set comparisons
        - String comparisons (partial match)
        - Boolean comparisons
        - Dict comparisons for complex results
        """
        if isinstance(expected, (int, float)):
            # Numeric comparison with 1% tolerance
            try:
                actual_num = float(actual)
                expected_num = float(expected)
                rel_error = abs(actual_num - expected_num) / abs(expected_num) if expected_num != 0 else abs(actual_num)
                return rel_error < 0.01  # 1% tolerance
            except (ValueError, TypeError):
                return False
        
        elif isinstance(expected, list):
            # List comparison (order-preserving for numeric vectors)
            if isinstance(actual, (list, tuple)):
                if len(actual) != len(expected):
                    return False
                if all(isinstance(x, (int, float)) for x in expected) and all(
                    isinstance(x, (int, float)) for x in actual
                ):
                    for a, e in zip(actual, expected):
                        if not self._compare_answers(a, e):
                            return False
                    return True
                return list(actual) == list(expected)
            return False

        elif isinstance(expected, set):
            # Set comparison
            try:
                actual_set = set(actual) if isinstance(actual, list) else actual
                return actual_set == expected
            except Exception:
                return False
        
        elif isinstance(expected, str):
            # String comparison
            if isinstance(actual, str):
                # Exact match
                if actual.lower().strip() == expected.lower().strip():
                    return True
                # Partial match (for more lenient grading)
                return expected.lower().strip() in actual.lower().strip()
            return False
        
        elif isinstance(expected, dict):
            # Dict comparison (check key values)
            if isinstance(actual, dict):
                for key, value in expected.items():
                    if key not in actual:
                        return False
                    if not self._compare_answers(actual[key], value):
                        return False
                return True
        
        else:
            return False
    
    def _compare_baseline_answer(self, baseline: str, expected: Any) -> bool:
        """
        Compare baseline LLM answer with expected.
        
        More lenient than tool comparison since LLM may not follow exact format.
        """
        if isinstance(expected, (int, float)):
            # Try to extract number from baseline
            import re
            numbers = re.findall(r'\d+\.?\d*', baseline)
            if numbers:
                expected_num = float(expected)
                # Check if any number is close to expected
                for num_str in numbers:
                    num = float(num_str)
                    if abs(num - expected_num) < 0.1 * abs(expected_num):
                        return True
            return False
        
        elif isinstance(expected, str):
            # Check if expected string appears in baseline
            return expected.lower().strip() in baseline.lower().strip()
        
        elif isinstance(expected, list):
            # Check if all list elements appear in baseline
            expected_strs = [str(exp) for exp in expected]
            return all(exp_str in baseline for exp_str in expected_strs)
        
        elif isinstance(expected, dict):
            # Check if dict keys/values appear in baseline
            return False  # Too complex for baseline
        
        else:
            return False
    
    def add_result(self, result: EvaluationResult):
        """Store an evaluation result."""
        self.results.append(result)
        
        if self.verbose:
            print(f"Added result: {result.prompt[:50]}...")
    
    def aggregate_metrics(self) -> Dict[str, DomainMetrics]:
        """
        Aggregate metrics by domain.
        
        Returns:
            Dictionary mapping domain names to aggregated metrics
        """
        # Group by domain
        domain_results: Dict[str, List[EvaluationResult]] = {}
        for result in self.results:
            if result.domain not in domain_results:
                domain_results[result.domain] = []
            domain_results[result.domain].append(result)
        
        metrics: Dict[str, DomainMetrics] = {}
        
        for domain, results in domain_results.items():
            if not results:
                continue
            
            total = len(results)
            
            # Domain detection accuracy
            domain_correct = sum(1 for r in results if r.domain_correct)
            domain_detection_accuracy = domain_correct / total * 100
            
            # Tool success rate
            tool_success_count = sum(1 for r in results if r.tool_success)
            tool_success_rate = tool_success_count / total * 100
            
            # Answer accuracy (for successful tool calls)
            answer_correct_count = sum(1 for r in results if r.answer_correct)
            answer_accuracy = answer_correct_count / total * 100

            # Baseline accuracy
            baseline_results = [r for r in results if r.baseline_correct is not None]
            baseline_total = len(baseline_results)
            if baseline_total:
                baseline_correct_count = sum(
                    1 for r in baseline_results if r.baseline_correct
                )
                baseline_accuracy = baseline_correct_count / baseline_total * 100
            else:
                baseline_accuracy = None
            
            # Average times
            tool_times = [r.tool_execution_time_ms for r in results if r.tool_execution_time_ms > 0]
            avg_tool_time = sum(tool_times) / len(tool_times) if tool_times else 0.0
            
            baseline_times = [
                r.baseline_time_ms
                for r in results
                if r.baseline_time_ms is not None and r.baseline_time_ms > 0
            ]
            avg_baseline_time = (
                sum(baseline_times) / len(baseline_times)
                if baseline_times
                else None
            )
            
            # Improvement ratio
            if baseline_accuracy is None:
                improvement_ratio = None
            elif baseline_accuracy > 0:
                improvement_ratio = answer_accuracy / baseline_accuracy
            else:
                improvement_ratio = float('inf')
            
            metrics[domain] = DomainMetrics(
                domain=domain,
                total_queries=total,
                domain_detection_accuracy=domain_detection_accuracy,
                tool_success_rate=tool_success_rate,
                answer_accuracy=answer_accuracy,
                baseline_accuracy=baseline_accuracy,
                avg_tool_time_ms=avg_tool_time,
                avg_baseline_time_ms=avg_baseline_time,
                improvement_ratio=improvement_ratio,
            )
        
        return metrics
    
    def generate_report(self, metrics: Dict[str, DomainMetrics]) -> str:
        """
        Generate markdown report of evaluation results.
        
        Args:
            metrics: Aggregated metrics per domain
            
        Returns:
            Formatted markdown report
        """
        report_lines = [
            "# FluxEM + Qwen3-4B Tool-Calling Benchmark Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]

        def _fmt_pct(value: Optional[float]) -> str:
            return f"{value:5.1f}%" if value is not None else "  N/A "

        def _fmt_time(value: Optional[float]) -> str:
            return f"{value:7.1f}" if value is not None else "   N/A "

        def _fmt_improvement(value: Optional[float]) -> str:
            if value is None:
                return "  N/A "
            if value == float("inf"):
                return "  inf"
            return f"{value:5.1f}x"

        # Calculate overall statistics
        total_queries = sum(m.total_queries for m in metrics.values())
        total_correct = sum(
            m.answer_accuracy * m.total_queries / 100 for m in metrics.values()
        )
        overall_accuracy = total_correct / total_queries * 100 if total_queries else 0.0
        baseline_totals = [
            (m.baseline_accuracy, m.total_queries)
            for m in metrics.values()
            if m.baseline_accuracy is not None
        ]
        if baseline_totals and total_queries:
            baseline_total = sum(acc * total / 100 for acc, total in baseline_totals)
            overall_baseline_accuracy = baseline_total / total_queries * 100
        else:
            overall_baseline_accuracy = None

        # Average metrics
        avg_tool_time = sum(m.avg_tool_time_ms for m in metrics.values()) / len(metrics)
        baseline_times = [
            m.avg_baseline_time_ms
            for m in metrics.values()
            if m.avg_baseline_time_ms is not None
        ]
        avg_baseline_time = (
            sum(baseline_times) / len(baseline_times) if baseline_times else None
        )
        improvement_values = [
            m.improvement_ratio
            for m in metrics.values()
            if m.improvement_ratio not in (None, float("inf"))
        ]
        overall_improvement = (
            sum(improvement_values) / len(improvement_values)
            if improvement_values
            else None
        )

        report_lines.extend(
            [
                f"- **Total Queries**: {total_queries}",
                f"- **Overall Accuracy (Tool-Calling)**: {overall_accuracy:.1f}%",
                f"- **Overall Accuracy (Baseline)**: {_fmt_pct(overall_baseline_accuracy)}",
                f"- **Average Tool Time**: {avg_tool_time:.2f}ms",
                f"- **Average Baseline Time**: {_fmt_time(avg_baseline_time)}ms",
                f"- **Average Improvement Ratio**: {_fmt_improvement(overall_improvement)}",
                "",
                "## Results by Domain",
                "",
                "| Domain | Queries | Domain Detection | Tool Success | Tool Accuracy | Baseline Accuracy | Tool Time (ms) | Baseline Time (ms) | Improvement |",
                "|---------|---------|------------------|--------------|---------------|------------------|---------------|-------------------|-------------|",
            ]
        )
        
        # Domain-specific results
        sorted_domains = sorted(metrics.keys())
        for domain in sorted_domains:
            m = metrics[domain]
            report_lines.append(
                f"| {domain:15s} | {m.total_queries:2d} | {m.domain_detection_accuracy:5.1f}% | {m.tool_success_rate:5.1f}% | {m.answer_accuracy:5.1f}% | {_fmt_pct(m.baseline_accuracy)} | {m.avg_tool_time_ms:7.1f} | {_fmt_time(m.avg_baseline_time_ms)} | {_fmt_improvement(m.improvement_ratio)} |"
            )
        
        # Add summary notes
        report_lines.extend([
            "",
            "## Notes",
            "",
            "- **Domain Detection**: Percentage of correctly identified domains",
            "- **Tool Success**: Percentage of successful tool executions",
            "- **Tool Accuracy**: Percentage of correct answers from tool results",
            "- **Baseline Accuracy**: Percentage of correct answers from baseline responses",
            "- **Tool Time**: Average time for tool execution",
            "- **Baseline Time**: Average time for baseline LLM response",
            "- **Improvement**: Ratio of tool-calling accuracy to baseline accuracy",
            "",
            "## Interpretation",
            "",
            "- Improvement > 2.0x: Significant improvement",
            "- Improvement 1.0x-2.0x: Moderate improvement",
            "- Improvement < 1.0x: Minimal or no improvement",
            "",
        ])
        
        return "\n".join(report_lines)
    
    def save_results(self, output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save results (without extension)
        """
        import json
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {
                "prompt": result.prompt,
                "domain": result.domain,
                "expected_domain": result.expected_domain,
                "domain_correct": result.domain_correct,
                "tool_detected": result.tool_detected,
                "tool_name": result.tool_name,
                "tool_success": result.tool_success,
                "tool_result": str(result.tool_result) if not isinstance(result.tool_result, str) else result.tool_result,
                "baseline_response": result.baseline_response,
                "baseline_correct": result.baseline_correct,
                "answer_correct": result.answer_correct,
                "answer_expected": str(result.answer_expected) if result.answer_expected is not None else None,
                "tool_execution_time_ms": result.tool_execution_time_ms,
                "baseline_time_ms": result.baseline_time_ms,
                "total_time_ms": result.total_time_ms,
            }
            serializable_results.append(serializable_result)
        
        # Save to JSON
        json_path = f"{output_path}_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_queries": len(self.results),
                "results": serializable_results,
            }, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to: {json_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.results:
            return {
                "total_queries": 0,
                "overall_accuracy": 0.0,
                "domains_tested": 0,
            }
        
        metrics = self.aggregate_metrics()
        
        total_queries = sum(m.total_queries for m in metrics.values())
        total_correct = sum(m.answer_accuracy * m.total_queries / 100 for m in metrics.values())
        overall_accuracy = total_correct / total_queries * 100
        
        return {
            "total_queries": total_queries,
            "overall_accuracy": overall_accuracy,
            "overall_baseline_accuracy": overall_baseline_accuracy,
            "domains_tested": len(metrics),
            "domain_metrics": metrics,
        }


def create_evaluator(verbose: bool = False) -> Evaluator:
    """
    Create an evaluator instance.
    
    Args:
        verbose: Print detailed evaluation information
        
    Returns:
        Evaluator instance
    """
    return Evaluator(verbose=verbose)


if __name__ == "__main__":
    # Demo evaluator
    print("FluxEM Evaluator Demo")
    print("=" * 50)
    
    evaluator = create_evaluator(verbose=True)
    
    # Example: Add some test results
    from .benchmark_data import BENCHMARK_PROMPTS
    
    # Arithmetic tests
    evaluator.add_result(EvaluationResult(
        prompt="What is 54 * 44?",
        domain="arithmetic",
        expected_domain="arithmetic",
        tool_detected=True,
        tool_name="arithmetic",
        tool_success=True,
        tool_result=2376.0,
        tool_execution_time_ms=23.5,
        baseline_response="The answer is 2376",
        baseline_time_ms=45.2,
        answer_correct=True,
        answer_expected=2376.0,
        total_time_ms=68.7,
    ))
    
    evaluator.add_result(EvaluationResult(
        prompt="What is 54 * 44?",
        domain="arithmetic",
        expected_domain="arithmetic",
        tool_detected=True,
        tool_name="arithmetic",
        tool_success=True,
        tool_result=2376.0,
        tool_execution_time_ms=18.2,
        baseline_response="I don't know how to calculate that",
        baseline_time_ms=38.1,
        answer_correct=True,
        answer_expected=2376.0,
        total_time_ms=56.3,
    ))
    
    # Generate report
    metrics = evaluator.aggregate_metrics()
    report = evaluator.generate_report(metrics)
    print(report)
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(evaluator.get_summary())
