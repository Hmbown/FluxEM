"""
Interactive session for FluxEM + Qwen3 tool calling.

Run a multi-turn loop that exercises tool routing, parsing,
and context-aware follow-ups.
"""

import argparse

from .qwen3_wrapper import create_wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FluxEM + Qwen3 Tool-Calling Interactive Session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to Qwen3-4B MLX model",
    )
    parser.add_argument(
        "--tool-selection",
        choices=["pattern", "llm", "hybrid"],
        default="pattern",
        help="Tool selection strategy (default: pattern)",
    )
    parser.add_argument(
        "--response-style",
        choices=["structured", "plain"],
        default="structured",
        help="Response formatting style (default: structured)",
    )
    parser.add_argument(
        "--llm-query-extraction",
        action="store_true",
        help="Use LLM to refine tool queries when LLM routing is enabled",
    )
    parser.add_argument(
        "--show-tool-info",
        action="store_true",
        help="Print tool metadata for each turn",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output",
    )
    return parser.parse_args()


def _format_tool_info(result: dict) -> str:
    tool_name = result.get("tool_name") or "none"
    domain = result.get("domain") or "none"
    success = result.get("tool_success")
    success_label = "ok" if success else "fail"
    time_ms = result.get("execution_time_ms") or 0.0
    return f"[tool] domain={domain} tool={tool_name} status={success_label} time={time_ms:.1f}ms"


def main() -> None:
    args = parse_args()
    wrapper = create_wrapper(
        model_path=args.model_path,
        tool_selection=args.tool_selection,
        llm_query_extraction=args.llm_query_extraction,
        response_style=args.response_style,
        verbose=args.verbose,
    )

    if args.model_path:
        loaded = wrapper.load_model()
        if not loaded:
            print("Warning: model failed to load; using pattern-based routing.")

    print("FluxEM Tool-Calling Session")
    print("Type :reset to clear context, :quit to exit.\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {":quit", ":exit", "quit", "exit"}:
            print("Exiting session.")
            break

        if lowered == ":reset":
            wrapper.reset_context()
            print("Context cleared.")
            continue

        result = wrapper.generate_with_tools(user_input)
        print(result.get("response", ""))
        if args.show_tool_info:
            print(_format_tool_info(result))


if __name__ == "__main__":
    main()
