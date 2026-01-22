"""Integration modules for various LLM APIs.

This module provides helpers for integrating FluxEM tools with different
LLM APIs and frameworks.

Supported integrations:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - HuggingFace Transformers
"""

from typing import Any, Dict, List

__all__ = [
    "format_for_openai",
    "format_for_anthropic",
    "format_for_transformers",
    "parse_tool_call",
]


def format_for_openai(tools: List[Any]) -> List[Dict[str, Any]]:
    """Format tools for OpenAI API.

    Args:
        tools: List of ToolSpec objects or the registry

    Returns:
        List of tool schemas in OpenAI format
    """
    from ..registry import ToolSpec, ToolRegistry

    if isinstance(tools, ToolRegistry):
        return tools.to_openai_tools()

    return [
        t.to_openai_schema() if isinstance(t, ToolSpec) else t
        for t in tools
    ]


def format_for_anthropic(tools: List[Any]) -> List[Dict[str, Any]]:
    """Format tools for Anthropic Claude API.

    Args:
        tools: List of ToolSpec objects or the registry

    Returns:
        List of tool schemas in Anthropic format
    """
    from ..registry import ToolSpec, ToolRegistry

    if isinstance(tools, ToolRegistry):
        return tools.to_anthropic_tools()

    return [
        t.to_anthropic_schema() if isinstance(t, ToolSpec) else t
        for t in tools
    ]


def format_for_transformers(tools: List[Any]) -> List[Dict[str, Any]]:
    """Format tools for HuggingFace Transformers.

    This returns tools in the format expected by models that support
    tool/function calling (like Qwen, Llama, etc.).

    Args:
        tools: List of ToolSpec objects or the registry

    Returns:
        List of tool schemas
    """
    # HuggingFace typically uses a format similar to OpenAI
    return format_for_openai(tools)


def parse_tool_call(response: str) -> Dict[str, Any]:
    """Parse a tool call from LLM response text.

    This handles various formats:
    - OpenAI: {"name": "tool", "arguments": {...}}
    - Anthropic: <tool_use name="tool"><input>...</input></tool_use>
    - Raw JSON: {"tool_name": "...", "args": {...}}

    Args:
        response: The LLM response text containing a tool call

    Returns:
        Dict with "name" and "arguments" keys

    Raises:
        ValueError: If no valid tool call found
    """
    import json
    import re

    # Try JSON format first
    json_match = re.search(r'\{[^{}]*"(?:name|tool_name)"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            name = data.get("name") or data.get("tool_name")
            args = data.get("arguments") or data.get("args") or data.get("query")
            if name:
                return {"name": name, "arguments": args}
        except json.JSONDecodeError:
            pass

    # Try Anthropic XML format
    xml_match = re.search(r'<tool_use\s+name="([^"]+)"[^>]*>.*?<input>(.*?)</input>', response, re.DOTALL)
    if xml_match:
        name = xml_match.group(1)
        try:
            args = json.loads(xml_match.group(2))
        except:
            args = xml_match.group(2)
        return {"name": name, "arguments": args}

    # Try <tool_call> format (Qwen style)
    tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if tool_call_match:
        try:
            data = json.loads(tool_call_match.group(1))
            name = data.get("name") or data.get("tool_name")
            args = data.get("arguments") or data.get("args")
            if name:
                return {"name": name, "arguments": args}
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse tool call from response: {response[:200]}...")
