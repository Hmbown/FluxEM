"""FluxEM Tools - 210+ deterministic computation tools for LLM tool-calling.

This package provides a comprehensive collection of deterministic tools across
40+ domains including mathematics, physics, chemistry, biology, engineering,
and everyday practical calculations.

Quick Start:
    >>> from fluxem_tools import get_registry, call_tool
    >>> registry = get_registry()
    >>> print(f"Available tools: {len(registry)}")
    >>> result = call_tool("arithmetic", "2 + 2 * 3")
    >>> print(result)  # 8

For LLM Integration:
    >>> # OpenAI format
    >>> tools = registry.to_openai_tools()
    >>> # Anthropic/Claude format
    >>> tools = registry.to_anthropic_tools()
"""

__version__ = "0.1.0"

from .registry import (
    ToolSpec,
    ToolRegistry,
    get_registry,
    register_tool,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "ToolSpec",
    "ToolRegistry",
    # Functions
    "get_registry",
    "register_tool",
    "call_tool",
    "list_tools",
    "list_domains",
    "get_tool",
    "search_tools",
    # Export functions
    "to_openai_tools",
    "to_anthropic_tools",
    "export_json",
]


def call_tool(name: str, *args, **kwargs):
    """Call a tool by name.

    Args:
        name: Tool name (e.g., "arithmetic", "electrical_ohms_law")
        *args: Positional arguments to pass to the tool
        **kwargs: Keyword arguments to pass to the tool

    Returns:
        The result of the tool execution

    Example:
        >>> call_tool("arithmetic", "2 + 2")
        4
        >>> call_tool("electrical_ohms_law", voltage=12, current=2)
        6.0
    """
    registry = get_registry()
    if args and not kwargs:
        # Single positional argument
        if len(args) == 1:
            return registry.call(name, args[0])
        return registry.call(name, list(args))
    elif kwargs:
        return registry.call(name, kwargs)
    else:
        return registry.call(name, None)


def list_tools() -> list:
    """List all available tool names.

    Returns:
        List of tool names

    Example:
        >>> tools = list_tools()
        >>> print(len(tools))  # 242
    """
    return get_registry().list_tools()


def list_domains() -> list:
    """List all available domains.

    Returns:
        List of domain names

    Example:
        >>> domains = list_domains()
        >>> print(domains)  # ['arithmetic', 'physics', 'electrical', ...]
    """
    return get_registry().list_domains()


def get_tool(name: str) -> ToolSpec:
    """Get a tool specification by name.

    Args:
        name: Tool name

    Returns:
        ToolSpec object with name, description, parameters, etc.

    Raises:
        KeyError: If tool not found
    """
    tool = get_registry().get(name)
    if tool is None:
        raise KeyError(f"Tool '{name}' not found")
    return tool


def search_tools(query: str) -> list:
    """Search for tools by name, description, or tags.

    Args:
        query: Search string (case-insensitive)

    Returns:
        List of matching ToolSpec objects

    Example:
        >>> results = search_tools("voltage")
        >>> for tool in results:
        ...     print(tool.name)
    """
    return get_registry().search(query)


def to_openai_tools() -> list:
    """Export all tools in OpenAI function calling format.

    Returns:
        List of tool schemas in OpenAI format

    Example:
        >>> import openai
        >>> tools = to_openai_tools()
        >>> response = openai.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[...],
        ...     tools=tools
        ... )
    """
    return get_registry().to_openai_tools()


def to_anthropic_tools() -> list:
    """Export all tools in Anthropic/Claude format.

    Returns:
        List of tool schemas in Anthropic format

    Example:
        >>> import anthropic
        >>> tools = to_anthropic_tools()
        >>> response = anthropic.messages.create(
        ...     model="claude-3-opus-20240229",
        ...     messages=[...],
        ...     tools=tools
        ... )
    """
    return get_registry().to_anthropic_tools()


def export_json(filepath: str) -> None:
    """Export full tool registry to JSON file.

    Args:
        filepath: Path to write JSON file

    Example:
        >>> export_json("tools.json")
    """
    get_registry().export_json(filepath)
