"""Tool registry for FluxEM deterministic computation tools.

This module provides the core infrastructure for registering and calling tools.
All tools are deterministic - given the same input, they always return the same output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ToolSpec:
    """Specification for a callable tool.

    Attributes:
        name: Unique identifier for the tool (e.g., "electrical_ohms_law")
        function: The Python function that implements the tool
        description: Human-readable description of what the tool does
        parameters: JSON Schema describing the input parameters
        returns: Description of what the tool returns
        examples: List of example inputs and outputs
        domain: Category/domain this tool belongs to (e.g., "electrical")
        tags: Additional tags for search/filtering
    """
    name: str
    function: Callable[..., Any]
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    domain: str = ""
    tags: List[str] = field(default_factory=list)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Convert to Anthropic/Claude tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters or {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "examples": self.examples,
            "domain": self.domain,
            "tags": self.tags,
        }


class ToolRegistry:
    """Registry for managing and calling tools.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(ToolSpec(
        ...     name="add",
        ...     function=lambda a, b: a + b,
        ...     description="Add two numbers"
        ... ))
        >>> registry.call("add", {"a": 1, "b": 2})
        3
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._domains: Dict[str, List[str]] = {}  # domain -> list of tool names

    def register(self, tool: ToolSpec) -> None:
        """Register a tool in the registry.

        Args:
            tool: The ToolSpec to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

        # Track by domain
        if tool.domain:
            if tool.domain not in self._domains:
                self._domains[tool.domain] = []
            self._domains[tool.domain].append(tool.name)

    def get(self, name: str) -> Optional[ToolSpec]:
        """Get a tool by name.

        Args:
            name: The tool name

        Returns:
            The ToolSpec if found, None otherwise
        """
        return self._tools.get(name)

    def call(self, name: str, args: Optional[Union[Dict, List, Any]] = None) -> Any:
        """Call a tool by name with the given arguments.

        Args:
            name: The tool name
            args: Arguments to pass to the tool (passed directly to the tool function)

        Returns:
            The result of calling the tool

        Raises:
            KeyError: If the tool is not found
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())[:10]}...")

        if args is None:
            return tool.function()
        # Pass args directly - tools handle their own parsing
        return tool.function(args)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_domains(self) -> List[str]:
        """List all domains."""
        return list(self._domains.keys())

    def get_domain_tools(self, domain: str) -> List[str]:
        """Get all tool names in a domain."""
        return self._domains.get(domain, [])

    def search(self, query: str) -> List[ToolSpec]:
        """Search for tools by name, description, or tags.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching ToolSpecs
        """
        query = query.lower()
        results = []
        for tool in self._tools.values():
            if (query in tool.name.lower() or
                query in tool.description.lower() or
                any(query in tag.lower() for tag in tool.tags)):
                results.append(tool)
        return results

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Export all tools in OpenAI function calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Export all tools in Anthropic/Claude format."""
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    def to_json_schema(self) -> Dict[str, Any]:
        """Export full registry as JSON Schema."""
        return {
            "tools": {name: tool.to_dict() for name, tool in self._tools.items()},
            "domains": self._domains,
            "total_tools": len(self._tools),
            "total_domains": len(self._domains),
        }

    def export_json(self, filepath: str) -> None:
        """Export registry to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_json_schema(), f, indent=2, default=str)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        # Load all domains
        from . import domains
        domains.register_all(_global_registry)
    return _global_registry


def register_tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    returns: str = "",
    examples: Optional[List[Dict[str, Any]]] = None,
    domain: str = "",
    tags: Optional[List[str]] = None,
):
    """Decorator to register a function as a tool.

    Example:
        >>> @register_tool(
        ...     name="add",
        ...     description="Add two numbers",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "a": {"type": "number", "description": "First number"},
        ...             "b": {"type": "number", "description": "Second number"}
        ...         },
        ...         "required": ["a", "b"]
        ...     },
        ...     domain="arithmetic"
        ... )
        ... def add(a: float, b: float) -> float:
        ...     return a + b
    """
    def decorator(func: Callable) -> Callable:
        tool = ToolSpec(
            name=name,
            function=func,
            description=description,
            parameters=parameters or {},
            returns=returns,
            examples=examples or [],
            domain=domain,
            tags=tags or [],
        )
        # Will be registered when get_registry() is called
        func._tool_spec = tool
        return func
    return decorator
