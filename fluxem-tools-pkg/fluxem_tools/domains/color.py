"""Color domain - color space conversions, mixing.

This module provides deterministic color computations using the perceptually
uniform OKLab color space.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def srgb_to_linear(c: float) -> float:
    """Convert sRGB component (0-1) to linear RGB."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    """Convert linear RGB component to sRGB (0-1)."""
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * (c ** (1 / 2.4)) - 0.055


def rgb_to_oklab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert sRGB (0-255) to OKLab color space.

    Args:
        r, g, b: sRGB values (0-255)

    Returns:
        (L, a, b) where L is lightness (0-1), a is green-red, b is blue-yellow
    """
    # Convert to linear RGB
    r_lin = srgb_to_linear(r / 255)
    g_lin = srgb_to_linear(g / 255)
    b_lin = srgb_to_linear(b / 255)

    # RGB to LMS (approximate)
    l = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin
    m = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin
    s = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin

    # Cube root
    l_ = l ** (1/3) if l >= 0 else -((-l) ** (1/3))
    m_ = m ** (1/3) if m >= 0 else -((-m) ** (1/3))
    s_ = s ** (1/3) if s >= 0 else -((-s) ** (1/3))

    # LMS to OKLab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return (round(L, 4), round(a, 4), round(b_out, 4))


def oklab_to_rgb(L: float, a: float, b: float) -> Tuple[int, int, int]:
    """Convert OKLab to sRGB (0-255).

    Args:
        L: Lightness (0-1)
        a: Green-red axis
        b: Blue-yellow axis

    Returns:
        (r, g, b) as integers 0-255
    """
    # OKLab to LMS
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # Cube
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    # LMS to linear RGB
    r_lin = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    # Convert to sRGB and clamp
    r = int(max(0, min(255, round(linear_to_srgb(r_lin) * 255))))
    g = int(max(0, min(255, round(linear_to_srgb(g_lin) * 255))))
    b = int(max(0, min(255, round(linear_to_srgb(b_lin) * 255))))

    return (r, g, b)


def color_mix(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int],
              ratio: float = 0.5) -> Tuple[int, int, int]:
    """Mix two colors in OKLab space (perceptually uniform mixing).

    Args:
        rgb1: First color (r, g, b)
        rgb2: Second color (r, g, b)
        ratio: Mix ratio (0 = all rgb1, 1 = all rgb2)

    Returns:
        Mixed color as (r, g, b)
    """
    lab1 = rgb_to_oklab(*rgb1)
    lab2 = rgb_to_oklab(*rgb2)

    L = lab1[0] * (1 - ratio) + lab2[0] * ratio
    a = lab1[1] * (1 - ratio) + lab2[1] * ratio
    b = lab1[2] * (1 - ratio) + lab2[2] * ratio

    return oklab_to_rgb(L, a, b)


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_str: Hex color '#RRGGBB' or 'RRGGBB'

    Returns:
        (r, g, b) tuple
    """
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join(c * 2 for c in hex_str)
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color string.

    Args:
        r, g, b: RGB values (0-255)

    Returns:
        Hex string '#rrggbb'
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def color_contrast(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """Calculate contrast ratio between two colors (WCAG formula).

    Args:
        rgb1: First color
        rgb2: Second color

    Returns:
        Contrast ratio (1 to 21)
    """
    def relative_luminance(r, g, b):
        r_lin = srgb_to_linear(r / 255)
        g_lin = srgb_to_linear(g / 255)
        b_lin = srgb_to_linear(b / 255)
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    L1 = relative_luminance(*rgb1)
    L2 = relative_luminance(*rgb2)

    lighter = max(L1, L2)
    darker = min(L1, L2)

    return (lighter + 0.05) / (darker + 0.05)


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_rgb(args) -> Tuple[int, int, int]:
    if isinstance(args, dict):
        r = int(args.get("r", args.get("red", 0)))
        g = int(args.get("g", args.get("green", 0)))
        b = int(args.get("b", args.get("blue", 0)))
        return (r, g, b)
    if isinstance(args, str):
        if ',' in args:
            parts = args.split(',')
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        return hex_to_rgb(args)
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return (int(args[0]), int(args[1]), int(args[2]))
    raise ValueError(f"Cannot parse RGB: {args}")


def _parse_oklab(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        L = float(args.get("L", args.get("l", args.get("lightness", 0))))
        a = float(args.get("a", 0))
        b = float(args.get("b", 0))
        return (L, a, b)
    if isinstance(args, str) and ',' in args:
        parts = args.split(',')
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return (float(args[0]), float(args[1]), float(args[2]))
    raise ValueError(f"Cannot parse OKLab: {args}")


def _parse_mix_args(args):
    if isinstance(args, dict):
        c1 = _parse_rgb(args.get("color1", args.get("c1")))
        c2 = _parse_rgb(args.get("color2", args.get("c2")))
        ratio = float(args.get("ratio", 0.5))
        return color_mix(c1, c2, ratio)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        c1 = _parse_rgb(args[0])
        c2 = _parse_rgb(args[1])
        ratio = float(args[2]) if len(args) > 2 else 0.5
        return color_mix(c1, c2, ratio)
    raise ValueError(f"Cannot parse mix args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register color tools in the registry."""

    registry.register(ToolSpec(
        name="color_rgb_to_oklab",
        function=lambda args: rgb_to_oklab(*_parse_rgb(args)),
        description="Converts RGB color (0-255) to OKLab perceptually uniform color space.",
        parameters={
            "type": "object",
            "properties": {
                "r": {"type": "integer", "description": "Red (0-255)"},
                "g": {"type": "integer", "description": "Green (0-255)"},
                "b": {"type": "integer", "description": "Blue (0-255)"},
            },
            "required": ["r", "g", "b"]
        },
        returns="(L, a, b) tuple where L is lightness",
        examples=[
            {"input": {"r": 255, "g": 128, "b": 0}, "output": [0.79, 0.06, 0.16]},
        ],
        domain="color",
        tags=["oklab", "rgb", "convert"],
    ))

    registry.register(ToolSpec(
        name="color_oklab_to_rgb",
        function=lambda args: oklab_to_rgb(*_parse_oklab(args)),
        description="Converts OKLab color to sRGB (0-255).",
        parameters={
            "type": "object",
            "properties": {
                "L": {"type": "number", "description": "Lightness (0-1)"},
                "a": {"type": "number", "description": "Green-red axis"},
                "b": {"type": "number", "description": "Blue-yellow axis"},
            },
            "required": ["L", "a", "b"]
        },
        returns="(r, g, b) tuple with values 0-255",
        examples=[
            {"input": {"L": 0.79, "a": 0.06, "b": 0.16}, "output": [255, 128, 0]},
        ],
        domain="color",
        tags=["oklab", "rgb", "convert"],
    ))

    registry.register(ToolSpec(
        name="color_mix",
        function=_parse_mix_args,
        description="Mixes two colors in perceptually uniform OKLab space.",
        parameters={
            "type": "object",
            "properties": {
                "color1": {"type": "array", "description": "[r, g, b] or hex"},
                "color2": {"type": "array", "description": "[r, g, b] or hex"},
                "ratio": {"type": "number", "description": "Mix ratio (0=color1, 1=color2, default=0.5)"},
            },
            "required": ["color1", "color2"]
        },
        returns="Mixed color as (r, g, b)",
        examples=[
            {"input": {"color1": [255, 0, 0], "color2": [0, 0, 255], "ratio": 0.5}, "output": [188, 0, 190]},
        ],
        domain="color",
        tags=["mix", "blend", "oklab"],
    ))

    registry.register(ToolSpec(
        name="color_hex_to_rgb",
        function=lambda args: hex_to_rgb(str(args) if not isinstance(args, dict) else str(args.get("hex", list(args.values())[0]))),
        description="Converts hex color string to RGB tuple.",
        parameters={
            "type": "object",
            "properties": {
                "hex": {"type": "string", "description": "Hex color '#RRGGBB' or 'RRGGBB'"}
            },
            "required": ["hex"]
        },
        returns="(r, g, b) tuple",
        examples=[
            {"input": {"hex": "#FF8000"}, "output": [255, 128, 0]},
        ],
        domain="color",
        tags=["hex", "rgb", "convert"],
    ))

    registry.register(ToolSpec(
        name="color_rgb_to_hex",
        function=lambda args: rgb_to_hex(*_parse_rgb(args)),
        description="Converts RGB values to hex color string.",
        parameters={
            "type": "object",
            "properties": {
                "r": {"type": "integer", "description": "Red (0-255)"},
                "g": {"type": "integer", "description": "Green (0-255)"},
                "b": {"type": "integer", "description": "Blue (0-255)"},
            },
            "required": ["r", "g", "b"]
        },
        returns="Hex color string '#rrggbb'",
        examples=[
            {"input": {"r": 255, "g": 128, "b": 0}, "output": "#ff8000"},
        ],
        domain="color",
        tags=["rgb", "hex", "convert"],
    ))
